import os
import re
import sys
import csv
import json
import math
import time
import copy
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Increase CSV field size limit safely (helps when JSON blobs are stored in CSV cells)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# HF Transformers (SELFIES-TED)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

# Shared encoders/helpers from PolyFusion
from PolyFusion.GINE import GineEncoder
from PolyFusion.SchNet import NodeSchNetWrapper
from PolyFusion.Transformer import PooledFingerprintEncoder as FingerprintEncoder
from PolyFusion.DeBERTav2 import PSMILESDebertaEncoder, build_psmiles_tokenizer

# Optional chemistry dependencies (recommended)
RDKit_AVAILABLE = False
SELFIES_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

try:
    import selfies as sf
    SELFIES_AVAILABLE = True
except Exception:
    SELFIES_AVAILABLE = False


# =============================================================================
# Configuration (paths are placeholders; replace with your actual filesystem paths)
# =============================================================================

@dataclass(frozen=True)
class Config:
    # -------------------------------------------------------------------------
    # Input data and pretrained artifacts (placeholders)
    # -------------------------------------------------------------------------
    BASE_DIR: str = "/path/to/Polymer_Foundational_Model"
    POLYINFO_PATH: str = "/path/to/polyinfo_with_modalities.csv"

    # Multimodal CL checkpoint (for the fused encoder)
    PRETRAINED_MULTIMODAL_DIR: str = "/path/to/multimodal_output/best"

    # Unimodal encoder checkpoints 
    BEST_GINE_DIR: str = "/path/to/gin_output/best"
    BEST_SCHNET_DIR: str = "/path/to/schnet_output/best"
    BEST_FP_DIR: str = "/path/to/fingerprint_mlm_output/best"
    BEST_PSMILES_DIR: str = "/path/to/polybert_output/best"

    # SentencePiece model for PSMILES tokenizer (placeholder)
    SPM_MODEL_PATH: str = "/path/to/spm.model"

    # -------------------------------------------------------------------------
    # Output folders
    # -------------------------------------------------------------------------
    OUTPUT_DIR: str = "/path/to/multimodal_inverse_design_output"

    @property
    def OUTPUT_RESULTS(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "inverse_design_results.txt")

    @property
    def OUTPUT_MODELS_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "best_models")

    @property
    def OUTPUT_GENERATIONS_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "best_fold_generations")


CFG = Config()

# Properties to run
REQUESTED_PROPERTIES = [
    "density",
    "glass transition",
    "melting",
    "thermal decomposition",
]

# -------------------------------------------------------------------------
# Model sizes / dims (match CL encoder + pretraining)
# -------------------------------------------------------------------------
CL_EMB_DIM = 600

MAX_ATOMIC_Z = 85
MASK_ATOM_ID = MAX_ATOMIC_Z + 1

# GINE params
NODE_EMB_DIM = 300
EDGE_EMB_DIM = 300
NUM_GNN_LAYERS = 5

# SchNet params
SCHNET_NUM_GAUSSIANS = 50
SCHNET_NUM_INTERACTIONS = 6
SCHNET_CUTOFF = 10.0
SCHNET_MAX_NEIGHBORS = 64
SCHNET_HIDDEN = 600

# Fingerprint params
FP_LENGTH = 2048
MASK_TOKEN_ID_FP = 2
VOCAB_SIZE_FP = 3

# DeBERTa params
DEBERTA_HIDDEN = 600
PSMILES_MAX_LEN = 128

# SELFIES-TED generation limits
GEN_MAX_LEN = 256
GEN_MIN_LEN = 10

# -------------------------------------------------------------------------
# Decoder fine-tuning schedule (single head)
# -------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 10
WEIGHT_DECAY = 0.0
LEARNING_RATE = 1e-4
COSINE_ETA_MIN = 1e-6

# Noise injection (latent space)
LATENT_NOISE_STD_TRAIN = 0.10  # training-time denoising std
LATENT_NOISE_STD_GEN = 0.15    # generation-time exploration std
N_FOLD_NOISE_SAMPLING = 16     # sampling multiplicity around each seed embedding

# Sampling config (decoder)
GEN_TOP_P = 0.92
GEN_TEMPERATURE = 1.0
GEN_REPETITION_PENALTY = 1.05

# Cross-validation
NUM_FOLDS = 5

# Property guidance tolerance (scaled space)
PROP_TOL_SCALED = 0.5
PROP_TOL_UNSCALED_ABS = None

# GPR settings (PSMILES latent)
USE_PCA_BEFORE_GPR = True
PCA_DIM = 64
GPR_ALPHA = 1e-6

# Verification (optional auxiliary predictor trained per fold)
VERIFY_GENERATED_PROPERTIES = True
PROP_PRED_EPOCHS = 20
PROP_PRED_PATIENCE = 5
PROP_PRED_BATCH_SIZE = 32
PROP_PRED_LR = 3e-4
PROP_PRED_WEIGHT_DECAY = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = bool(torch.cuda.is_available())
AMP_DTYPE = torch.float16
NUM_WORKERS = 0 if os.name == "nt" else 1

warnings.filterwarnings("ignore", category=UserWarning)


def ensure_output_dirs(cfg: Config) -> None:
    """Create output directories if they do not exist."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_GENERATIONS_DIR, exist_ok=True)


# =============================================================================
# Utilities
# =============================================================================
def _safe_json_load(x):
    """Robust JSON parsing for CSV cells that may contain dict/list JSON (or slightly malformed strings)."""
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return None


def set_seed(seed: int):
    """Set random seeds for reproducibility (best effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        # Keep cuDNN fast; reproducibility across GPUs/drivers is not guaranteed.
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def make_json_serializable(obj):
    """Convert common scientific objects into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return float(obj)
    if isinstance(obj, torch.Tensor):
        try:
            return obj.detach().cpu().tolist()
        except Exception:
            return None
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    try:
        if isinstance(obj, (float, int, str, bool, type(None))):
            return obj
    except Exception:
        pass
    return str(obj)


def find_property_columns(columns):
    """
    Heuristically map requested property names to dataframe columns.
    Notes:
      - Uses lowercase matching; prefers token-level matches.
      - Special-case: exclude "cohesive energy" when searching for "density".
    """
    lowered = {c.lower(): c for c in columns}
    found = {}

    for req in REQUESTED_PROPERTIES:
        req_low = req.lower().strip()
        exact = None

        # First, attempt a token match
        for c_low, c_orig in lowered.items():
            tokens = set(c_low.replace('_', ' ').split())
            if req_low in tokens or c_low == req_low:
                if req_low == "density" and ("cohesive" in c_low or "cohesive energy" in c_low):
                    continue
                exact = c_orig
                break
        if exact is not None:
            found[req] = exact
            continue

        # Fallback: substring match
        candidates = [c_orig for c_low, c_orig in lowered.items() if req_low in c_low]
        if req_low == "density":
            candidates = [c for c in candidates if "cohesive" not in c.lower() and "cohesive energy" not in c.lower()]

        chosen = candidates[0] if candidates else None
        found[req] = chosen

        if chosen is None:
            print(f"[WARN] Could not match requested property '{req}' to any column.")
        else:
            print(f"[INFO] Mapped requested property '{req}' -> column '{chosen}'")

    return found


# =============================================================================
# Graph / geometry / fingerprint parsing for multimodal CL encoding
# =============================================================================
def _parse_graph_for_gine(graph_field):
    """
    Convert a stored 'graph' JSON blob into the tensor inputs expected by GineEncoder.
    Returns None if graph is missing or malformed.
    """
    gf = _safe_json_load(graph_field)
    if not isinstance(gf, dict):
        return None

    node_features = gf.get("node_features", None)
    if not node_features or not isinstance(node_features, list):
        return None

    atomic_nums, chirality_vals, formal_charges = [], [], []
    for nf in node_features:
        if not isinstance(nf, dict):
            continue
        an = nf.get("atomic_num", nf.get("atomic_number", 0))
        ch = nf.get("chirality", 0)
        fc = nf.get("formal_charge", 0)
        try:
            atomic_nums.append(int(an))
        except Exception:
            atomic_nums.append(0)
        try:
            chirality_vals.append(float(ch))
        except Exception:
            chirality_vals.append(0.0)
        try:
            formal_charges.append(float(fc))
        except Exception:
            formal_charges.append(0.0)

    if len(atomic_nums) == 0:
        return None

    edge_indices_raw = gf.get("edge_indices", None)
    edge_features_raw = gf.get("edge_features", None)

    srcs, dsts = [], []

    # Handle two common representations:
    #   (a) edge_indices = [[u,v], [u,v], ...]
    #   (b) edge_indices = [[srcs...], [dsts...]]
    if edge_indices_raw is None:
        adj = gf.get("adjacency_matrix", None)
        if isinstance(adj, list):
            for i_r, row_adj in enumerate(adj):
                if not isinstance(row_adj, list):
                    continue
                for j, val in enumerate(row_adj):
                    if val:
                        srcs.append(i_r)
                        dsts.append(j)
    else:
        try:
            if isinstance(edge_indices_raw, list) and len(edge_indices_raw) > 0:
                if isinstance(edge_indices_raw[0], list) and len(edge_indices_raw[0]) == 2:
                    srcs = [int(p[0]) for p in edge_indices_raw]
                    dsts = [int(p[1]) for p in edge_indices_raw]
                elif len(edge_indices_raw) == 2:
                    srcs = [int(x) for x in edge_indices_raw[0]]
                    dsts = [int(x) for x in edge_indices_raw[1]]
        except Exception:
            srcs, dsts = [], []

    if len(srcs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
        return {
            "z": torch.tensor(atomic_nums, dtype=torch.long),
            "chirality": torch.tensor(chirality_vals, dtype=torch.float),
            "formal_charge": torch.tensor(formal_charges, dtype=torch.float),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

    # Edge attributes (bond_type, stereo, is_conjugated) if present; else zeros
    if isinstance(edge_features_raw, list) and len(edge_features_raw) == len(srcs):
        bt, st, ic = [], [], []
        for ef in edge_features_raw:
            if isinstance(ef, dict):
                bt.append(float(ef.get("bond_type", 0)))
                st.append(float(ef.get("stereo", 0)))
                ic.append(float(1.0 if ef.get("is_conjugated", False) else 0.0))
            else:
                bt.append(0.0)
                st.append(0.0)
                ic.append(0.0)
        edge_attr = torch.tensor(list(zip(bt, st, ic)), dtype=torch.float)
    else:
        edge_attr = torch.zeros((len(srcs), 3), dtype=torch.float)

    return {
        "z": torch.tensor(atomic_nums, dtype=torch.long),
        "chirality": torch.tensor(chirality_vals, dtype=torch.float),
        "formal_charge": torch.tensor(formal_charges, dtype=torch.float),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }


def _parse_geometry_for_schnet(geom_field):
    """
    Convert stored 'geometry' JSON blob into SchNet inputs:
      - atomic_numbers -> z
      - coordinates -> pos
    Returns None if missing/malformed.
    """
    gf = _safe_json_load(geom_field)
    if not isinstance(gf, dict):
        return None

    conf = gf.get("best_conformer", None)
    if not isinstance(conf, dict):
        return None

    atomic = conf.get("atomic_numbers", [])
    coords = conf.get("coordinates", [])
    if not (isinstance(atomic, list) and isinstance(coords, list)):
        return None
    if len(atomic) == 0 or len(atomic) != len(coords):
        return None

    return {"z": torch.tensor(atomic, dtype=torch.long), "pos": torch.tensor(coords, dtype=torch.float)}


def _parse_fingerprints(fp_field, fp_len: int = 2048):
    """
    Parse a fingerprint field (either list or dict containing 'morgan_r3_bits') into:
      - input_ids: LongTensor [fp_len] with 0/1 bits
      - attention_mask: BoolTensor [fp_len] all True
    """
    fp = _safe_json_load(fp_field)
    bits = None

    if isinstance(fp, dict):
        bits = fp.get("morgan_r3_bits", None)
    elif isinstance(fp, list):
        bits = fp
    elif fp is None:
        bits = None

    if bits is None:
        bits = [0] * fp_len
    else:
        norm = []
        for b in bits[:fp_len]:
            if isinstance(b, str):
                bc = b.strip().strip('"').strip("'")
                norm.append(1 if bc in ("1", "True", "true") else 0)
            elif isinstance(b, (int, np.integer, float, np.floating)):
                norm.append(1 if int(b) != 0 else 0)
            else:
                norm.append(0)
        if len(norm) < fp_len:
            norm.extend([0] * (fp_len - len(norm)))
        bits = norm

    return {
        "input_ids": torch.tensor(bits, dtype=torch.long),
        "attention_mask": torch.ones(fp_len, dtype=torch.bool),
    }


# =============================================================================
# PSELFIES utilities (polymer-safe SELFIES encoding with endpoint markers)
# =============================================================================

_SELFIES_TOKEN_RE = re.compile(r"\[[^\[\]]+\]")

def _split_selfies_tokens(selfies_str: str) -> List[str]:
    """Split a SELFIES string into tokens; prefers selfies.split_selfies if available."""
    if not isinstance(selfies_str, str) or len(selfies_str) == 0:
        return []
    if SELFIES_AVAILABLE:
        try:
            toks = list(sf.split_selfies(selfies_str.replace(" ", "")))
            return [t for t in toks if isinstance(t, str) and t]
        except Exception:
            pass
    return _SELFIES_TOKEN_RE.findall(selfies_str)

def _selfies_for_tokenizer(selfies_str: str) -> str:
    """Normalize SELFIES formatting so the HF tokenizer sees token boundaries."""
    s = str(selfies_str).strip()
    if not s:
        return ""
    s = s.replace(" ", "")
    s = s.replace("][", "] [")
    return s

def _selfies_compact(selfies_str: str) -> str:
    """Remove spaces and trim."""
    return str(selfies_str).replace(" ", "").strip()

def _ensure_two_at_endpoints(selfies_str: str) -> str:
    """
    Ensure polymer endpoints exist: enforce exactly two [At] tokens (one at each end).
    This is used as a polymerization marker compatible with the At-based conversion.
    """
    s = _selfies_compact(selfies_str)
    toks = _split_selfies_tokens(s)
    if not toks:
        return s

    at = "[At]"
    at_pos = [i for i, t in enumerate(toks) if t == at]

    if len(at_pos) == 0:
        toks = [at] + toks + [at]
    elif len(at_pos) == 1:
        toks = toks + [at]
    elif len(at_pos) > 2:
        first = at_pos[0]
        last = at_pos[-1]
        new = []
        for i, t in enumerate(toks):
            if t == at and i not in (first, last):
                continue
            new.append(t)
        toks = new

    return "".join(toks)


def psmiles_to_at_smiles(psmiles: str, root_at: bool = True) -> Optional[str]:
    """
    Convert polymer PSMILES (two [*]) into RDKit SMILES where [*] is represented as element At (Z=85).
    This allows SELFIES encoding/decoding while preserving polymer endpoints.
    """
    if not RDKit_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(psmiles)
        if mol is None:
            return None
        mol = Chem.RWMol(mol)

        at_indices = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(85)
                try:
                    atom.SetNoImplicit(True)
                except Exception:
                    pass
                try:
                    atom.SetNumExplicitHs(0)
                except Exception:
                    pass
                try:
                    atom.SetFormalCharge(0)
                except Exception:
                    pass
                at_indices.append(int(atom.GetIdx()))

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            return None

        if root_at and len(at_indices) > 0:
            try:
                can = Chem.MolToSmiles(mol, canonical=True, rootedAtAtom=at_indices[0])
            except Exception:
                can = Chem.MolToSmiles(mol, canonical=True)
        else:
            can = Chem.MolToSmiles(mol, canonical=True)

        return can
    except Exception:
        return None


def at_smiles_to_psmiles(at_smiles: str) -> Optional[str]:
    """Inverse of psmiles_to_at_smiles: convert At (Z=85) back to polymer [*] endpoints."""
    if not RDKit_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(at_smiles)
        if mol is None:
            return None

        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 85:
                atom.SetAtomicNum(0)
                try:
                    atom.SetNoImplicit(True)
                except Exception:
                    pass
                try:
                    atom.SetNumExplicitHs(0)
                except Exception:
                    pass
                try:
                    atom.SetFormalCharge(0)
                except Exception:
                    pass

        mol2 = rw.GetMol()
        try:
            Chem.SanitizeMol(mol2, catchErrors=True)
        except Exception:
            return None

        can = Chem.MolToSmiles(mol2, canonical=True)
        can = can.replace("[*]", "*")
        return can
    except Exception:
        return None


def smiles_to_pselfies(smiles: str) -> Optional[str]:
    """Encode RDKit-canonical SMILES into SELFIES."""
    if not (RDKit_AVAILABLE and SELFIES_AVAILABLE):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        can = Chem.MolToSmiles(mol, canonical=True)
        s = sf.encoder(can)
        if not isinstance(s, str) or len(s) == 0:
            return None
        return s
    except Exception:
        return None


def psmiles_to_pselfies(psmiles: str) -> Optional[str]:
    """Convert polymer PSMILES -> At-SMILES -> PSELFIES, ensuring endpoint markers."""
    if not (RDKit_AVAILABLE and SELFIES_AVAILABLE):
        return None
    at_smiles = psmiles_to_at_smiles(psmiles, root_at=True)
    if at_smiles is None:
        return None
    s = smiles_to_pselfies(at_smiles)
    if s is None:
        return None
    return _ensure_two_at_endpoints(s)


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """Decode SELFIES -> SMILES and canonicalize with RDKit."""
    if not (RDKit_AVAILABLE and SELFIES_AVAILABLE):
        return None
    try:
        s = _selfies_compact(selfies_str)
        smi = sf.decoder(s)
        if not isinstance(smi, str) or len(smi) == 0:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            return None
        can = Chem.MolToSmiles(mol, canonical=True)
        return can
    except Exception:
        return None


def pselfies_to_psmiles(selfies_str: str) -> Optional[str]:
    """Decode PSELFIES -> At-SMILES -> polymer PSMILES."""
    if not (RDKit_AVAILABLE and SELFIES_AVAILABLE):
        return None
    at_smiles = selfies_to_smiles(selfies_str)
    if at_smiles is None:
        return None
    return at_smiles_to_psmiles(at_smiles)


def canonicalize_psmiles(psmiles: str) -> Optional[str]:
    """RDKit-canonicalize PSMILES (best effort)."""
    psmiles = str(psmiles).strip()
    if not psmiles:
        return None
    if not RDKit_AVAILABLE:
        return psmiles
    try:
        mol = Chem.MolFromSmiles(psmiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            return None
        can = Chem.MolToSmiles(mol, canonical=True)
        can = can.replace("[*]", "*")
        return can
    except Exception:
        return None


def chem_validity_psmiles(psmiles: str) -> bool:
    """Basic chemical validity check via RDKit parse + sanitize."""
    if not RDKit_AVAILABLE:
        return False
    try:
        s = str(psmiles).strip()
        if not s:
            return False
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            return False
        return True
    except Exception:
        return False


def polymer_validity_psmiles_strict(psmiles: str) -> bool:
    """
    Strict polymer validity:
      - exactly two [*] atoms
      - each [*] has degree 1 (a single attachment)
    """
    if not RDKit_AVAILABLE:
        return False
    try:
        s = str(psmiles).strip()
        if not s:
            return False
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            return False
        stars = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(stars) != 2:
            return False
        for a in stars:
            if a.GetTotalDegree() != 1:
                return False
        return True
    except Exception:
        return False


# =============================================================================
# CL encoder (multimodal) + fusion pooling
# =============================================================================
def resolve_cl_checkpoint_path(cl_weights_dir: str) -> Optional[str]:
    """Resolve a checkpoint file inside a directory (or accept a file path directly)."""
    if cl_weights_dir is None:
        return None
    if os.path.isfile(cl_weights_dir):
        return cl_weights_dir
    if not os.path.isdir(cl_weights_dir):
        return None

    candidates = [
        os.path.join(cl_weights_dir, "pytorch_model.bin"),
        os.path.join(cl_weights_dir, "model.pt"),
        os.path.join(cl_weights_dir, "best.pt"),
        os.path.join(cl_weights_dir, "state_dict.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    for ext in ("*.bin", "*.pt"):
        files = sorted(Path(cl_weights_dir).glob(ext))
        if files:
            return str(files[0])

    return None


def load_state_dict_any(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint that may wrap the model state dict under common keys."""
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"Checkpoint at {ckpt_path} did not contain a state_dict-like dict.")
    return obj


def safe_load_into_module(module: nn.Module, sd: Dict[str, torch.Tensor], strict: bool = False) -> Tuple[int, int]:
    """Load a (possibly partial) state dict and return counts of missing/unexpected keys."""
    incompatible = module.load_state_dict(sd, strict=strict)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    return len(missing), len(unexpected)


class PolyFusionModule(nn.Module):
    """
    Tiny fusion transformer:
      - self-attention over modality tokens
      - learned query pooling (attention weights -> pooled representation)
    """
    def __init__(self, d_model: int, nhead: int = 8, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.pool_ln = nn.LayerNorm(d_model)
        self.pool_q = nn.Parameter(torch.randn(d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: True for valid tokens; MultiheadAttention uses key_padding_mask where True means "ignore"
        key_padding = ~mask
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))

        # query pooling
        x = self.pool_ln(x)
        q = self.pool_q.unsqueeze(0).unsqueeze(-1)  # [1, d, 1]
        scores = torch.matmul(x, q).squeeze(-1)     # [B, T]
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (x * w).sum(dim=1)  # [B, d]
        return pooled


class MultiModalCLPolymerEncoder(nn.Module):
    """
    Frozen multimodal encoder used as the conditioning interface:
      - encodes any subset of modalities (graph/geometry/fingerprint/psmiles)
      - projects each modality into a shared CL embedding space
      - fuses available modality tokens into a single normalized vector
    """
    def __init__(
        self,
        psmiles_tokenizer,
        emb_dim: int = CL_EMB_DIM,
        cl_weights_dir: Optional[str] = CFG.PRETRAINED_MULTIMODAL_DIR,
        use_gine: bool = True,
        use_schnet: bool = True,
        use_fp: bool = True,
        use_psmiles: bool = True,
    ):
        super().__init__()
        self.psm_tok = psmiles_tokenizer
        self.emb_dim = int(emb_dim)

        self.gine = None
        self.schnet = None
        self.fp = None
        self.psmiles = None

        if use_gine:
            try:
                self.gine = GineEncoder(NODE_EMB_DIM, EDGE_EMB_DIM, NUM_GNN_LAYERS, MAX_ATOMIC_Z)
            except Exception as e:
                print(f"[CL][WARN] Disabling GINE encoder: {e}")
                self.gine = None

        if use_schnet:
            try:
                self.schnet = NodeSchNetWrapper(
                    SCHNET_HIDDEN, SCHNET_NUM_INTERACTIONS, SCHNET_NUM_GAUSSIANS, SCHNET_CUTOFF, SCHNET_MAX_NEIGHBORS
                )
            except Exception as e:
                print(f"[CL][WARN] Disabling SchNet encoder: {e}")
                self.schnet = None

        if use_fp:
            try:
                self.fp = FingerprintEncoder(VOCAB_SIZE_FP, 256, FP_LENGTH, 4, 8, 1024, 0.1)
            except Exception as e:
                print(f"[CL][WARN] Disabling fingerprint encoder: {e}")
                self.fp = None

        if use_psmiles:
            enc_src = CFG.BEST_PSMILES_DIR if (CFG.BEST_PSMILES_DIR and os.path.isdir(CFG.BEST_PSMILES_DIR)) else None
            self.psmiles = PSMILESDebertaEncoder(
                model_dir_or_name=enc_src,
                vocab_fallback=int(getattr(psmiles_tokenizer, "vocab_size", 300)),
            )

        # Projection layers into shared CL space
        self.proj_gine = nn.Linear(NODE_EMB_DIM, self.emb_dim) if self.gine is not None else None
        self.proj_schnet = nn.Linear(SCHNET_HIDDEN, self.emb_dim) if self.schnet is not None else None
        self.proj_fp = nn.Linear(256, self.emb_dim) if self.fp is not None else None
        self.proj_psmiles = nn.Linear(DEBERTA_HIDDEN, self.emb_dim) if self.psmiles is not None else None

        self.dropout = nn.Dropout(0.1)
        self.out_dim = self.emb_dim
        self.fusion = PolyFusionModule(d_model=self.emb_dim, nhead=8, ffn_mult=4, dropout=0.1)

        # Optionally load a trained multimodal CL checkpoint
        self._load_multimodal_cl_checkpoint(cl_weights_dir)

    def _load_multimodal_cl_checkpoint(self, cl_weights_dir: Optional[str]):
        ckpt_path = resolve_cl_checkpoint_path(cl_weights_dir) if cl_weights_dir else None
        if ckpt_path is None:
            print(f"[CL][INFO] No multimodal CL checkpoint found at '{cl_weights_dir}'. Using initialized weights.")
            return

        sd = load_state_dict_any(ckpt_path)
        model_sd = self.state_dict()

        # Load only compatible keys (shape match) to be robust across versions
        filtered = {}
        for k, v in sd.items():
            if k not in model_sd:
                continue
            if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and tuple(v.shape) != tuple(model_sd[k].shape):
                continue
            filtered[k] = v

        missing, unexpected = safe_load_into_module(self, filtered, strict=False)
        print(
            f"[CL][INFO] Loaded multimodal CL checkpoint '{ckpt_path}'. "
            f"loaded_keys={len(filtered)} missing={missing} unexpected={unexpected}"
        )

    def freeze_cl_encoders(self):
        """Freeze encoders and fusion module (decoder training should not update them)."""
        for name, enc in [("gine", self.gine), ("schnet", self.schnet), ("fp", self.fp), ("psmiles", self.psmiles)]:
            if enc is None:
                continue
            enc.eval()
            for p in enc.parameters():
                p.requires_grad = False
            print(f"[CL][INFO] Froze {name} encoder parameters.")

        self.fusion.eval()
        for p in self.fusion.parameters():
            p.requires_grad = False
        print("[CL][INFO] Froze fusion module parameters.")

    def forward_multimodal(self, batch_mods: dict) -> torch.Tensor:
        """Encode a batch containing any subset of modalities and return normalized CL embeddings."""
        device = next(self.parameters()).device

        # Infer batch size from whichever modality is present
        if batch_mods.get("fp", None) is not None and isinstance(batch_mods["fp"].get("input_ids", None), torch.Tensor):
            B = int(batch_mods["fp"]["input_ids"].size(0))
        elif batch_mods.get("psmiles", None) is not None and isinstance(batch_mods["psmiles"].get("input_ids", None), torch.Tensor):
            B = int(batch_mods["psmiles"]["input_ids"].size(0))
        else:
            if batch_mods.get("gine", None) is not None and isinstance(batch_mods["gine"].get("batch", None), torch.Tensor):
                B = int(batch_mods["gine"]["batch"].max().item() + 1) if batch_mods["gine"]["batch"].numel() > 0 else 1
            elif batch_mods.get("schnet", None) is not None and isinstance(batch_mods["schnet"].get("batch", None), torch.Tensor):
                B = int(batch_mods["schnet"]["batch"].max().item() + 1) if batch_mods["schnet"]["batch"].numel() > 0 else 1
            else:
                B = 1

        tokens: List[torch.Tensor] = []

        def _append_token(z_token: torch.Tensor):
            tokens.append(z_token)

        # GINE token
        if self.gine is not None and batch_mods.get("gine", None) is not None:
            g = batch_mods["gine"]
            if isinstance(g.get("z", None), torch.Tensor) and g["z"].numel() > 0:
                emb_g = self.gine(
                    g["z"].to(device),
                    g.get("chirality", torch.zeros_like(g["z"], dtype=torch.float)).to(device) if isinstance(g.get("chirality", None), torch.Tensor) else None,
                    g.get("formal_charge", torch.zeros_like(g["z"], dtype=torch.float)).to(device) if isinstance(g.get("formal_charge", None), torch.Tensor) else None,
                    g.get("edge_index", torch.empty((2, 0), dtype=torch.long)).to(device),
                    g.get("edge_attr", torch.zeros((0, 3), dtype=torch.float)).to(device),
                    g.get("batch", None).to(device) if isinstance(g.get("batch", None), torch.Tensor) else None,
                )
                zg = self.proj_gine(emb_g)
                zg = self.dropout(zg)
                _append_token(zg)

        # SchNet token
        if self.schnet is not None and batch_mods.get("schnet", None) is not None:
            s = batch_mods["schnet"]
            if isinstance(s.get("z", None), torch.Tensor) and s["z"].numel() > 0:
                emb_s = self.schnet(
                    s["z"].to(device),
                    s["pos"].to(device),
                    s.get("batch", None).to(device) if isinstance(s.get("batch", None), torch.Tensor) else None,
                )
                zs = self.proj_schnet(emb_s)
                zs = self.dropout(zs)
                _append_token(zs)

        # Fingerprint token
        if self.fp is not None and batch_mods.get("fp", None) is not None:
            f = batch_mods["fp"]
            if isinstance(f.get("input_ids", None), torch.Tensor) and f["input_ids"].numel() > 0:
                emb_f = self.fp(
                    f["input_ids"].to(device),
                    f.get("attention_mask", None).to(device) if isinstance(f.get("attention_mask", None), torch.Tensor) else None,
                )
                zf = self.proj_fp(emb_f)
                zf = self.dropout(zf)
                _append_token(zf)

        # PSMILES token
        if self.psmiles is not None and batch_mods.get("psmiles", None) is not None:
            p = batch_mods["psmiles"]
            if isinstance(p.get("input_ids", None), torch.Tensor) and p["input_ids"].numel() > 0:
                emb_p = self.psmiles(
                    p["input_ids"].to(device),
                    p.get("attention_mask", None).to(device) if isinstance(p.get("attention_mask", None), torch.Tensor) else None,
                )
                zp = self.proj_psmiles(emb_p)
                zp = self.dropout(zp)
                _append_token(zp)

        if not tokens:
            # No modalities present; return a safe zero vector
            z = torch.zeros((B, self.emb_dim), device=device)
            return F.normalize(z, dim=-1)

        X = torch.stack(tokens, dim=1)  # [B, T, d]
        mask = torch.ones((B, X.size(1)), dtype=torch.bool, device=device)
        pooled = self.fusion(X, mask)
        pooled = F.normalize(pooled, dim=-1)
        return pooled

    @torch.no_grad()
    def encode_psmiles(
        self,
        psmiles_list: List[str],
        max_len: int = PSMILES_MAX_LEN,
        batch_size: int = 64,
        device: str = DEVICE,
    ) -> np.ndarray:
        self.eval()
        if self.psm_tok is None or self.psmiles is None or self.proj_psmiles is None:
            raise RuntimeError("PSMILES tokenizer/encoder/projection not available.")

        dev = torch.device(device)
        self.to(dev)

        outs = []
        for i in range(0, len(psmiles_list), batch_size):
            chunk = [str(x) for x in psmiles_list[i : i + batch_size]]
            enc = self.psm_tok(chunk, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(dev)
            attn = enc["attention_mask"].to(dev).bool()

            emb_p = self.psmiles(input_ids, attn)
            z = self.proj_psmiles(emb_p)
            z = F.normalize(z, dim=-1)
            outs.append(z.detach().cpu().numpy())

        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.emb_dim), dtype=np.float32)

    @torch.no_grad()
    def encode_multimodal(self, records: List[dict], batch_size: int = 32, device: str = DEVICE) -> np.ndarray:
        """Encode a list of records that may contain any subset of modalities."""
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        outs = []
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]

            # PSMILES tokenization
            psmiles_texts = [str(r.get("psmiles", "")) for r in chunk]
            p_enc = None
            if self.psm_tok is not None:
                p_enc = self.psm_tok(
                    psmiles_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=PSMILES_MAX_LEN,
                    return_tensors="pt",
                )

            # Fingerprints
            fp_ids, fp_attn = [], []
            for r in chunk:
                f = _parse_fingerprints(r.get("fingerprints", None), fp_len=FP_LENGTH)
                fp_ids.append(f["input_ids"])
                fp_attn.append(f["attention_mask"])
            fp_ids = torch.stack(fp_ids, dim=0)
            fp_attn = torch.stack(fp_attn, dim=0)

            # GINE batch assembly (concat nodes; keep per-graph batch indices)
            gine_all = {"z": [], "chirality": [], "formal_charge": [], "edge_index": [], "edge_attr": [], "batch": []}
            node_offset = 0
            for bi, r in enumerate(chunk):
                g = _parse_graph_for_gine(r.get("graph", None))
                if g is None or g["z"].numel() == 0:
                    continue
                n = g["z"].size(0)
                gine_all["z"].append(g["z"])
                gine_all["chirality"].append(g["chirality"])
                gine_all["formal_charge"].append(g["formal_charge"])
                gine_all["batch"].append(torch.full((n,), bi, dtype=torch.long))
                ei = g["edge_index"]
                ea = g["edge_attr"]
                if ei is not None and ei.numel() > 0:
                    gine_all["edge_index"].append(ei + node_offset)
                    gine_all["edge_attr"].append(ea)
                node_offset += n

            gine_batch = None
            if len(gine_all["z"]) > 0:
                z_b = torch.cat(gine_all["z"], dim=0)
                ch_b = torch.cat(gine_all["chirality"], dim=0)
                fc_b = torch.cat(gine_all["formal_charge"], dim=0)
                b_b = torch.cat(gine_all["batch"], dim=0)
                if len(gine_all["edge_index"]) > 0:
                    ei_b = torch.cat(gine_all["edge_index"], dim=1)
                    ea_b = torch.cat(gine_all["edge_attr"], dim=0)
                else:
                    ei_b = torch.empty((2, 0), dtype=torch.long)
                    ea_b = torch.zeros((0, 3), dtype=torch.float)

                gine_batch = {
                    "z": z_b,
                    "chirality": ch_b,
                    "formal_charge": fc_b,
                    "edge_index": ei_b,
                    "edge_attr": ea_b,
                    "batch": b_b,
                }

            # SchNet batch assembly (concat atoms; keep per-structure batch indices)
            sch_all_z, sch_all_pos, sch_all_batch = [], [], []
            for bi, r in enumerate(chunk):
                s = _parse_geometry_for_schnet(r.get("geometry", None))
                if s is None or s["z"].numel() == 0:
                    continue
                n = s["z"].size(0)
                sch_all_z.append(s["z"])
                sch_all_pos.append(s["pos"])
                sch_all_batch.append(torch.full((n,), bi, dtype=torch.long))

            schnet_batch = None
            if len(sch_all_z) > 0:
                schnet_batch = {
                    "z": torch.cat(sch_all_z, dim=0),
                    "pos": torch.cat(sch_all_pos, dim=0),
                    "batch": torch.cat(sch_all_batch, dim=0),
                }

            batch_mods = {
                "gine": gine_batch,
                "schnet": schnet_batch,
                "fp": {"input_ids": fp_ids, "attention_mask": fp_attn},
                "psmiles": {"input_ids": p_enc["input_ids"], "attention_mask": p_enc["attention_mask"]} if p_enc is not None else None,
            }

            z = self.forward_multimodal(batch_mods)
            outs.append(z.detach().cpu().numpy())

        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.emb_dim), dtype=np.float32)


# =============================================================================
# SELFIES-TED decoder conditioned on CL embeddings
# =============================================================================
SELFIES_TED_MODEL_NAME = os.environ.get("SELFIES_TED_MODEL_NAME", "ibm-research/materials.selfies-ted")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def _hf_load_with_retries(load_fn, max_tries: int = 5, base_sleep: float = 2.0):
    """Retry wrapper for HF downloads (useful when the hub is flaky or rate-limited)."""
    last_err = None
    for t in range(max_tries):
        try:
            return load_fn()
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (1.6 ** t) + random.random()
            print(f"[HF][WARN] Load attempt {t+1}/{max_tries} failed: {e} | retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to load model from HF after {max_tries} attempts. Last error: {last_err}")


def load_selfies_ted_and_tokenizer(model_name: str = SELFIES_TED_MODEL_NAME):
    """Load SELFIES-TED tokenizer and model from Hugging Face."""
    def _load_tok():
        return AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, use_fast=True)

    def _load_model():
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)

    tok = _hf_load_with_retries(_load_tok, max_tries=5)
    model = _hf_load_with_retries(_load_model, max_tries=5)
    return tok, model


class CLConditionedSelfiesTEDGenerator(nn.Module):
    """
    Condition SELFIES-TED on CL embeddings by:
      - mapping CL vector -> d_model
      - expanding into a short "memory" sequence (mem_len)
      - passing this as encoder_outputs to the seq2seq model
    """
    def __init__(self, tok, seq2seq_model, cl_emb_dim: int = CL_EMB_DIM, mem_len: int = 4):
        super().__init__()
        self.tok = tok
        self.model = seq2seq_model
        self.mem_len = int(mem_len)

        d_model = int(getattr(self.model.config, "d_model", 1024))
        self.cl_to_d = nn.Sequential(
            nn.Linear(cl_emb_dim, d_model),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )
        self.mem_pos = nn.Embedding(self.mem_len, d_model)

    def build_encoder_outputs(self, z: torch.Tensor) -> Tuple[BaseModelOutput, torch.Tensor]:
        """Determine encoder outputs from a CL latent vector."""
        device = z.device
        B = z.size(0)

        d = self.cl_to_d(z)  # [B, d_model]
        d = d.unsqueeze(1).expand(B, self.mem_len, d.size(-1)).contiguous()

        pos = torch.arange(self.mem_len, device=device).unsqueeze(0).expand(B, -1)
        d = d + self.mem_pos(pos)

        attn = torch.ones((B, self.mem_len), dtype=torch.long, device=device)
        return BaseModelOutput(last_hidden_state=d), attn

    def forward_train(self, z: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Teacher-forced training step (labels are decoder targets)."""
        enc_out, attn = self.build_encoder_outputs(z)
        out = self.model(encoder_outputs=enc_out, attention_mask=attn, labels=labels)
        loss = out.loss
        return {"loss": loss, "ce": loss.detach()}

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        num_return_sequences: int = 1,
        max_len: int = GEN_MAX_LEN,
        top_p: float = GEN_TOP_P,
        temperature: float = GEN_TEMPERATURE,
        repetition_penalty: float = GEN_REPETITION_PENALTY,
    ) -> List[str]:
        """Stochastic decoding from a batch of CL latents."""
        self.eval()
        z = z.to(next(self.parameters()).device)
        enc_out, attn = self.build_encoder_outputs(z)

        gen = self.model.generate(
            encoder_outputs=enc_out,
            attention_mask=attn,
            do_sample=True,
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_return_sequences=int(num_return_sequences),
            max_length=int(max_len),
            min_length=int(GEN_MIN_LEN),
            pad_token_id=int(self.tok.pad_token_id) if self.tok.pad_token_id is not None else None,
            eos_token_id=int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else None,
        )

        outs = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        outs = [_ensure_two_at_endpoints(_selfies_compact(o)) for o in outs]
        return outs


def create_optimizer_and_scheduler_decoder(model: CLConditionedSelfiesTEDGenerator):
    """Create AdamW + CosineAnnealingLR for decoder fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=COSINE_ETA_MIN)
    return opt, sch


# =============================================================================
# Datasets for latent-to-SELFIES training
# =============================================================================
class LatentToPSELFIESDataset(Dataset):
    """
    Each sample:
      - z: frozen CL embedding (optionally with Gaussian noise added for denoising)
      - labels: tokenized PSELFIES target sequence (pad tokens masked as -100)
    """
    def __init__(
        self,
        records: List[dict],
        cl_encoder: MultiModalCLPolymerEncoder,
        selfies_tok,
        max_len: int = GEN_MAX_LEN,
        latent_noise_std: float = 0.0,
        cache_embeddings: bool = True,
        renormalize_after_noise: bool = True,
        use_multimodal: bool = True,
    ):
        self.records = records
        self.cl_encoder = cl_encoder
        self.tok = selfies_tok
        self.max_len = int(max_len)
        self.latent_noise_std = float(latent_noise_std)
        self.renorm = bool(renormalize_after_noise)
        self.use_multimodal = bool(use_multimodal)

        self.pad_id = int(self.tok.pad_token_id) if getattr(self.tok, "pad_token_id", None) is not None else 1
        self._cache = None

        # Optionally precompute latents (saves a lot of time during decoder training)
        if cache_embeddings:
            if self.use_multimodal:
                emb = self.cl_encoder.encode_multimodal(self.records, batch_size=32, device=DEVICE)
            else:
                psm = [str(r.get("psmiles", "")) for r in self.records]
                emb = self.cl_encoder.encode_psmiles(psm, max_len=PSMILES_MAX_LEN, batch_size=64, device=DEVICE)
            self._cache = emb.astype(np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        tgt = str(r["pselfies"]).strip()
        tgt = _selfies_for_tokenizer(tgt)

        # Get latent z (cached or computed on the fly)
        if self._cache is not None:
            z = torch.tensor(self._cache[idx], dtype=torch.float32)
        else:
            if self.use_multimodal:
                z_np = self.cl_encoder.encode_multimodal([r], batch_size=1, device=DEVICE)
                z = torch.tensor(z_np[0], dtype=torch.float32)
            else:
                psm = str(r.get("psmiles", "")).strip()
                z_np = self.cl_encoder.encode_psmiles([psm], max_len=PSMILES_MAX_LEN, batch_size=1, device=DEVICE)
                z = torch.tensor(z_np[0], dtype=torch.float32)

        # Denoising noise
        if self.latent_noise_std > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std
            if self.renorm:
                z = F.normalize(z, dim=-1)

        # Tokenize target SELFIES; mask padding to -100 for CE
        enc = self.tok(tgt, truncation=True, padding="max_length", max_length=self.max_len, return_tensors=None)
        labels = torch.tensor(enc["input_ids"], dtype=torch.long)
        labels = labels.masked_fill(labels == self.pad_id, -100)

        return {
            "z": z,
            "labels": labels,
            "psmiles": str(r.get("psmiles", "")).strip(),
            "pselfies_raw": _selfies_compact(r["pselfies"]),
        }


def latent_collate(batch: List[dict]) -> dict:
    """Collate latents and labels into batch tensors."""
    z = torch.stack([b["z"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {
        "z": z,
        "labels": labels,
        "psmiles": [b["psmiles"] for b in batch],
        "pselfies_raw": [b["pselfies_raw"] for b in batch],
    }


def move_latent_batch_to_device(batch: dict, device: str):
    batch["z"] = batch["z"].to(device)
    batch["labels"] = batch["labels"].to(device)


# =============================================================================
# Aux PSMILES property oracle (optional)
# =============================================================================
class PSMILESPropertyDataset(Dataset):
    """Text regression dataset: PSMILES -> scaled property (single scalar)."""
    def __init__(self, samples: List[dict], psmiles_tokenizer, max_len: int = PSMILES_MAX_LEN):
        self.samples = samples
        self.tok = psmiles_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = str(self.samples[idx].get("psmiles", "")).strip()
        y = float(self.samples[idx].get("target_scaled", self.samples[idx].get("target", 0.0)))
        enc = self.tok(s, truncation=True, padding="max_length", max_length=self.max_len)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.bool),
            "y": torch.tensor([y], dtype=torch.float32),
        }


def psmiles_prop_collate_fn(batch: List[dict]):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attn = torch.stack([b["attention_mask"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attn, "y": y}


class TextPropertyOracle(nn.Module):
    """
    Lightweight regressor for verification:
      - Frozen PSMILES encoder (DeBERTa variant)
      - Trainable MLP head
    """
    def __init__(self, encoder_dir: Optional[str], vocab_size: Optional[int] = None, y_dim: int = 1):
        super().__init__()
        if encoder_dir is not None and os.path.isdir(encoder_dir):
            enc_src = encoder_dir
        elif os.path.isdir(CFG.BEST_PSMILES_DIR):
            enc_src = CFG.BEST_PSMILES_DIR
        else:
            enc_src = "microsoft/deberta-v2-xlarge"

        self.encoder = PSMILESDebertaEncoder(
            model_dir_or_name=enc_src,
            vocab_fallback=int(vocab_size) if vocab_size is not None else 300,
        )
        h = getattr(self.encoder, "out_dim", DEBERTA_HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, y_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(h)


def move_prop_batch_to_device(batch: dict, device: str):
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    batch["y"] = batch["y"].to(device)


def train_prop_oracle_one_epoch(model: TextPropertyOracle, dl: DataLoader, opt, scaler_amp, device: str):
    model.train()
    total = 0.0
    n = 0
    for batch in dl:
        move_prop_batch_to_device(batch, device)
        y = batch["y"]
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            y_hat = model(batch["input_ids"], batch["attention_mask"])
            loss = F.smooth_l1_loss(y_hat, y, beta=1.0)

        if USE_AMP:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(opt)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


@torch.no_grad()
def eval_prop_oracle(model: TextPropertyOracle, dl: DataLoader, device: str):
    model.eval()
    total = 0.0
    n = 0
    for batch in dl:
        move_prop_batch_to_device(batch, device)
        y = batch["y"]
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            y_hat = model(batch["input_ids"], batch["attention_mask"])
            loss = F.smooth_l1_loss(y_hat, y, beta=1.0)
        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


def train_property_oracle_per_fold(
    train_samples: List[dict],
    val_samples: List[dict],
    psmiles_tokenizer,
    device: str,
    max_len: int = PSMILES_MAX_LEN,
) -> Optional[TextPropertyOracle]:
    """Train a per-fold auxiliary oracle for scaled property prediction (verification only)."""
    if psmiles_tokenizer is None:
        return None

    try:
        model = TextPropertyOracle(
            encoder_dir=CFG.BEST_PSMILES_DIR if os.path.isdir(CFG.BEST_PSMILES_DIR) else None,
            vocab_size=getattr(psmiles_tokenizer, "vocab_size", None),
            y_dim=1,
        ).to(device)
    except Exception as e:
        print(f"[ORACLE][WARN] Could not initialize auxiliary property predictor: {e}")
        return None

    # Freeze encoder; train only head (fast + stable)
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    ds_tr = PSMILESPropertyDataset(train_samples, psmiles_tokenizer, max_len=max_len)
    ds_va = PSMILESPropertyDataset(val_samples, psmiles_tokenizer, max_len=max_len)
    dl_tr = DataLoader(ds_tr, batch_size=PROP_PRED_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=psmiles_prop_collate_fn)
    dl_va = DataLoader(ds_va, batch_size=PROP_PRED_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=psmiles_prop_collate_fn)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=PROP_PRED_LR, weight_decay=PROP_PRED_WEIGHT_DECAY)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val = float("inf")
    best_state = None
    no_imp = 0

    for epoch in range(1, PROP_PRED_EPOCHS + 1):
        tr = train_prop_oracle_one_epoch(model, dl_tr, opt, scaler_amp, device)
        va = eval_prop_oracle(model, dl_va, device)
        if va < best_val - 1e-8:
            best_val = va
            no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PROP_PRED_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)

    try:
        model.aux_val_loss = float(best_val)
    except Exception:
        pass

    return model


@torch.no_grad()
def oracle_predict_scaled(
    oracle: Optional[TextPropertyOracle],
    psmiles_tokenizer,
    psmiles_list: List[str],
    device: str,
    max_len: int = PSMILES_MAX_LEN,
) -> Optional[np.ndarray]:
    """Batch predict scaled properties with the auxiliary oracle."""
    if oracle is None or psmiles_tokenizer is None:
        return None
    if not psmiles_list:
        return np.array([], dtype=np.float32)

    oracle.eval()
    ys = []
    bs = 32
    for i in range(0, len(psmiles_list), bs):
        chunk = psmiles_list[i : i + bs]
        enc = psmiles_tokenizer(chunk, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device).bool()
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            y_hat = oracle(input_ids, attn)
        ys.append(y_hat.detach().cpu().numpy().reshape(-1))

    return np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.float32)


# =============================================================================
# Latent property model (per property)
# =============================================================================
@dataclass
class LatentPropertyModel:
    y_scaler: StandardScaler
    pca: Optional[PCA]
    gpr: GaussianProcessRegressor


def fit_latent_property_model(z_train: np.ndarray, y_train: np.ndarray, y_scaler: StandardScaler) -> LatentPropertyModel:
    """
    Fit a GPR mapping (PSMILES latent) -> (scaled property).
    Uses optional PCA for stability when latent dim is large.
    """
    y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    y_s = y_scaler.transform(y_train).reshape(-1).astype(np.float32)

    z_use = z_train.astype(np.float32)
    pca = None

    if USE_PCA_BEFORE_GPR:
        ncomp = int(min(PCA_DIM, z_use.shape[0] - 1, z_use.shape[1]))
        ncomp = max(2, ncomp)
        pca = PCA(n_components=ncomp, random_state=0)
        z_use = pca.fit_transform(z_use)

    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=GPR_ALPHA, normalize_y=True, random_state=0, n_restarts_optimizer=2)
    gpr.fit(z_use, y_s)

    return LatentPropertyModel(y_scaler=y_scaler, pca=pca, gpr=gpr)


def predict_latent_property(model: LatentPropertyModel, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict scaled and unscaled properties for candidate latents."""
    z_use = z.astype(np.float32)
    if model.pca is not None:
        z_use = model.pca.transform(z_use)
    y_s = model.gpr.predict(z_use, return_std=False)
    y_s = np.array(y_s, dtype=np.float32).reshape(-1)
    y_u = model.y_scaler.inverse_transform(y_s.reshape(-1, 1)).reshape(-1)
    return y_s, y_u


# =============================================================================
# Train / eval loops (decoder)
# =============================================================================
def train_one_epoch_decoder(model: CLConditionedSelfiesTEDGenerator, dl: DataLoader, optimizer, scaler_amp, device: str):
    """One epoch of teacher-forced decoder fine-tuning."""
    model.train()
    total = 0.0
    n = 0
    ce_sum = 0.0

    for batch in dl:
        move_latent_batch_to_device(batch, device)
        z = batch["z"]
        labels = batch["labels"]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            out = model.forward_train(z, labels)
            loss = out["loss"]

        if USE_AMP:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = z.size(0)
        total += float(loss.item()) * bs
        ce_sum += float(out["ce"].item()) * bs
        n += bs

    return {"loss": total / max(1, n), "ce": ce_sum / max(1, n)}


@torch.no_grad()
def evaluate_decoder(model: CLConditionedSelfiesTEDGenerator, dl: DataLoader, device: str):
    """Validation loss for early stopping."""
    model.eval()
    total = 0.0
    n = 0
    ce_sum = 0.0

    for batch in dl:
        move_latent_batch_to_device(batch, device)
        z = batch["z"]
        labels = batch["labels"]

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            out = model.forward_train(z, labels)
            loss = out["loss"]

        bs = z.size(0)
        total += float(loss.item()) * bs
        ce_sum += float(out["ce"].item()) * bs
        n += bs

    return {"loss": total / max(1, n), "ce": ce_sum / max(1, n)}


# =============================================================================
# Generation / filtering (per target value, per property)
# =============================================================================
def compute_diversity_morgan(smiles_list: List[str], radius: int = 2, nbits: int = 2048, p: float = 1.0) -> Optional[float]:
    """
    Diversity = 1 - mean(Tanimoto), computed on Morgan fingerprints of unique valid SMILES.
    Returns None if RDKit unavailable or insufficient valid molecules.
    """
    if not RDKit_AVAILABLE:
        return None

    try:
        p = float(p)
        if not np.isfinite(p) or p <= 0:
            p = 1.0
    except Exception:
        p = 1.0

    uniq = []
    seen = set()
    for smi in smiles_list:
        smi = str(smi).strip()
        if not smi or smi in seen:
            continue
        seen.add(smi)
        uniq.append(smi)

    fps = []
    for smi in uniq:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
            except Exception:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(fp)
        except Exception:
            continue

    if len(fps) < 2:
        return 0.0 if len(fps) == 1 else None

    sims_p = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            try:
                s = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
                sims_p.append(s ** p)
            except Exception:
                continue

    if not sims_p:
        return None

    mean_sim_p = float(np.mean(sims_p))
    try:
        mean_sim = mean_sim_p ** (1.0 / p)
    except Exception:
        mean_sim = float(
            np.mean([float(DataStructs.TanimotoSimilarity(fps[i], fps[j])) for i in range(len(fps)) for j in range(i + 1, len(fps))])
        )

    return float(1.0 - mean_sim)


@torch.no_grad()
def decode_from_latents(generator: CLConditionedSelfiesTEDGenerator, z: torch.Tensor, n_samples: int = 1) -> List[str]:
    """Decode PSELFIES from a batch of CL latents."""
    return generator.generate(
        z=z,
        num_return_sequences=int(n_samples),
        max_len=GEN_MAX_LEN,
        top_p=GEN_TOP_P,
        temperature=GEN_TEMPERATURE,
        repetition_penalty=GEN_REPETITION_PENALTY,
    )


def generate_for_target(
    target_y_scaled: float,
    prop_model: LatentPropertyModel,
    cl_encoder: MultiModalCLPolymerEncoder,
    generator: CLConditionedSelfiesTEDGenerator,
    train_seed_pool: List[dict],
    train_targets_set: set,
    n_seeds: int = 8,
    n_noise: int = N_FOLD_NOISE_SAMPLING,
    noise_std: float = LATENT_NOISE_STD_GEN,
    prop_tol_scaled: float = PROP_TOL_SCALED,
    oracle: Optional[TextPropertyOracle] = None,
    psmiles_tokenizer=None,
) -> Dict[str, Any]:
    """
    Core generation routine for a single target property value (scaled):
      1) Pick seed polymers close to target (in scaled property space).
      2) Encode seeds (multimodal) -> latent vectors.
      3) Add Gaussian noise to latents (exploration), renormalize.
      4) Decode to PSELFIES -> convert to polymer PSMILES.
      5) Filter by polymer/chem validity and property closeness (via GPR on PSMILES latents).
      6) Compute novelty/uniqueness/diversity metrics; optionally score with aux oracle.
    """

    def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(n, eps, None)

    # Choose nearest seeds by property distance (scaled)
    ys = np.array([float(d["y_scaled"]) for d in train_seed_pool], dtype=np.float32)
    diffs = np.abs(ys - float(target_y_scaled))
    order = np.argsort(diffs)
    chosen = [train_seed_pool[i] for i in order[: max(1, int(n_seeds))]]

    # Encode chosen seeds using multimodal encoder
    z_seed = cl_encoder.encode_multimodal(chosen, batch_size=32, device=DEVICE)
    if z_seed.shape[0] == 0:
        return {"generated": [], "metrics": {}}

    # Sample noise around each seed latent
    z_list = []
    for i in range(z_seed.shape[0]):
        z0 = z_seed[i].astype(np.float32)
        for _ in range(int(n_noise)):
            z = z0 + np.random.randn(z0.shape[0]).astype(np.float32) * float(noise_std)
            z = _l2_normalize_np(z.reshape(1, -1)).reshape(-1)
            z_list.append(z)

    z_all = np.stack(z_list, axis=0).astype(np.float32)
    z_t = torch.tensor(z_all, dtype=torch.float32, device=DEVICE)

    # Decode to PSELFIES
    pselfies = decode_from_latents(generator, z_t, n_samples=1)

    # Convert to polymer PSMILES; record validity flags
    valid_psmiles = []
    valid_flags, poly_flags = [], []

    for s in pselfies:
        s = _ensure_two_at_endpoints(_selfies_compact(s))
        psm = pselfies_to_psmiles(s) if (RDKit_AVAILABLE and SELFIES_AVAILABLE) else None
        if psm is None:
            valid_flags.append(False)
            poly_flags.append(False)
            continue

        psm_can = canonicalize_psmiles(psm)
        ok = chem_validity_psmiles(psm_can) if psm_can else False
        poly_ok = polymer_validity_psmiles_strict(psm_can) if psm_can else False
        valid_flags.append(bool(ok))
        poly_flags.append(bool(poly_ok))

        if ok and poly_ok and psm_can:
            valid_psmiles.append(psm_can)

    uniq_valid = sorted(set(valid_psmiles))
    novelty_valid = [1.0 if s not in train_targets_set else 0.0 for s in uniq_valid] if uniq_valid else []

    n_valid_poly = int(len(valid_psmiles))
    uniqueness_valid_unique = float(len(uniq_valid)) / float(max(1, n_valid_poly)) if n_valid_poly > 0 else 0.0

    # Property prediction via GPR on PSMILES latents (for filtering)
    if uniq_valid:
        z_cand = cl_encoder.encode_psmiles(uniq_valid, max_len=PSMILES_MAX_LEN, batch_size=64, device=DEVICE)
    else:
        z_cand = np.zeros((0, cl_encoder.out_dim), dtype=np.float32)

    yhat_s, yhat_u = (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    if z_cand.shape[0] > 0:
        yhat_s, yhat_u = predict_latent_property(prop_model, z_cand)

    keep, keep_pred_scaled, keep_pred_unscaled = [], [], []
    for i, psm in enumerate(uniq_valid):
        if abs(float(yhat_s[i]) - float(target_y_scaled)) <= float(prop_tol_scaled):
            keep.append(psm)
            keep_pred_scaled.append(float(yhat_s[i]))
            keep_pred_unscaled.append(float(yhat_u[i]))

    novelty_keep = [1.0 if s not in train_targets_set else 0.0 for s in keep] if keep else []

    # Optional aux oracle prediction for additional sanity checking
    aux_pred_scaled = None
    if VERIFY_GENERATED_PROPERTIES and oracle is not None and psmiles_tokenizer is not None and keep:
        aux = oracle_predict_scaled(oracle, psmiles_tokenizer, keep, DEVICE, PSMILES_MAX_LEN)
        aux_pred_scaled = aux.tolist() if aux is not None else None

    # Diversity computed on At-SMILES (to avoid polymer "*" parsing issues)
    at_smiles = []
    if RDKit_AVAILABLE and keep:
        for psm in keep:
            at_smi = psmiles_to_at_smiles(psm, root_at=False)
            if at_smi is not None:
                at_smiles.append(at_smi)
    div = compute_diversity_morgan(at_smiles) if at_smiles else None

    metrics = {
        "n_total": int(len(pselfies)),
        "validity": float(np.mean(valid_flags)) if valid_flags else 0.0,
        "polymer_validity": float(np.mean(poly_flags)) if poly_flags else 0.0,
        "n_valid_unique": int(len(uniq_valid)),
        "novelty_valid_unique": float(np.mean(novelty_valid)) if novelty_valid else 0.0,
        "uniqueness_valid_unique": float(uniqueness_valid_unique),
        "n_kept_property_filtered": int(len(keep)),
        "novelty_kept": float(np.mean(novelty_keep)) if novelty_keep else 0.0,
        "diversity": float(div) if div is not None else 0.0,
    }

    return {
        "generated": keep,
        "pred_scaled_kept": keep_pred_scaled,
        "pred_unscaled_kept": keep_pred_unscaled,
        "aux_pred_scaled": aux_pred_scaled,
        "metrics": metrics,
    }


# =============================================================================
# Data assembly (per property)
# =============================================================================
def build_polymer_records(df: pd.DataFrame, prop_col: str) -> List[dict]:
    """
    Build records for a single property:
      - require chemically valid + strictly polymer-valid PSMILES
      - require finite property value
      - generate PSELFIES for decoder targets
      - preserve optional modalities for multimodal seed encoding
    """
    if not (RDKit_AVAILABLE and SELFIES_AVAILABLE):
        raise RuntimeError("RDKit + selfies are required for this pipeline.")

    recs = []
    for _, row in df.iterrows():
        psmiles_raw = str(row.get("psmiles", "")).strip()
        if not psmiles_raw:
            continue

        psm_can = canonicalize_psmiles(psmiles_raw)
        if not psm_can:
            continue
        if not chem_validity_psmiles(psm_can):
            continue
        if not polymer_validity_psmiles_strict(psm_can):
            continue

        val = row.get(prop_col, None)
        if val is None:
            continue
        try:
            y = float(val)
            if not np.isfinite(y):
                continue
        except Exception:
            continue

        pself = psmiles_to_pselfies(psm_can)
        if pself is None:
            continue

        recs.append(
            {
                "psmiles": psm_can,
                "pselfies": pself,
                "y": y,
                "graph": row.get("graph", None),
                "geometry": row.get("geometry", None),
                "fingerprints": row.get("fingerprints", None),
            }
        )
    return recs


# =============================================================================
# Best-fold artifact saving (per property)
# =============================================================================
def save_best_fold_artifacts_for_property(
    property_name: str,
    fold_idx: int,
    decoder_state: Dict[str, torch.Tensor],
    prop_model: Optional[LatentPropertyModel],
    scaler: Optional[StandardScaler],
    best_val_loss: float,
    generations_payload: List[dict],
):
    """
    Persist the best fold for a property:
      - decoder state_dict
      - scaler + GPR (joblib, if available)
      - meta.json describing hyperparams
      - jsonl generations payload for traceability
    """
    safe_prop = property_name.replace(" ", "_")
    prop_dir = os.path.join(CFG.OUTPUT_MODELS_DIR, safe_prop)
    os.makedirs(prop_dir, exist_ok=True)

    decoder_path = os.path.join(prop_dir, f"decoder_best_fold{fold_idx+1}.pt")
    torch.save(decoder_state, decoder_path)

    try:
        import joblib
    except Exception:
        joblib = None

    if joblib is not None:
        if scaler is not None:
            joblib.dump(scaler, os.path.join(prop_dir, f"standardscaler_{safe_prop}.joblib"))
        if prop_model is not None:
            joblib.dump(prop_model, os.path.join(prop_dir, f"gpr_psmiles_{safe_prop}.joblib"))

    meta = {
        "property": property_name,
        "best_fold": int(fold_idx + 1),
        "best_val_loss": float(best_val_loss),
        "selfies_ted_model": str(SELFIES_TED_MODEL_NAME),
        "cl_emb_dim": int(CL_EMB_DIM),
        "mem_len": 4,
        "tol_scaled": float(PROP_TOL_SCALED),
        "tol_unscaled_abs": float(PROP_TOL_UNSCALED_ABS) if PROP_TOL_UNSCALED_ABS is not None else None,
        "optimizer": "AdamW",
        "lr": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "lr_scheduler": "CosineAnnealingLR",
        "epochs": int(NUM_EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "patience": int(PATIENCE),
    }

    try:
        with open(os.path.join(prop_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    out_path = os.path.join(CFG.OUTPUT_GENERATIONS_DIR, f"{safe_prop}_best_fold{fold_idx+1}_generated_psmiles.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in generations_payload:
                fh.write(json.dumps(make_json_serializable({"property": property_name, "best_fold": fold_idx + 1, **r})) + "\n")
    except Exception as e:
        print(f"[SAVE][WARN] Could not write generations for '{property_name}': {e}")


# =============================================================================
# Main per-property CV loop (single-task)
# =============================================================================
def run_inverse_design_single_property(
    df: pd.DataFrame,
    property_name: str,
    prop_col: str,
    cl_encoder: MultiModalCLPolymerEncoder,
    selfies_tok,
    selfies_model,
) -> Dict[str, Any]:
    """
    Run fivefold CV for a single property and log fold-level metrics.
    Best fold is tracked by decoder validation loss and saved to disk.
    """
    polymers = build_polymer_records(df, prop_col)

    if len(polymers) < 200:
        print(f"[{property_name}][WARN] Only {len(polymers)} usable samples; results may be noisy.")
    if len(polymers) < 50:
        print(f"[{property_name}][WARN] Skipping due to insufficient usable samples (<50).")
        return {"property": property_name, "runs": [], "agg": None, "n_samples": len(polymers)}

    indices = np.arange(len(polymers))
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    runs = []
    best_overall_val = float("inf")
    best_bundle = None  # kept for completeness; artifacts saved immediately when best improves

    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(indices)):
        seed = 42 + fold_idx
        set_seed(seed)

        print(f"\n[{property_name}] Fold {fold_idx+1}/{NUM_FOLDS} | seed={seed}")

        trainval_polys = [polymers[i] for i in trainval_idx]
        test_polys = [polymers[i] for i in test_idx]

        # Train/val split within trainval
        tr_idx, va_idx = train_test_split(np.arange(len(trainval_polys)), test_size=0.10, random_state=seed, shuffle=True)
        train_polys = [copy.deepcopy(trainval_polys[i]) for i in tr_idx]
        val_polys = [copy.deepcopy(trainval_polys[i]) for i in va_idx]

        # Scale property targets using TRAIN only
        sc = StandardScaler()
        sc.fit(np.array([p["y"] for p in train_polys], dtype=np.float32).reshape(-1, 1))

        # Helper to format records for latent dataset
        def _to_rec(p):
            return {
                "psmiles": p["psmiles"],
                "pselfies": p["pselfies"],
                "graph": p.get("graph", None),
                "geometry": p.get("geometry", None),
                "fingerprints": p.get("fingerprints", None),
            }

        # Decoder training datasets (cache CL embeddings for speed)
        ds_train = LatentToPSELFIESDataset(
            [_to_rec(p) for p in train_polys],
            cl_encoder,
            selfies_tok,
            max_len=GEN_MAX_LEN,
            latent_noise_std=LATENT_NOISE_STD_TRAIN,
            cache_embeddings=True,
            use_multimodal=True,
        )
        ds_val = LatentToPSELFIESDataset(
            [_to_rec(p) for p in val_polys],
            cl_encoder,
            selfies_tok,
            max_len=GEN_MAX_LEN,
            latent_noise_std=0.0,
            cache_embeddings=True,
            use_multimodal=True,
        )

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=latent_collate)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=latent_collate)

        # Fit GPR on PSMILES latents for this property (train only)
        y_tr = [float(p["y"]) for p in train_polys]
        psm_tr = [p["psmiles"] for p in train_polys]
        z_tr = cl_encoder.encode_psmiles(psm_tr, max_len=PSMILES_MAX_LEN, batch_size=64, device=DEVICE)
        prop_model = fit_latent_property_model(z_tr, np.array(y_tr, dtype=np.float32), y_scaler=sc)
        print(f"[{property_name}] Fit PSMILES-latent GPR (n_train={len(y_tr)})")

        # Optional aux oracle (scaled)
        oracle = None
        if VERIFY_GENERATED_PROPERTIES and len(train_polys) >= 200 and len(val_polys) >= 50:
            tr_s, va_s = [], []
            for p in train_polys:
                y_s = float(sc.transform(np.array([[p["y"]]], dtype=np.float32))[0, 0])
                tr_s.append({"psmiles": p["psmiles"], "target": p["y"], "target_scaled": y_s})
            for p in val_polys:
                y_s = float(sc.transform(np.array([[p["y"]]], dtype=np.float32))[0, 0])
                va_s.append({"psmiles": p["psmiles"], "target": p["y"], "target_scaled": y_s})
            try:
                oracle = train_property_oracle_per_fold(tr_s, va_s, cl_encoder.psm_tok, DEVICE, PSMILES_MAX_LEN)
                print(f"[{property_name}] Trained aux oracle (val_loss={getattr(oracle, 'aux_val_loss', None)})")
            except Exception as e:
                print(f"[{property_name}][WARN] Oracle training failed: {e}")
                oracle = None

        # Fresh decoder per fold + optimizer
        selfies_tok_f, selfies_model_f = load_selfies_ted_and_tokenizer(SELFIES_TED_MODEL_NAME)
        decoder = CLConditionedSelfiesTEDGenerator(selfies_tok_f, selfies_model_f, cl_emb_dim=CL_EMB_DIM, mem_len=4).to(DEVICE)
        optimizer, scheduler = create_optimizer_and_scheduler_decoder(decoder)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        best_val = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            tr = train_one_epoch_decoder(decoder, dl_train, optimizer, scaler_amp, DEVICE)
            va = evaluate_decoder(decoder, dl_val, DEVICE)
            try:
                scheduler.step()
            except Exception:
                pass

            try:
                lr = float(optimizer.param_groups[0]["lr"])
                print(
                    f"[{property_name}] fold {fold_idx+1}/{NUM_FOLDS} | epoch {epoch:03d} | "
                    f"lr={lr:.2e} | train_loss={tr['loss']:.6f} | val_loss={va['loss']:.6f}"
                )
            except Exception:
                print(
                    f"[{property_name}] fold {fold_idx+1}/{NUM_FOLDS} | epoch {epoch:03d} | "
                    f"train_loss={tr['loss']:.6f} | val_loss={va['loss']:.6f}"
                )

            if va["loss"] < best_val - 1e-8:
                best_val = va["loss"]
                no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"[{property_name}] Early stopping (no val improvement for {PATIENCE} epochs).")
                    break

        if best_state is None:
            print(f"[{property_name}][WARN] No best state captured; skipping this fold.")
            continue

        decoder.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()}, strict=False)

        # Seed pool for generation (scaled property values, plus modalities for encoding)
        seed_pool = []
        for p in train_polys:
            y_s = float(sc.transform(np.array([[p["y"]]], dtype=np.float32))[0, 0])
            seed_pool.append(
                {
                    "psmiles": p["psmiles"],
                    "y_scaled": y_s,
                    "graph": p.get("graph", None),
                    "geometry": p.get("geometry", None),
                    "fingerprints": p.get("fingerprints", None),
                }
            )

        train_targets_set = set(ps["psmiles"] for ps in train_polys)

        # Compose test targets (scaled); subsample for runtime control
        ys_test_scaled = []
        for p in test_polys:
            ys_test_scaled.append(float(sc.transform(np.array([[p["y"]]], dtype=np.float32))[0, 0]))
        ys_test_scaled = np.array(ys_test_scaled, dtype=np.float32)
        if len(ys_test_scaled) > 64:
            ys_test_scaled = np.random.choice(ys_test_scaled, size=64, replace=False)

        # Generate per target
        all_valid, all_poly, all_kept, success_scaled, mae_best, diversity_vals = [], [], [], [], [], []
        novelty_vals, uniqueness_vals = [], []
        per_target_records = []

        for y_t in ys_test_scaled:
            out = generate_for_target(
                target_y_scaled=float(y_t),
                prop_model=prop_model,
                cl_encoder=cl_encoder,
                generator=decoder,
                train_seed_pool=seed_pool,
                train_targets_set=train_targets_set,
                n_seeds=8,
                n_noise=min(N_FOLD_NOISE_SAMPLING, 16),
                noise_std=LATENT_NOISE_STD_GEN,
                prop_tol_scaled=PROP_TOL_SCALED,
                oracle=oracle,
                psmiles_tokenizer=cl_encoder.psm_tok,
            )

            m = out["metrics"]
            all_valid.append(float(m.get("validity", 0.0)))
            all_poly.append(float(m.get("polymer_validity", 0.0)))
            all_kept.append(int(m.get("n_kept_property_filtered", 0)))
            diversity_vals.append(float(m.get("diversity", 0.0)))
            success_scaled.append(1.0 if int(m.get("n_kept_property_filtered", 0)) > 0 else 0.0)
            novelty_vals.append(float(m.get("novelty_kept", 0.0)))
            uniqueness_vals.append(float(m.get("uniqueness_valid_unique", 0.0)))

            # Best error among kept candidates
            if out["generated"]:
                z_keep = cl_encoder.encode_psmiles(out["generated"], max_len=PSMILES_MAX_LEN, batch_size=64, device=DEVICE)
                y_pred_s, _ = predict_latent_property(prop_model, z_keep)
                if len(y_pred_s):
                    err = np.abs(y_pred_s - float(y_t))
                    mae_best.append(float(np.min(err)))
                else:
                    mae_best.append(float("inf"))
            else:
                mae_best.append(float("inf"))

            target_y_unscaled = float(sc.inverse_transform(np.array([[float(y_t)]], dtype=np.float32))[0, 0])
            aux_list = out.get("aux_pred_scaled", None)
            if aux_list is not None and not isinstance(aux_list, list):
                aux_list = None

            candidates = []
            gen_list = out.get("generated", []) or []
            pred_s_list = out.get("pred_scaled_kept", []) or []
            pred_u_list = out.get("pred_unscaled_kept", []) or []

            for i_c, psm in enumerate(gen_list):
                cand = {
                    "psmiles": str(psm),
                    "pred_scaled": float(pred_s_list[i_c]) if i_c < len(pred_s_list) else None,
                    "pred_unscaled": float(pred_u_list[i_c]) if i_c < len(pred_u_list) else None,
                    "aux_pred_scaled": float(aux_list[i_c]) if (aux_list is not None and i_c < len(aux_list)) else None,
                }
                candidates.append(cand)

            scaler_meta = {
                "scaler_type": "StandardScaler",
                "mean_": getattr(sc, "mean_", None),
                "scale_": getattr(sc, "scale_", None),
                "with_mean": bool(getattr(sc, "with_mean", True)),
                "with_std": bool(getattr(sc, "with_std", True)),
            }

            per_target_records.append(
                {
                    "target_y_scaled": float(y_t),
                    "target_y_unscaled": float(target_y_unscaled),
                    "tol_scaled": float(PROP_TOL_SCALED),
                    "tol_unscaled_abs": float(PROP_TOL_UNSCALED_ABS) if PROP_TOL_UNSCALED_ABS is not None else None,
                    "scaler_meta": scaler_meta,
                    "candidates": candidates,
                    "metrics": m,
                }
            )

        def _finite(xs):
            return [x for x in xs if np.isfinite(x)]

        metrics_fold = {
            "validity_mean": float(np.mean(all_valid)) if all_valid else 0.0,
            "polymer_validity_mean": float(np.mean(all_poly)) if all_poly else 0.0,
            "avg_n_kept": float(np.mean(all_kept)) if all_kept else 0.0,
            "success_at_k_scaled": float(np.mean(success_scaled)) if success_scaled else 0.0,
            "mae_best_scaled": float(np.mean(_finite(mae_best))) if _finite(mae_best) else 0.0,
            "diversity_mean": float(np.mean(diversity_vals)) if diversity_vals else 0.0,
            "novelty_mean": float(np.mean(novelty_vals)) if novelty_vals else 0.0,
            "uniqueness_mean": float(np.mean(uniqueness_vals)) if uniqueness_vals else 0.0,
            "tol_scaled": float(PROP_TOL_SCALED),
            "tol_unscaled_abs": float(PROP_TOL_UNSCALED_ABS) if PROP_TOL_UNSCALED_ABS is not None else None,
        }

        run_record = {
            "property": property_name,
            "fold": int(fold_idx + 1),
            "seed": int(seed),
            "n_train": int(len(train_polys)),
            "n_val": int(len(val_polys)),
            "n_test": int(len(test_polys)),
            "best_val_loss": float(best_val),
            "gen_metrics": metrics_fold,
        }
        runs.append(run_record)

        with open(CFG.OUTPUT_RESULTS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(make_json_serializable(run_record)) + "\n")

        # Save best fold artifacts by lowest validation loss
        if best_val < best_overall_val - 1e-8:
            best_overall_val = best_val
            best_bundle = {
                "fold": int(fold_idx + 1),
                "decoder_state": best_state,
                "prop_model": prop_model,
                "scaler": sc,
                "best_val_loss": float(best_val),
                "generations": per_target_records,
            }
            save_best_fold_artifacts_for_property(
                property_name=property_name,
                fold_idx=fold_idx,
                decoder_state=best_state,
                prop_model=prop_model,
                scaler=sc,
                best_val_loss=best_val,
                generations_payload=per_target_records,
            )
            print(f"[{property_name}] Saved best-fold artifacts (fold {fold_idx+1}, val_loss={best_val:.6f}).")

    # Aggregate across folds
    if not runs:
        return {"property": property_name, "runs": [], "agg": None, "n_samples": len(polymers)}

    def _collect(key):
        xs = [float(r["gen_metrics"].get(key, 0.0)) for r in runs if r.get("gen_metrics", None) is not None]
        return (float(np.mean(xs)) if xs else 0.0, float(np.std(xs)) if xs else 0.0)

    agg = {}
    for k in [
        "validity_mean",
        "polymer_validity_mean",
        "avg_n_kept",
        "success_at_k_scaled",
        "mae_best_scaled",
        "diversity_mean",
        "novelty_mean",
        "uniqueness_mean",
    ]:
        m, s = _collect(k)
        agg[k] = {"mean": m, "std": s}

    agg["tol_scaled"] = float(PROP_TOL_SCALED)
    agg["tol_unscaled_abs"] = float(PROP_TOL_UNSCALED_ABS) if PROP_TOL_UNSCALED_ABS is not None else None

    with open(CFG.OUTPUT_RESULTS, "a", encoding="utf-8") as fh:
        fh.write("AGG_PROPERTY: " + json.dumps(make_json_serializable({property_name: agg})) + "\n")

    return {"property": property_name, "runs": runs, "agg": agg, "n_samples": len(polymers)}


# =============================================================================
# Entrypoint (single-task per property)
# =============================================================================
def main():
    ensure_output_dirs(CFG)

    if not (RDKit_AVAILABLE and SELFIES_AVAILABLE):
        raise RuntimeError("This script requires RDKit and selfies. Install them before running.")

    # Reset results file
    if os.path.exists(CFG.OUTPUT_RESULTS):
        backup = CFG.OUTPUT_RESULTS + ".bak"
        shutil.copy(CFG.OUTPUT_RESULTS, backup)
        print(f"[IO][INFO] Backed up existing results file to: {backup}")
    open(CFG.OUTPUT_RESULTS, "w", encoding="utf-8").close()

    # Load dataset
    if not os.path.isfile(CFG.POLYINFO_PATH):
        raise FileNotFoundError(f"PolyInfo CSV not found: {CFG.POLYINFO_PATH}")

    df = pd.read_csv(CFG.POLYINFO_PATH, engine="python")
    found = find_property_columns(df.columns)
    prop_map = {req: found.get(req) for req in REQUESTED_PROPERTIES}

    print("\n" + "=" * 80)
    print("[RUN] Inverse design (single-task per property)")
    print("=" * 80)
    print(f"[ENV] RDKit_AVAILABLE={RDKit_AVAILABLE} | SELFIES_AVAILABLE={SELFIES_AVAILABLE}")
    print(f"[ENV] DEVICE={DEVICE} | USE_AMP={USE_AMP} | NUM_WORKERS={NUM_WORKERS}")
    print(f"[DATA] POLYINFO_PATH={CFG.POLYINFO_PATH}")
    print(f"[DATA] Property map: {prop_map}")
    print(f"[CL]  CL checkpoint dir: {CFG.PRETRAINED_MULTIMODAL_DIR}")
    print(f"[DEC] SELFIES_TED_MODEL_NAME={SELFIES_TED_MODEL_NAME}")
    print(
        f"[DEC] FT params: batch={BATCH_SIZE}, epochs={NUM_EPOCHS}, patience={PATIENCE}, "
        f"lr={LEARNING_RATE}, wd={WEIGHT_DECAY}, sched=CosineAnnealingLR(eta_min={COSINE_ETA_MIN})"
    )
    print(f"[GEN] Latent noise: train_std={LATENT_NOISE_STD_TRAIN}, gen_std={LATENT_NOISE_STD_GEN}, n_noise={N_FOLD_NOISE_SAMPLING}")
    print(f"[GEN] Filter tol: scaled={PROP_TOL_SCALED}, abs={PROP_TOL_UNSCALED_ABS}")
    print(f"[AUX] VERIFY_GENERATED_PROPERTIES={VERIFY_GENERATED_PROPERTIES}")
    print("=" * 80 + "\n")

    # Build PSMILES tokenizer for CL text encoder
    psmiles_tok = build_psmiles_tokenizer(spm_path=CFG.SPM_MODEL_PATH, max_len=PSMILES_MAX_LEN)
    if psmiles_tok is None:
        raise RuntimeError("Failed to build PSMILES tokenizer (check SPM_MODEL_PATH).")

    # Multimodal CL encoder (frozen; used as conditioning interface)
    cl_encoder = MultiModalCLPolymerEncoder(
        psmiles_tokenizer=psmiles_tok,
        emb_dim=CL_EMB_DIM,
        cl_weights_dir=CFG.PRETRAINED_MULTIMODAL_DIR,
        use_gine=True,
        use_schnet=True,
        use_fp=True,
        use_psmiles=True,
    ).to(DEVICE)
    cl_encoder.freeze_cl_encoders()

    # Load SELFIES-TED backbone 
    selfies_tok, selfies_model = load_selfies_ted_and_tokenizer(SELFIES_TED_MODEL_NAME)
    print(f"[HF][INFO] Loaded SELFIES-TED backbone: {SELFIES_TED_MODEL_NAME}")

    overall = {"per_property": {}}

    # Single-task loop per property
    for pname in REQUESTED_PROPERTIES:
        pcol = prop_map.get(pname, None)
        if pcol is None:
            print(f"[{pname}][WARN] No column match found; skipping.")
            continue

        print(f"\n>>> Property: '{pname}' | column='{pcol}'")
        res = run_inverse_design_single_property(df, pname, pcol, cl_encoder, selfies_tok, selfies_model)
        overall["per_property"][pname] = res

    # Final summary (aggregated per property)
    final_agg = {}
    for pname, info in overall["per_property"].items():
        final_agg[pname] = info.get("agg", None)

    with open(CFG.OUTPUT_RESULTS, "a", encoding="utf-8") as fh:
        fh.write("\nFINAL_SUMMARY\n")
        fh.write(json.dumps(make_json_serializable(final_agg), indent=2))
        fh.write("\n")

    print("\n" + "=" * 80)
    print("Finished inverse design runs.")
    print(f"Results file: {CFG.OUTPUT_RESULTS}")
    print(f"Best models dir: {CFG.OUTPUT_MODELS_DIR}")
    print(f"Best-fold generations dir: {CFG.OUTPUT_GENERATIONS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
