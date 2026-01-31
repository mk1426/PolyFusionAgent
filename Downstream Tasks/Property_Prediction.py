
import os
import random
import time
from pathlib import Path
import math
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import csv
import copy
from typing import List, Dict, Optional, Tuple, Any

csv.field_size_limit(sys.maxsize)

# Shared encoders/helpers from PolyFusion
from PolyFusion.GINE import GineEncoder, match_edge_attr_to_index, safe_get
from PolyFusion.SchNet import NodeSchNetWrapper
from PolyFusion.Transformer import PooledFingerprintEncoder as FingerprintEncoder
from PolyFusion.DeBERTav2 import PSMILESDebertaEncoder, build_psmiles_tokenizer

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = "/path/to/Polymer_Foundational_Model"
POLYINFO_PATH = "/path/to/polyinfo_with_modalities.csv"

# Pretrained model directories 
PRETRAINED_MULTIMODAL_DIR = "/path/to/multimodal_output/best"
BEST_GINE_DIR = "/path/to/gin_output/best"
BEST_SCHNET_DIR = "/path/to/schnet_output/best"
BEST_FP_DIR = "/path/to/fingerprint_mlm_output/best"
BEST_PSMILES_DIR = "/path/to/polybert_output/best"

OUTPUT_RESULTS = "/path/to/multimodal_downstream_results.txt"

# NEW: directory to save best-performing weights per property (best run/"fold")
BEST_WEIGHTS_DIR = "/path/to/multimodal_downstream_bestweights"

# Model hyperparameters (matching multimodal training)
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

CL_EMB_DIM = 600

# PSMILES/Deberta params
DEBERTA_HIDDEN = 600
PSMILES_MAX_LEN = 128

# Uni-Poly fusion params
UNIPOLY_EMB_DIM = 600
UNIPOLY_ATTN_HEADS = 8
UNIPOLY_DROPOUT = 0.1
UNIPOLY_FF_MULT = 4  # FFN hidden = 4*d (common transformer practice)

# Training parameters (Uni-Poly fine-tuning)
MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REQUESTED_PROPERTIES = [
    "density",
    "glass transition",
    "melting",
    "specific volume",
    "thermal decomposition"
]

# Uni-Poly reports stats over fivefold CV; we implement true 5-fold CV here
NUM_RUNS = 5
TEST_SIZE = 0.10
VAL_SIZE_WITHIN_TRAINVAL = 0.10  # of trainval (i.e., 9% overall when TEST_SIZE=0.10)

# Optional: PolyInfo duplicate aggregation key preference order
AGG_KEYS_PREFERENCE = ["polymer_id", "PolymerID", "poly_id", "psmiles", "smiles", "canonical_smiles"]

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_json_serializable(obj):
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
    return obj

def summarize_state_dict_load(full_state: dict, model_state: dict, filtered_state: dict):
    n_ckpt = len(full_state)
    n_model = len(model_state)
    n_loaded = len(filtered_state)

    missing_in_model = [k for k in full_state.keys() if k not in model_state]
    shape_mismatch = [
        k for k in full_state.keys()
        if k in model_state and hasattr(full_state[k], "shape")
        and tuple(full_state[k].shape) != tuple(model_state[k].shape)
    ]

    print("\n[CKPT LOAD SUMMARY]")
    print(f"  ckpt keys:   {n_ckpt}")
    print(f"  model keys:  {n_model}")
    print(f"  loaded keys: {n_loaded}")
    print(f"  skipped (not in model): {len(missing_in_model)}")
    print(f"  skipped (shape mismatch): {len(shape_mismatch)}")

    if missing_in_model:
        print("  examples missing_in_model:", missing_in_model[:10])
    if shape_mismatch:
        print("  examples shape_mismatch:")
        for k in shape_mismatch[:10]:
            print(f"    {k}: ckpt={tuple(full_state[k].shape)} model={tuple(model_state[k].shape)}")
    print("")


def find_property_columns(columns):
    """
    Safer property column matching:
    - Prefer exact token matches (word-level).
    - For 'density' explicitly avoid matching columns that contain 'cohesive' (e.g., 'Cohesive energy density').
    - Log candidates when ambiguity exists.
    """
    lowered = {c.lower(): c for c in columns}
    found = {}

    for req in REQUESTED_PROPERTIES:
        req_low = req.lower().strip()
        exact = None

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

        candidates = [c_orig for c_low, c_orig in lowered.items() if req_low in c_low]
        if req_low == "density":
            candidates = [c for c in candidates if "cohesive" not in c.lower() and "cohesive energy" not in c.lower()]

        if len(candidates) == 1:
            found[req] = candidates[0]
        else:
            chosen = candidates[0] if candidates else None
            found[req] = chosen
            print(f"[INFO] Requested property '{req}' -> chosen column: {chosen}")
            if candidates:
                print(f"[INFO] Candidates for '{req}': {candidates}")
            else:
                print(f"[WARN] No candidates found for '{req}' using substring search.")
    return found

def choose_aggregation_key(df: pd.DataFrame) -> Optional[str]:
    for k in AGG_KEYS_PREFERENCE:
        if k in df.columns:
            return k
    return None


def aggregate_polyinfo_duplicates(df: pd.DataFrame, modality_cols: List[str], property_cols: List[str]) -> pd.DataFrame:
    """
    Uni-Poly describes grouping samples for the same monomer and averaging properties to reduce noise.
    We apply that here if we can identify a stable key (polymer_id / psmiles / smiles).
    """
    key = choose_aggregation_key(df)
    if key is None:
        print("[INFO] No aggregation key found; skipping duplicate aggregation.")
        return df

    # Keep only rows with a non-empty key
    df2 = df.copy()
    df2[key] = df2[key].astype(str)
    df2 = df2[df2[key].str.strip() != ""].copy()
    if len(df2) == 0:
        print("[INFO] Aggregation key exists but is empty; skipping duplicate aggregation.")
        return df

    agg_dict = {}
    # modalities: take first non-null string (they should be identical per polymer id in your dataset)
    for mc in modality_cols:
        if mc in df2.columns:
            agg_dict[mc] = "first"
    # properties: mean
    for pc in property_cols:
        if pc in df2.columns:
            agg_dict[pc] = "mean"

    grouped = df2.groupby(key, as_index=False).agg(agg_dict)
    print(f"[INFO] Aggregated duplicates by '{key}': {len(df)} rows -> {len(grouped)} unique keys")
    return grouped


def _sanitize_name(s: str) -> str:
    s2 = str(s).strip().lower()
    keep = []
    for ch in s2:
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "-", "_", "."):
            keep.append("_")
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out or "property"

class MultimodalContrastiveModel(nn.Module):
    """
    Multimodal encoder wrapper that:
      - Takes pretrained modality encoders (GINE / SchNet / FP / PSMILES)
      - Projects each modality to emb_dim
      - L2-normalizes per modality embedding, then masked-mean combines across available modalities
      - L2-normalizes final combined embedding

    This version matches your downstream instantiation:
        MultimodalContrastiveModel(gine_encoder, schnet_encoder, fp_encoder, psmiles_encoder,
                                  emb_dim=..., modalities=[...])

    And your downstream forward usage:
        z = model(batch_mods, modality_mask=modality_mask)
    """

    def __init__(
        self,
        gine_encoder: Optional[nn.Module] = None,
        schnet_encoder: Optional[nn.Module] = None,
        fp_encoder: Optional[nn.Module] = None,
        psmiles_encoder: Optional[nn.Module] = None,
        *,
        emb_dim: int = CL_EMB_DIM,
        modalities: Optional[List[str]] = None,
        dropout: float = 0.1,
        psmiles_tokenizer: Optional[Any] = None,
    ):
        super().__init__()

        self.gine = gine_encoder
        self.schnet = schnet_encoder
        self.fp = fp_encoder
        self.psmiles = psmiles_encoder
        self.psm_tok = psmiles_tokenizer

        self.emb_dim = int(emb_dim)
        self.out_dim = self.emb_dim
        self.dropout = nn.Dropout(float(dropout))

        # Decide which modalities are enabled
        if modalities is None:
            mods = []
            if self.gine is not None:
                mods.append("gine")
            if self.schnet is not None:
                mods.append("schnet")
            if self.fp is not None:
                mods.append("fp")
            if self.psmiles is not None:
                mods.append("psmiles")
            self.modalities = mods
        else:
            self.modalities = [m for m in modalities if m in ("gine", "schnet", "fp", "psmiles")]

        # Projection heads (match dims used elsewhere in your script)
        self.proj_gine = nn.Linear(NODE_EMB_DIM, self.emb_dim) if self.gine is not None else None
        self.proj_schnet = nn.Linear(SCHNET_HIDDEN, self.emb_dim) if self.schnet is not None else None
        self.proj_fp = nn.Linear(256, self.emb_dim) if self.fp is not None else None

        # PSMILES encoder may not always be exactly DEBERTA_HIDDEN; try to infer
        psm_in = None
        if self.psmiles is not None:
            if hasattr(self.psmiles, "out_dim"):
                try:
                    psm_in = int(self.psmiles.out_dim)
                except Exception:
                    psm_in = None
            if psm_in is None and hasattr(self.psmiles, "model") and hasattr(self.psmiles.model, "config"):
                try:
                    psm_in = int(self.psmiles.model.config.hidden_size)
                except Exception:
                    psm_in = None
            if psm_in is None:
                psm_in = int(DEBERTA_HIDDEN)

        self.proj_psmiles = nn.Linear(psm_in, self.emb_dim) if (self.psmiles is not None) else None

    def freeze_cl_encoders(self):
        """Optional helper to freeze modality encoders."""
        for enc in (self.gine, self.schnet, self.fp, self.psmiles):
            if enc is None:
                continue
            enc.eval()
            for p in enc.parameters():
                p.requires_grad = False

    def _masked_mean_combine(self, zs: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """
        zs:   list of (B,D)
        masks:list of (B,) boolean, True where modality is present
        returns: (B,D) masked mean
        """
        if not zs:
            device = next(self.parameters()).device
            return torch.zeros((1, self.emb_dim), device=device)

        device = zs[0].device
        B = zs[0].size(0)

        sum_z = torch.zeros((B, self.emb_dim), device=device)
        count = torch.zeros((B, 1), device=device)

        for z, m in zip(zs, masks):
            m = m.to(device).view(B, 1).float()
            sum_z = sum_z + z * m
            count = count + m

        count = count.clamp(min=1.0)
        return sum_z / count

    def forward(self, batch_mods: dict, modality_mask: Optional[dict] = None) -> torch.Tensor:
        """
        batch_mods keys: 'gine', 'schnet', 'fp', 'psmiles'
        modality_mask: dict with boolean tensors of shape (B,) for each modality
        """
        device = next(self.parameters()).device

        zs = []
        ms = []

        # Determine batch size B from any available modality tensor
        B = None
        if modality_mask is not None:
            for k, v in modality_mask.items():
                if isinstance(v, torch.Tensor) and v.numel() > 0:
                    B = int(v.size(0))
                    break

        # Fallback: infer B from fp/psmiles tensors
        if B is None:
            if "fp" in batch_mods and batch_mods["fp"] is not None and isinstance(batch_mods["fp"].get("input_ids", None), torch.Tensor):
                B = int(batch_mods["fp"]["input_ids"].size(0))
            elif "psmiles" in batch_mods and batch_mods["psmiles"] is not None and isinstance(batch_mods["psmiles"].get("input_ids", None), torch.Tensor):
                B = int(batch_mods["psmiles"]["input_ids"].size(0))

        if B is None:
            return torch.zeros((1, self.emb_dim), device=device)

        # Helper to get modality present mask
        def _get_mask(name: str) -> torch.Tensor:
            if modality_mask is not None and name in modality_mask and isinstance(modality_mask[name], torch.Tensor):
                return modality_mask[name].to(device).bool()
            return torch.ones((B,), device=device, dtype=torch.bool)

        # GINE
        if "gine" in self.modalities and self.gine is not None and batch_mods.get("gine", None) is not None:
            g = batch_mods["gine"]
            if isinstance(g.get("z", None), torch.Tensor) and g["z"].numel() > 0:
                emb_g = self.gine(
                    g["z"].to(device),
                    g.get("chirality", None).to(device) if isinstance(g.get("chirality", None), torch.Tensor) else None,
                    g.get("formal_charge", None).to(device) if isinstance(g.get("formal_charge", None), torch.Tensor) else None,
                    g.get("edge_index", torch.empty((2, 0), dtype=torch.long)).to(device) if isinstance(g.get("edge_index", None), torch.Tensor) else torch.empty((2, 0), dtype=torch.long, device=device),
                    g.get("edge_attr", torch.zeros((0, 3), dtype=torch.float)).to(device) if isinstance(g.get("edge_attr", None), torch.Tensor) else torch.zeros((0, 3), dtype=torch.float, device=device),
                    g.get("batch", None).to(device) if isinstance(g.get("batch", None), torch.Tensor) else None
                )
                z = self.proj_gine(emb_g) if self.proj_gine is not None else emb_g
                z = F.normalize(z, dim=-1)
                z = self.dropout(z)
                zs.append(z)
                ms.append(_get_mask("gine"))

        # SchNet
        if "schnet" in self.modalities and self.schnet is not None and batch_mods.get("schnet", None) is not None:
            s = batch_mods["schnet"]
            if isinstance(s.get("z", None), torch.Tensor) and s["z"].numel() > 0:
                emb_s = self.schnet(
                    s["z"].to(device),
                    s["pos"].to(device) if isinstance(s.get("pos", None), torch.Tensor) else torch.zeros((0, 3), device=device),
                    s.get("batch", None).to(device) if isinstance(s.get("batch", None), torch.Tensor) else None
                )
                z = self.proj_schnet(emb_s) if self.proj_schnet is not None else emb_s
                z = F.normalize(z, dim=-1)
                z = self.dropout(z)
                zs.append(z)
                ms.append(_get_mask("schnet"))

        # FP
        if "fp" in self.modalities and self.fp is not None and batch_mods.get("fp", None) is not None:
            f = batch_mods["fp"]
            if isinstance(f.get("input_ids", None), torch.Tensor) and f["input_ids"].numel() > 0:
                emb_f = self.fp(
                    f["input_ids"].to(device),
                    f.get("attention_mask", None).to(device) if isinstance(f.get("attention_mask", None), torch.Tensor) else None
                )
                z = self.proj_fp(emb_f) if self.proj_fp is not None else emb_f
                z = F.normalize(z, dim=-1)
                z = self.dropout(z)
                zs.append(z)
                ms.append(_get_mask("fp"))

        # PSMILES
        if "psmiles" in self.modalities and self.psmiles is not None and batch_mods.get("psmiles", None) is not None:
            p = batch_mods["psmiles"]
            if isinstance(p.get("input_ids", None), torch.Tensor) and p["input_ids"].numel() > 0:
                emb_p = self.psmiles(
                    p["input_ids"].to(device),
                    p.get("attention_mask", None).to(device) if isinstance(p.get("attention_mask", None), torch.Tensor) else None
                )
                z = self.proj_psmiles(emb_p) if self.proj_psmiles is not None else emb_p
                z = F.normalize(z, dim=-1)
                z = self.dropout(z)
                zs.append(z)
                ms.append(_get_mask("psmiles"))

        # Combine (masked mean), then normalize
        if not zs:
            return torch.zeros((B, self.emb_dim), device=device)

        z = self._masked_mean_combine(zs, ms)
        z = F.normalize(z, dim=-1)
        return z


    @torch.no_grad()
    def encode_psmiles(
        self,
        psmiles_list: List[str],
        max_len: int = PSMILES_MAX_LEN,
        batch_size: int = 64,
        device: str = DEVICE
    ) -> np.ndarray:
        """
        PSMILES-only embeddings (used for generated candidates + oracle).
        """
        self.eval()
        if self.psm_tok is None or self.psmiles is None or self.proj_psmiles is None:
            raise RuntimeError("PSMILES tokenizer/encoder/projection not available.")

        outs = []
        for i in range(0, len(psmiles_list), batch_size):
            chunk = [str(x) for x in psmiles_list[i:i + batch_size]]
            enc = self.psm_tok(chunk, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device).bool()
            emb_p = self.psmiles(input_ids, attn)
            z = F.normalize(self.proj_psmiles(emb_p), dim=-1)
            outs.append(z.detach().cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.emb_dim), dtype=np.float32)

    @torch.no_grad()
    def encode_multimodal(
        self,
        records: List[dict],
        batch_size: int = 32,
        device: str = DEVICE
    ) -> np.ndarray:
        """
        Full multimodal embeddings for records that carry:
          graph, geometry, fingerprints, psmiles
        Missing modalities are skipped per-sample (combine is mean over available).
        """
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        outs = []
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]

            # Build per-sample modality tensors; do minimal batching (stack where possible)
            # PSMILES batch
            psmiles_texts = [str(r.get("psmiles", "")) for r in chunk]
            p_enc = None
            if self.psm_tok is not None:
                p_enc = self.psm_tok(psmiles_texts, truncation=True, padding="max_length", max_length=PSMILES_MAX_LEN, return_tensors="pt")

            # FP batch
            fp_ids, fp_attn = [], []
            for r in chunk:
                f = _parse_fingerprints(r.get("fingerprints", None), fp_len=FP_LENGTH)
                fp_ids.append(f["input_ids"])
                fp_attn.append(f["attention_mask"])
            fp_ids = torch.stack(fp_ids, dim=0)
            fp_attn = torch.stack(fp_attn, dim=0)

            # GINE and SchNet: variable-size; we keep as “packed” with simple batch vectors
            # (mean pooling per sample already handled in encoder via batch vectors)
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
                gine_batch = {"z": z_b, "chirality": ch_b, "formal_charge": fc_b, "edge_index": ei_b, "edge_attr": ea_b, "batch": b_b}

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
                "psmiles": {"input_ids": p_enc["input_ids"], "attention_mask": p_enc["attention_mask"]} if p_enc is not None else None
            }

            z = self.forward_multimodal(batch_mods)
            outs.append(z.detach().cpu().numpy())

        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.emb_dim), dtype=np.float32)

# -----------------------------
# Tokenizer setup
# -----------------------------
SPM_MODEL = "/path/to/spm.model"
tokenizer = build_psmiles_tokenizer(spm_path=SPM_MODEL, max_len=PSMILES_MAX_LEN)

# -----------------------------
# Dataset (single-task, Uni-Poly fine-tuning style)
# -----------------------------
class PolymerPropertyDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # Parse graph data for GINE
        gine_data = None
        if 'graph' in data and data['graph']:
            try:
                graph_field = json.loads(data['graph']) if isinstance(data['graph'], str) else data['graph']

                node_features = safe_get(graph_field, "node_features", None)
                if node_features:
                    atomic_nums = []
                    chirality_vals = []
                    formal_charges = []
                    for nf in node_features:
                        an = safe_get(nf, "atomic_num", None)
                        if an is None:
                            an = safe_get(nf, "atomic_number", 0)
                        ch = safe_get(nf, "chirality", 0)
                        fc = safe_get(nf, "formal_charge", 0)
                        try:
                            atomic_nums.append(int(an))
                        except Exception:
                            atomic_nums.append(0)
                        chirality_vals.append(float(ch))
                        formal_charges.append(float(fc))

                    edge_indices_raw = safe_get(graph_field, "edge_indices", None)
                    edge_features_raw = safe_get(graph_field, "edge_features", None)
                    edge_index = None
                    edge_attr = None

                    if edge_indices_raw is None:
                        adj_mat = safe_get(graph_field, "adjacency_matrix", None)
                        if adj_mat:
                            srcs = []
                            dsts = []
                            for i_r, row_adj in enumerate(adj_mat):
                                for j, val2 in enumerate(row_adj):
                                    if val2:
                                        srcs.append(i_r)
                                        dsts.append(j)
                            if len(srcs) > 0:
                                edge_index = [srcs, dsts]
                                E = len(srcs)
                                edge_attr = [[0.0, 0.0, 0.0] for _ in range(E)]
                    else:
                        srcs, dsts = [], []
                        if isinstance(edge_indices_raw, list) and len(edge_indices_raw) > 0:
                            if isinstance(edge_indices_raw[0], list):
                                first = edge_indices_raw[0]
                                if len(first) == 2 and isinstance(first[0], int):
                                    try:
                                        srcs = [int(p[0]) for p in edge_indices_raw]
                                        dsts = [int(p[1]) for p in edge_indices_raw]
                                    except Exception:
                                        srcs, dsts = [], []
                                else:
                                    try:
                                        srcs = [int(x) for x in edge_indices_raw[0]]
                                        dsts = [int(x) for x in edge_indices_raw[1]]
                                    except Exception:
                                        srcs, dsts = [], []
                            else:
                                try:
                                    srcs = [int(x) for x in edge_indices_raw[0]]
                                    dsts = [int(x) for x in edge_indices_raw[1]]
                                except Exception:
                                    srcs, dsts = [], []

                        if len(srcs) > 0:
                            edge_index = [srcs, dsts]

                        if edge_features_raw and isinstance(edge_features_raw, list):
                            bond_types = []
                            stereos = []
                            is_conjs = []
                            for ef in edge_features_raw:
                                bt = safe_get(ef, "bond_type", 0)
                                st = safe_get(ef, "stereo", 0)
                                ic = safe_get(ef, "is_conjugated", False)
                                bond_types.append(float(bt))
                                stereos.append(float(st))
                                is_conjs.append(float(1.0 if ic else 0.0))
                            edge_attr = list(zip(bond_types, stereos, is_conjs))
                        else:
                            E = len(srcs)
                            edge_attr = [[0.0, 0.0, 0.0] for _ in range(E)]

                    if edge_index is not None:
                        gine_data = {
                            'z': torch.tensor(atomic_nums, dtype=torch.long),
                            'chirality': torch.tensor(chirality_vals, dtype=torch.float),
                            'formal_charge': torch.tensor(formal_charges, dtype=torch.float),
                            'edge_index': torch.tensor(edge_index, dtype=torch.long),
                            'edge_attr': torch.tensor(edge_attr, dtype=torch.float)
                        }
            except Exception:
                gine_data = None

        # Parse geometry data for SchNet
        schnet_data = None
        if 'geometry' in data and data['geometry']:
            try:
                geom = json.loads(data['geometry']) if isinstance(data['geometry'], str) else data['geometry']
                conf = geom.get("best_conformer") if isinstance(geom, dict) else None
                if conf:
                    atomic = conf.get("atomic_numbers", [])
                    coords = conf.get("coordinates", [])
                    if len(atomic) == len(coords) and len(atomic) > 0:
                        schnet_data = {
                            'z': torch.tensor(atomic, dtype=torch.long),
                            'pos': torch.tensor(coords, dtype=torch.float)
                        }
            except Exception:
                schnet_data = None

        # Parse fingerprints
        fp_data = None
        if 'fingerprints' in data and data['fingerprints']:
            try:
                fpval = data['fingerprints']
                if fpval is not None and not (isinstance(fpval, str) and fpval.strip() == ""):
                    try:
                        fp_json = json.loads(fpval) if isinstance(fpval, str) else fpval
                    except Exception:
                        try:
                            fp_json = json.loads(str(fpval).replace("'", '"'))
                        except Exception:
                            parts = [p.strip().strip('"').strip("'") for p in str(fpval).split(",")]
                            bits = [1 if p in ("1", "True", "true") else 0 for p in parts[:FP_LENGTH]]
                            if len(bits) < FP_LENGTH:
                                bits += [0] * (FP_LENGTH - len(bits))
                            fp_json = bits

                    if isinstance(fp_json, dict):
                        bits = safe_get(fp_json, "morgan_r3_bits", None)
                        if bits is None:
                            bits = [0] * FP_LENGTH
                        else:
                            normalized = []
                            for b in bits:
                                if isinstance(b, str):
                                    b_clean = b.strip().strip('"').strip("'")
                                    normalized.append(1 if b_clean in ("1", "True", "true") else 0)
                                elif isinstance(b, (int, np.integer)):
                                    normalized.append(1 if int(b) != 0 else 0)
                                else:
                                    normalized.append(0)
                                if len(normalized) >= FP_LENGTH:
                                    break
                            if len(normalized) < FP_LENGTH:
                                normalized.extend([0] * (FP_LENGTH - len(normalized)))
                            bits = normalized[:FP_LENGTH]
                    elif isinstance(fp_json, list):
                        bits = fp_json[:FP_LENGTH]
                        if len(bits) < FP_LENGTH:
                            bits += [0] * (FP_LENGTH - len(bits))
                    else:
                        bits = [0] * FP_LENGTH

                    fp_data = {
                        'input_ids': torch.tensor(bits, dtype=torch.long),
                        'attention_mask': torch.ones(FP_LENGTH, dtype=torch.bool)
                    }
            except Exception:
                fp_data = None

        # Parse PSMILES
        psmiles_data = None
        if 'psmiles' in data and data['psmiles'] and self.tokenizer is not None:
            try:
                s = str(data['psmiles'])
                enc = self.tokenizer(
                    s,
                    truncation=True,
                    padding="max_length",
                    max_length=PSMILES_MAX_LEN
                )
                psmiles_data = {
                    'input_ids': torch.tensor(enc["input_ids"], dtype=torch.long),
                    'attention_mask': torch.tensor(enc["attention_mask"], dtype=torch.bool)
                }
            except Exception:
                psmiles_data = None

        # Defaults for missing modalities
        if gine_data is None:
            gine_data = {
                'z': torch.tensor([], dtype=torch.long),
                'chirality': torch.tensor([], dtype=torch.float),
                'formal_charge': torch.tensor([], dtype=torch.float),
                'edge_index': torch.tensor([[], []], dtype=torch.long),
                'edge_attr': torch.zeros((0, 3), dtype=torch.float)
            }

        if schnet_data is None:
            schnet_data = {
                'z': torch.tensor([], dtype=torch.long),
                'pos': torch.tensor([], dtype=torch.float)
            }

        if fp_data is None:
            fp_data = {
                'input_ids': torch.zeros(FP_LENGTH, dtype=torch.long),
                'attention_mask': torch.zeros(FP_LENGTH, dtype=torch.bool)
            }

        if psmiles_data is None:
            psmiles_data = {
                'input_ids': torch.zeros(PSMILES_MAX_LEN, dtype=torch.long),
                'attention_mask': torch.zeros(PSMILES_MAX_LEN, dtype=torch.bool)
            }

        # Single-task target (scaled)
        target_scaled = float(data.get("target_scaled", 0.0))

        return {
            'gine': gine_data,
            'schnet': schnet_data,
            'fp': fp_data,
            'psmiles': psmiles_data,
            'target': torch.tensor(target_scaled, dtype=torch.float32),
        }


def multimodal_collate_fn(batch):
    B = len(batch)

    # GINE batching
    all_z = []
    all_ch = []
    all_fc = []
    all_edge_index = []
    all_edge_attr = []
    batch_mapping = []
    node_offset = 0
    gine_present = []

    for i, item in enumerate(batch):
        g = item["gine"]
        z = g["z"]
        n = z.size(0)
        gine_present.append(bool(n > 0))

        all_z.append(z)
        all_ch.append(g["chirality"])
        all_fc.append(g["formal_charge"])
        batch_mapping.append(torch.full((n,), i, dtype=torch.long))

        if g["edge_index"] is not None and g["edge_index"].numel() > 0:
            ei_offset = g["edge_index"] + node_offset
            all_edge_index.append(ei_offset)
            ea = match_edge_attr_to_index(g["edge_index"], g["edge_attr"], target_dim=3)
            all_edge_attr.append(ea)
        node_offset += n

    if len(all_z) == 0:
        z_batch = torch.tensor([], dtype=torch.long)
        ch_batch = torch.tensor([], dtype=torch.float)
        fc_batch = torch.tensor([], dtype=torch.float)
        batch_batch = torch.tensor([], dtype=torch.long)
        edge_index_batched = torch.empty((2, 0), dtype=torch.long)
        edge_attr_batched = torch.zeros((0, 3), dtype=torch.float)
    else:
        z_batch = torch.cat(all_z, dim=0)
        ch_batch = torch.cat(all_ch, dim=0)
        fc_batch = torch.cat(all_fc, dim=0)
        batch_batch = torch.cat(batch_mapping, dim=0) if len(batch_mapping) > 0 else torch.tensor([], dtype=torch.long)
        if len(all_edge_index) > 0:
            edge_index_batched = torch.cat(all_edge_index, dim=1)
            edge_attr_batched = torch.cat(all_edge_attr, dim=0) if len(all_edge_attr) > 0 else torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index_batched = torch.empty((2, 0), dtype=torch.long)
            edge_attr_batched = torch.zeros((0, 3), dtype=torch.float)

    # SchNet batching
    all_sz = []
    all_pos = []
    schnet_batch = []
    schnet_present = [False] * B
    for i, item in enumerate(batch):
        s = item["schnet"]
        s_z = s["z"]
        s_pos = s["pos"]
        if s_z.numel() == 0:
            continue
        schnet_present[i] = True
        all_sz.append(s_z)
        all_pos.append(s_pos)
        schnet_batch.append(torch.full((s_z.size(0),), i, dtype=torch.long))

    if len(all_sz) == 0:
        s_z_batch = torch.tensor([], dtype=torch.long)
        s_pos_batch = torch.tensor([], dtype=torch.float)
        s_batch_batch = torch.tensor([], dtype=torch.long)
    else:
        s_z_batch = torch.cat(all_sz, dim=0)
        s_pos_batch = torch.cat(all_pos, dim=0)
        s_batch_batch = torch.cat(schnet_batch, dim=0)

    # FP batching
    fp_ids = torch.stack([item["fp"]["input_ids"] for item in batch], dim=0)
    fp_attn = torch.stack([item["fp"]["attention_mask"] for item in batch], dim=0)
    fp_present = (fp_attn.sum(dim=1) > 0).cpu().numpy().tolist()

    # PSMILES batching
    p_ids = torch.stack([item["psmiles"]["input_ids"] for item in batch], dim=0)
    p_attn = torch.stack([item["psmiles"]["attention_mask"] for item in batch], dim=0)
    psmiles_present = (p_attn.sum(dim=1) > 0).cpu().numpy().tolist()

    # Target
    target = torch.stack([item["target"] for item in batch], dim=0)  # (B,)

    modality_mask = {
        "gine": torch.tensor(gine_present, dtype=torch.bool),
        "schnet": torch.tensor(schnet_present, dtype=torch.bool),
        "fp": torch.tensor(fp_present, dtype=torch.bool),
        "psmiles": torch.tensor(psmiles_present, dtype=torch.bool),
    }

    return {
        "gine": {
            "z": z_batch,
            "chirality": ch_batch,
            "formal_charge": fc_batch,
            "edge_index": edge_index_batched,
            "edge_attr": edge_attr_batched,
            "batch": batch_batch
        },
        "schnet": {
            "z": s_z_batch,
            "pos": s_pos_batch,
            "batch": s_batch_batch
        },
        "fp": {
            "input_ids": fp_ids,
            "attention_mask": fp_attn
        },
        "psmiles": {
            "input_ids": p_ids,
            "attention_mask": p_attn
        },
        "target": target,
        "modality_mask": modality_mask
    }


# -----------------------------
# Property regression head (single task)
# -----------------------------
class UniPolyPropertyRegressor(nn.Module):
    def __init__(self, unipoly_model: MultimodalContrastiveModel, emb_dim: int = UNIPOLY_EMB_DIM, dropout: float = 0.1):
        super().__init__()
        self.unipoly = unipoly_model
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(self, batch_mods, modality_mask=None):
        emb = self.unipoly(batch_mods, modality_mask=modality_mask)  # (B,d)
        y = self.head(emb).squeeze(-1)  # (B,)
        return y


# -----------------------------
# Training / evaluation
# -----------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in dataloader:
        for k in batch:
            if k == "target":
                batch[k] = batch[k].to(device)
            elif k == "modality_mask":
                for mk in batch[k]:
                    if isinstance(batch[k][mk], torch.Tensor):
                        batch[k][mk] = batch[k][mk].to(device)
            else:
                for subk in batch[k]:
                    if isinstance(batch[k][subk], torch.Tensor):
                        batch[k][subk] = batch[k][subk].to(device)

        y = batch["target"]  # (B,)
        modality_mask = batch.get("modality_mask", None)
        batch_mods = {k: v for k, v in batch.items() if k not in ("target", "modality_mask")}

        pred = model(batch_mods, modality_mask=modality_mask)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(1, total_n)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    trues = []
    total_loss = 0.0
    total_n = 0

    for batch in dataloader:
        for k in batch:
            if k == "target":
                batch[k] = batch[k].to(device)
            elif k == "modality_mask":
                for mk in batch[k]:
                    if isinstance(batch[k][mk], torch.Tensor):
                        batch[k][mk] = batch[k][mk].to(device)
            else:
                for subk in batch[k]:
                    if isinstance(batch[k][subk], torch.Tensor):
                        batch[k][subk] = batch[k][subk].to(device)

        y = batch["target"]
        modality_mask = batch.get("modality_mask", None)
        batch_mods = {k: v for k, v in batch.items() if k not in ("target", "modality_mask")}

        pred = model(batch_mods, modality_mask=modality_mask)
        loss = F.mse_loss(pred, y)

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    if total_n == 0:
        return None, None, None

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    avg_loss = total_loss / max(1, total_n)
    return float(avg_loss), preds, trues


# -----------------------------
# Pretrained loading
# -----------------------------
def load_pretrained_multimodal(pretrained_path: str) -> MultimodalContrastiveModel:
    gine_encoder = GineEncoder(
        node_emb_dim=NODE_EMB_DIM,
        edge_emb_dim=EDGE_EMB_DIM,
        num_layers=NUM_GNN_LAYERS,
        max_atomic_z=MAX_ATOMIC_Z
    )
    if os.path.exists(os.path.join(BEST_GINE_DIR, "pytorch_model.bin")):
        try:
            gine_encoder.load_state_dict(
                torch.load(os.path.join(BEST_GINE_DIR, "pytorch_model.bin"), map_location="cpu"),
                strict=False
            )
            print(f"Loaded GINE weights from {BEST_GINE_DIR}")
        except Exception as e:
            print(f"Warning: Could not load GINE weights: {e}")

    schnet_encoder = NodeSchNetWrapper(
        hidden_channels=SCHNET_HIDDEN,
        num_interactions=SCHNET_NUM_INTERACTIONS,
        num_gaussians=SCHNET_NUM_GAUSSIANS,
        cutoff=SCHNET_CUTOFF,
        max_num_neighbors=SCHNET_MAX_NEIGHBORS
    )
    if os.path.exists(os.path.join(BEST_SCHNET_DIR, "pytorch_model.bin")):
        try:
            schnet_encoder.load_state_dict(
                torch.load(os.path.join(BEST_SCHNET_DIR, "pytorch_model.bin"), map_location="cpu"),
                strict=False
            )
            print(f"Loaded SchNet weights from {BEST_SCHNET_DIR}")
        except Exception as e:
            print(f"Warning: Could not load SchNet weights: {e}")

    fp_encoder = FingerprintEncoder(
        vocab_size=VOCAB_SIZE_FP,
        hidden_dim=256,
        seq_len=FP_LENGTH,
        num_layers=4,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1
    )
    if os.path.exists(os.path.join(BEST_FP_DIR, "pytorch_model.bin")):
        try:
            fp_encoder.load_state_dict(
                torch.load(os.path.join(BEST_FP_DIR, "pytorch_model.bin"), map_location="cpu"),
                strict=False
            )
            print(f"Loaded fingerprint encoder weights from {BEST_FP_DIR}")
        except Exception as e:
            print(f"Warning: Could not load fingerprint weights: {e}")

    psmiles_encoder = None
    if os.path.isdir(BEST_PSMILES_DIR):
        try:
            psmiles_encoder = PSMILESDebertaEncoder(model_dir_or_name=BEST_PSMILES_DIR)
            print(f"Loaded PSMILES encoder from {BEST_PSMILES_DIR}")
        except Exception as e:
            print(f"Warning: Could not load PSMILES encoder: {e}")

    if psmiles_encoder is None:
        try:
            psmiles_encoder = PSMILESDebertaEncoder(
                model_dir_or_name=None,
                vocab_fallback=int(getattr(tokenizer, "vocab_size", 300))
            )
        except Exception as e:
            print(f"Warning: Could not initialize PSMILES encoder: {e}")

    multimodal_model = MultimodalContrastiveModel(
        gine_encoder,
        schnet_encoder,
        fp_encoder,
        psmiles_encoder,
        emb_dim=UNIPOLY_EMB_DIM,
        modalities=["gine", "schnet", "fp", "psmiles"]
    )

    ckpt_path = os.path.join(pretrained_path, "pytorch_model.bin")
    if os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            model_state = multimodal_model.state_dict()
            filtered_state = {}
            for k, v in state.items():
                if k not in model_state:
                    continue
                if model_state[k].shape != v.shape:
                    continue
                filtered_state[k] = v

            summarize_state_dict_load(state, model_state, filtered_state)
            missing, unexpected = multimodal_model.load_state_dict(filtered_state, strict=False)
            print(f"[INFO] Loaded multimodal pretrained weights from {ckpt_path}")
            print(f"[INFO] load_state_dict() -> missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("[INFO] Missing keys (sample):", missing[:50])
            if unexpected:
                print("[INFO] Unexpected keys (sample):", unexpected[:50])

        except Exception as e:
            print(f"Warning: Failed to load multimodal pretrained weights: {e}")

    return multimodal_model


# -----------------------------
# Downstream run (Uni-Poly matched)
# -----------------------------
def build_samples_for_property(df: pd.DataFrame, prop_col: str) -> List[dict]:
    samples = []
    for _, row in df.iterrows():
        # Require at least one modality
        has_modality = False
        for col in ['graph', 'geometry', 'fingerprints', 'psmiles']:
            if col in row and row[col] and str(row[col]).strip() != "":
                has_modality = True
                break
        if not has_modality:
            continue

        val = row.get(prop_col, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        try:
            y = float(val)
        except Exception:
            continue

        samples.append({
            'graph': row.get('graph', ''),
            'geometry': row.get('geometry', ''),
            'fingerprints': row.get('fingerprints', ''),
            'psmiles': row.get('psmiles', ''),
            'target_raw': y
        })
    return samples


def run_unipoly_downstream(property_list: List[str], property_cols: List[str], df_raw: pd.DataFrame,
                           pretrained_path: str, output_file: str):
    os.makedirs(pretrained_path, exist_ok=True)

    # Aggregate duplicates (Uni-Poly noise reduction) if possible
    modality_cols = ["graph", "geometry", "fingerprints", "psmiles"]
    df_proc = aggregate_polyinfo_duplicates(df_raw, modality_cols=modality_cols, property_cols=property_cols)

    all_results = {"per_property": {}, "mode": "UNIPOLY_MATCHED_SINGLE_TASK"}

    # Load base pretrained model once per property-run (we reload inside each run to avoid leakage across runs)
    for pname, pcol in zip(property_list, property_cols):
        print(f"\n=== Uni-Poly matched fine-tuning: {pname} (col='{pcol}') ===")
        samples = build_samples_for_property(df_proc, pcol)
        if len(samples) < 200:
            print(f"[WARN] Too few samples for '{pname}': {len(samples)}. Results may be unstable.")
        if len(samples) < 50:
            print(f"[WARN] Skipping '{pname}' due to insufficient samples.")
            continue

        run_metrics = []
        run_records = []

        # NEW: track which run ("fold") gives best performance and save its weights
        best_overall_r2 = -1e18
        best_overall_payload = None  # contains model_state_dict (cpu), scaler stats, and metadata

        idxs = np.arange(len(samples))
        cv = KFold(n_splits=NUM_RUNS, shuffle=True, random_state=42)

        for run_idx, (trainval_idx, test_idx) in enumerate(cv.split(idxs)):
            seed = 42 + run_idx
            set_seed(seed)

            trainval = [copy.deepcopy(samples[i]) for i in trainval_idx]
            test = [copy.deepcopy(samples[i]) for i in test_idx]

            # train/val split within trainval
            tr_idx, va_idx = train_test_split(
                np.arange(len(trainval)),
                test_size=VAL_SIZE_WITHIN_TRAINVAL,
                random_state=seed,
                shuffle=True
            )
            train = [copy.deepcopy(trainval[i]) for i in tr_idx]
            val = [copy.deepcopy(trainval[i]) for i in va_idx]

            # Z-score normalization (StandardScaler) fit on train
            sc = StandardScaler()
            sc.fit(np.array([s["target_raw"] for s in train]).reshape(-1, 1))

            def _apply_scale(lst):
                for s in lst:
                    s["target_scaled"] = float(sc.transform(np.array([[s["target_raw"]]])).ravel()[0])

            _apply_scale(train)
            _apply_scale(val)
            _apply_scale(test)

            ds_train = PolymerPropertyDataset(train, tokenizer, max_length=MAX_LEN)
            ds_val = PolymerPropertyDataset(val, tokenizer, max_length=MAX_LEN)
            ds_test = PolymerPropertyDataset(test, tokenizer, max_length=MAX_LEN)

            dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)
            dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)
            dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=multimodal_collate_fn)

            # Load pretrained base, then attach Uni-Poly head; fine-tune end-to-end
            unipoly_base = load_pretrained_multimodal(pretrained_path)
            model = UniPolyPropertyRegressor(unipoly_base, emb_dim=UNIPOLY_EMB_DIM, dropout=UNIPOLY_DROPOUT).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

            best_val = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(1, NUM_EPOCHS + 1):
                tr_loss = train_one_epoch(model, dl_train, optimizer, DEVICE)
                va_loss, _, _ = evaluate(model, dl_val, DEVICE)
                va_loss = va_loss if va_loss is not None else float("inf")

                scheduler.step()

                print(f"[{pname}] fold {run_idx+1}/{NUM_RUNS} epoch {epoch:03d}  train={tr_loss:.6f}  val={va_loss:.6f}")

                if va_loss < best_val - 1e-8:
                    best_val = va_loss
                    no_improve = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        print(f"[{pname}] Early stopping at epoch {epoch} (patience={PATIENCE}).")
                        break

            if best_state is None:
                print(f"[WARN] No best state saved for {pname} fold {run_idx+1}; skipping.")
                continue

            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()}, strict=True)
            _, pred_scaled, true_scaled = evaluate(model, dl_test, DEVICE)
            if pred_scaled is None:
                continue

            # Inverse transform to original units
            pred = sc.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            true = sc.inverse_transform(true_scaled.reshape(-1, 1)).ravel()

            m = compute_metrics(true, pred)
            run_metrics.append(m)

            record = {
                "property": pname,
                "property_col": pcol,
                "run": run_idx + 1,
                "seed": seed,
                "n_train": len(ds_train),
                "n_val": len(ds_val),
                "n_test": len(ds_test),
                "best_val_loss": float(best_val),
                "test_metrics": m
            }
            run_records.append(record)

            with open(output_file, "a") as fh:
                fh.write(json.dumps(make_json_serializable(record)) + "\n")

            # NEW: if this run is best for the property (by test R2), keep its weights + scaler
            if float(m.get("r2", -1e18)) > float(best_overall_r2):
                best_overall_r2 = float(m.get("r2", -1e18))
                best_overall_payload = {
                    "property": pname,
                    "property_col": pcol,
                    "best_run": int(run_idx + 1),
                    "seed": int(seed),
                    "n_train": int(len(ds_train)),
                    "n_val": int(len(ds_val)),
                    "n_test": int(len(ds_test)),
                    "best_val_loss": float(best_val),
                    "test_metrics": make_json_serializable(m),
                    "scaler_mean": make_json_serializable(getattr(sc, "mean_", None)),
                    "scaler_scale": make_json_serializable(getattr(sc, "scale_", None)),
                    "scaler_var": make_json_serializable(getattr(sc, "var_", None)),
                    "scaler_n_samples_seen": make_json_serializable(getattr(sc, "n_samples_seen_", None)),
                    "model_state_dict": best_state,  # already CPU tensors
                }

        # NEW: save best run ("fold") weights for this property into a new directory
        if best_overall_payload is not None and "model_state_dict" in best_overall_payload:
            os.makedirs(BEST_WEIGHTS_DIR, exist_ok=True)
            prop_dir = os.path.join(BEST_WEIGHTS_DIR, _sanitize_name(pname))
            os.makedirs(prop_dir, exist_ok=True)

            # Save a single checkpoint bundle + a lightweight metadata json
            ckpt_bundle = {
                k: v for k, v in best_overall_payload.items()
                if k != "test_metrics"
            }
            ckpt_bundle["test_metrics"] = best_overall_payload["test_metrics"]

            torch.save(ckpt_bundle, os.path.join(prop_dir, "best_run_checkpoint.pt"))

            meta = {k: v for k, v in best_overall_payload.items() if k != "model_state_dict"}
            with open(os.path.join(prop_dir, "best_run_metadata.json"), "w") as fh:
                fh.write(json.dumps(make_json_serializable(meta), indent=2))

            print(f"[INFO] Saved best weights for '{pname}' to: {prop_dir}")
            print(f"[INFO]   best_run={best_overall_payload['best_run']}  best_test_r2={best_overall_payload['test_metrics'].get('r2', None)}")

        # Aggregate across runs
        if run_metrics:
            r2s = [x["r2"] for x in run_metrics]
            maes = [x["mae"] for x in run_metrics]
            rmses = [x["rmse"] for x in run_metrics]
            mses = [x["mse"] for x in run_metrics]
            agg = {
                "r2": {"mean": float(np.mean(r2s)), "std": float(np.std(r2s, ddof=0))},
                "mae": {"mean": float(np.mean(maes)), "std": float(np.std(maes, ddof=0))},
                "rmse": {"mean": float(np.mean(rmses)), "std": float(np.std(rmses, ddof=0))},
                "mse": {"mean": float(np.mean(mses)), "std": float(np.std(mses, ddof=0))},
            }
        else:
            agg = None

        all_results["per_property"][pname] = {
            "property_col": pcol,
            "n_samples": len(samples),
            "runs": run_records,
            "agg": agg
        }

        with open(output_file, "a") as fh:
            fh.write("AGG_PROPERTY: " + json.dumps(make_json_serializable({pname: agg})) + "\n")

    return all_results


# -----------------------------
# Main
# -----------------------------
def main():
    if os.path.exists(OUTPUT_RESULTS):
        backup = OUTPUT_RESULTS + ".bak"
        shutil.copy(OUTPUT_RESULTS, backup)
        print(f"[INFO] Existing {OUTPUT_RESULTS} backed up to {backup}")
    open(OUTPUT_RESULTS, "w").close()

    if not os.path.isfile(POLYINFO_PATH):
        raise FileNotFoundError(f"PolyInfo file not found at {POLYINFO_PATH}")
    polyinfo_raw = pd.read_csv(POLYINFO_PATH, engine="python")

    found = find_property_columns(polyinfo_raw.columns)
    prop_map = {req: col for req, col in found.items()}
    print(f"[INFO] Property-to-column map: {prop_map}")

    property_list = []
    property_cols = []
    for req in REQUESTED_PROPERTIES:
        col = prop_map.get(req)
        if col is None:
            print(f"[WARN] Could not find a column for requested property '{req}'. Skipping.")
            continue
        property_list.append(req)
        property_cols.append(col)

    print(f"\n=== Running Uni-Poly MATCHED downstream fine-tuning for properties: {property_list} ===")
    overall = run_unipoly_downstream(property_list, property_cols, polyinfo_raw, PRETRAINED_MULTIMODAL_DIR, OUTPUT_RESULTS)

    # Write final summary (aggregated per property)
    final_agg = {}
    if overall and "per_property" in overall:
        for pname, info in overall["per_property"].items():
            final_agg[pname] = info.get("agg", None)

    with open(OUTPUT_RESULTS, "a") as fh:
        fh.write("\nFINAL_SUMMARY\n")
        fh.write(json.dumps(make_json_serializable(final_agg), indent=2))
        fh.write("\n")

    print(f"\n[DONE] Results appended to {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
