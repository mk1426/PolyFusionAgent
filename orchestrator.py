#!/usr/bin/env python3
"""
Graph-to-SMILES Generation Using Pretrained Multimodal Contrastive Learning (CL) Encoder
This script uses the four-encoder CL system (GINE, SchNet, DebertaV2, Fingerprint Transformer)
with pretrained weights from multimodal contrastive learning for conditional generation.
"""
import os
import time
import random
import json
import gc
import sys
import csv
import pickle
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Increase csv field size limit safely
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# PyG for geometry batching
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, GINEConv
from torch_geometric.nn.models import SchNet as PyGSchNet
from torch_geometric.nn import radius_graph

# Transformers
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer

# RDKit optional utilities for validity/canonicalization
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, DataStructs
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

# ---------------- Configuration ----------------
BASE_DIR = "Polymer_Foundational_Model/Datasets"
POLYINFO_CSV_PATH = os.path.join("Polymer_Foundational_Model", "polyinfo_with_modalities.csv")
POLYINFO_XLSX_PATH = os.path.join("Polymer_Foundational_Model", "Datasets/PolyInfo.xlsx")

# Pretrained CL encoder checkpoints (from multimodal contrastive learning)
BEST_MULTIMODAL_DIR = "./multimodal_output/best"  # Main CL model
BEST_GINE_DIR = "./gin_output/best"
BEST_SCHNET_DIR = "./schnet_output/best"
BEST_FP_DIR = "./fingerprint_mlm_output/best"
BEST_PSMILES_DIR = "./polybert_output/best"

RESULTS_TXT = "generation_cl_results.txt"
GEN_OUTPUT_DIR = "gen_cl_outputs"

MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
GRAD_ACCUM_STEPS = 2
NUM_WORKERS = 2
SEED = 42

NUM_SAMPLES_PER_CONDITION = 10
TOP_P = 0.95
TEMPERATURE = 1.0

COMPUTE_DIVERSITY_ON_N = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

sp = None
BOS_ID = None
EOS_ID = None
PAD_ID = None
VOCAB_SIZE = None

# CL encoder hyperparameters (from multimodal_pretrain_streaming.py)
P_MASK = 0.15
MAX_ATOMIC_Z = 85
MASK_ATOM_ID = MAX_ATOMIC_Z + 1
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

# PSMILES/Deberta params
DEBERTA_HIDDEN = 600
PSMILES_MAX_LEN = 128

# Contrastive embedding dimension
CL_EMB_DIM = 600

# ---------------- Helper: safe_get & match_edge_attr ----------------
def safe_get(d: dict, key: str, default=None):
    return d[key] if (isinstance(d, dict) and key in d) else default

def match_edge_attr_to_index(edge_index: torch.Tensor, edge_attr: torch.Tensor, target_dim: int = 3):
    """
    Ensure edge_attr has shape [E_index, D]. Handles common mismatches.
    """
    dev = None
    if edge_attr is not None and hasattr(edge_attr, "device"):
        dev = edge_attr.device
    elif edge_index is not None and hasattr(edge_index, "device"):
        dev = edge_index.device
    else:
        dev = torch.device("cpu")

    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((0, target_dim), dtype=torch.float, device=dev)
    E_idx = edge_index.size(1)
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.zeros((E_idx, target_dim), dtype=torch.float, device=dev)

    E_attr = edge_attr.size(0)
    if E_attr == E_idx:
        if edge_attr.size(1) != target_dim:
            D = edge_attr.size(1)
            if D < target_dim:
                pad = torch.zeros((E_attr, target_dim - D), dtype=torch.float, device=edge_attr.device)
                return torch.cat([edge_attr, pad], dim=1)
            else:
                return edge_attr[:, :target_dim]
        return edge_attr

    if E_attr * 2 == E_idx:
        try:
            return torch.cat([edge_attr, edge_attr], dim=0)
        except Exception:
            pass

    reps = (E_idx + E_attr - 1) // E_attr
    edge_rep = edge_attr.repeat(reps, 1)[:E_idx]
    if edge_rep.size(1) != target_dim:
        D = edge_rep.size(1)
        if D < target_dim:
            pad = torch.zeros((E_idx, target_dim - D), dtype=torch.float, device=edge_rep.device)
            edge_rep = torch.cat([edge_rep, pad], dim=1)
        else:
            edge_rep = edge_rep[:, :target_dim]
    return edge_rep

# ---------------- Parse graph, geometry, fingerprint, psmiles ----------------
def parse_graph_data(graph_json_str):
    """Parse graph JSON to PyG Data object."""
    try:
        if isinstance(graph_json_str, str):
            graph_field = json.loads(graph_json_str)
        else:
            graph_field = graph_json_str

        node_features = safe_get(graph_field, "node_features", None)
        if not node_features:
            return None

        atomic_nums = []
        chirality_vals = []
        formal_charges = []

        for nf in node_features:
            an = safe_get(nf, "atomic_num", None)
            if an is None:
                an = safe_get(nf, "atomic_number", 0)
            ch = safe_get(nf, "chirality", 0)
            fc = safe_get(nf, "formal_charge", 0)
            atomic_nums.append(int(an))
            chirality_vals.append(float(ch))
            formal_charges.append(float(fc))

        n_nodes = len(atomic_nums)
        if n_nodes == 0:
            return None

        edge_indices_raw = safe_get(graph_field, "edge_indices", None)
        edge_features_raw = safe_get(graph_field, "edge_features", None)

        if edge_indices_raw is None:
            adj_mat = safe_get(graph_field, "adjacency_matrix", None)
            if adj_mat:
                srcs = []
                dsts = []
                for i, row in enumerate(adj_mat):
                    for j, val in enumerate(row):
                        if val:
                            srcs.append(int(i))
                            dsts.append(int(j))
                edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
                E = edge_index.size(1)
                edge_attr = torch.zeros((E, 3), dtype=torch.float)
            else:
                return None
        else:
            srcs = []
            dsts = []
            try:
                if isinstance(edge_indices_raw, list) and len(edge_indices_raw) > 0 and isinstance(edge_indices_raw[0], list):
                    if all(isinstance(pair, list) and len(pair) == 2 for pair in edge_indices_raw):
                        srcs = [int(p[0]) for p in edge_indices_raw]
                        dsts = [int(p[1]) for p in edge_indices_raw]
                    elif len(edge_indices_raw) >= 2 and all(isinstance(x, (int, np.integer)) for x in edge_indices_raw[0]):
                        srcs = [int(x) for x in edge_indices_raw[0]]
                        dsts = [int(x) for x in edge_indices_raw[1]]
            except Exception:
                srcs = []
                dsts = []

            if len(srcs) == 0:
                return None

            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

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
                try:
                    edge_attr = torch.tensor(np.stack([bond_types, stereos, is_conjs], axis=1), dtype=torch.float)
                except Exception:
                    E = edge_index.size(1)
                    edge_attr = torch.zeros((E, 3), dtype=torch.float)
            else:
                E = edge_index.size(1)
                edge_attr = torch.zeros((E, 3), dtype=torch.float)

        edge_attr = match_edge_attr_to_index(edge_index, edge_attr, target_dim=3)

        data = Data()
        data.z = torch.tensor(atomic_nums, dtype=torch.long)
        data.chirality = torch.tensor(chirality_vals, dtype=torch.float)
        data.formal_charge = torch.tensor(formal_charges, dtype=torch.float)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.num_nodes = n_nodes
        return data
    except Exception as e:
        print("Parse error:", e)
        return None

def parse_geometry_data(geometry_json_str):
    """Parse geometry JSON to atomic numbers and 3D coordinates."""
    try:
        if isinstance(geometry_json_str, str):
            geom = json.loads(geometry_json_str)
        else:
            geom = geometry_json_str
        conf = geom.get("best_conformer") if isinstance(geom, dict) else None
        if conf:
            atomic = conf.get("atomic_numbers", [])
            coords = conf.get("coordinates", [])
            if len(atomic) == len(coords) and len(atomic) > 0:
                return {"atomic": atomic, "coords": coords}
    except Exception:
        pass
    return None

def parse_fingerprint_data(fp_json_str):
    """Parse fingerprint JSON to list of 0/1 bits."""
    if fp_json_str is None or (isinstance(fp_json_str, str) and fp_json_str.strip() == ""):
        return [0] * FP_LENGTH
    try:
        fp_json = json.loads(fp_json_str) if isinstance(fp_json_str, str) else fp_json_str
    except Exception:
        try:
            fp_json = json.loads(str(fp_json_str).replace("'", '"'))
        except Exception:
            parts = [p.strip().strip('"').strip("'") for p in str(fp_json_str).split(",")]
            bits = [1 if p in ("1", "True", "true") else 0 for p in parts[:FP_LENGTH]]
            if len(bits) < FP_LENGTH:
                bits += [0] * (FP_LENGTH - len(bits))
            return bits[:FP_LENGTH]

    bits = safe_get(fp_json, "morgan_r3_bits", None) if isinstance(fp_json, dict) else (fp_json if isinstance(fp_json, list) else None)
    if bits is None:
        return [0] * FP_LENGTH

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
    return normalized[:FP_LENGTH]

# ---------------- CL Encoder Components (from multimodal_pretrain_streaming.py) ----------------
class GineBlock(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.conv = GINEConv(self.mlp)
        self.bn = nn.BatchNorm1d(node_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = self.act(x)
        return x

class GineEncoder(nn.Module):
    def __init__(self, node_emb_dim=NODE_EMB_DIM, edge_emb_dim=EDGE_EMB_DIM, num_layers=NUM_GNN_LAYERS, max_atomic_z=MAX_ATOMIC_Z):
        super().__init__()
        self.atom_emb = nn.Embedding(num_embeddings=MASK_ATOM_ID+1, embedding_dim=node_emb_dim, padding_idx=None)
        self.node_attr_proj = nn.Sequential(
            nn.Linear(2, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, node_emb_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim)
        )
        if edge_emb_dim != node_emb_dim:
            self._edge_to_node_proj = nn.Linear(edge_emb_dim, node_emb_dim)
        else:
            self._edge_to_node_proj = None
        self.gnn_layers = nn.ModuleList([GineBlock(node_emb_dim) for _ in range(num_layers)])
        self.pool_proj = nn.Linear(node_emb_dim, node_emb_dim)
        self.node_classifier = nn.Linear(node_emb_dim, MASK_ATOM_ID+1)

    def _compute_node_reps(self, z, chirality, formal_charge, edge_index, edge_attr):
        device = next(self.parameters()).device
        atom_embedding = self.atom_emb(z.to(device))
        if chirality is None or formal_charge is None:
            node_attr = torch.zeros((z.size(0), 2), device=device)
        else:
            node_attr = torch.stack([chirality, formal_charge], dim=1).to(atom_embedding.device)
        node_attr_emb = self.node_attr_proj(node_attr)
        x = atom_embedding + node_attr_emb
        if edge_attr is None or edge_attr.numel() == 0:
            edge_emb = torch.zeros((0, EDGE_EMB_DIM), dtype=torch.float, device=x.device)
        else:
            edge_emb = self.edge_encoder(edge_attr.to(x.device))
        if self._edge_to_node_proj is not None and edge_emb.numel() > 0:
            edge_for_conv = self._edge_to_node_proj(edge_emb)
        else:
            edge_for_conv = edge_emb

        h = x
        for layer in self.gnn_layers:
            h = layer(h, edge_index.to(h.device), edge_for_conv)
        return h

    def forward(self, z, chirality, formal_charge, edge_index, edge_attr, batch=None):
        h = self._compute_node_reps(z, chirality, formal_charge, edge_index, edge_attr)
        if batch is None:
            pooled = torch.mean(h, dim=0, keepdim=True)
        else:
            bsize = int(batch.max().item() + 1) if batch.numel() > 0 else 1
            pooled = torch.zeros((bsize, h.size(1)), device=h.device)
            for i in range(bsize):
                mask = batch == i
                if mask.sum() == 0:
                    continue
                pooled[i] = h[mask].mean(dim=0)
        return self.pool_proj(pooled)

class NodeSchNetWrapper(nn.Module):
    def __init__(self, hidden_channels=SCHNET_HIDDEN, num_interactions=SCHNET_NUM_INTERACTIONS, num_gaussians=SCHNET_NUM_GAUSSIANS, cutoff=SCHNET_CUTOFF, max_num_neighbors=SCHNET_MAX_NEIGHBORS):
        super().__init__()
        self.schnet = PyGSchNet(hidden_channels=hidden_channels, num_filters=hidden_channels, num_interactions=num_interactions, num_gaussians=num_gaussians, cutoff=cutoff, max_num_neighbors=max_num_neighbors)
        self.pool_proj = nn.Linear(hidden_channels, hidden_channels)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.node_classifier = nn.Linear(hidden_channels, MASK_ATOM_ID+1)

    def forward(self, z, pos, batch=None):
        device = next(self.parameters()).device
        z = z.to(device)
        pos = pos.to(device)
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        else:
            batch = batch.to(device)

        try:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        except Exception:
            edge_index = None

        node_h = None
        try:
            if hasattr(self.schnet, "embedding"):
                node_h = self.schnet.embedding(z)
        except Exception:
            node_h = None

        if node_h is not None and edge_index is not None and edge_index.numel() > 0:
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            edge_attr = None
            if hasattr(self.schnet, "distance_expansion"):
                try:
                    edge_attr = self.schnet.distance_expansion(edge_weight)
                except Exception:
                    edge_attr = None
            if edge_attr is None and hasattr(self.schnet, "gaussian_smearing"):
                try:
                    edge_attr = self.schnet.gaussian_smearing(edge_weight)
                except Exception:
                    edge_attr = None
            if hasattr(self.schnet, "interactions") and getattr(self.schnet, "interactions") is not None:
                for interaction in self.schnet.interactions:
                    try:
                        node_h = node_h + interaction(node_h, edge_index, edge_weight, edge_attr)
                    except TypeError:
                        node_h = node_h + interaction(node_h, edge_index, edge_weight)

        if node_h is None:
            try:
                out = self.schnet(z=z, pos=pos, batch=batch)
                if isinstance(out, torch.Tensor):
                    # Handle both 2D [N, D] and 3D [B, N, D] outputs
                    if out.dim() == 2 and out.size(0) == z.size(0):
                        node_h = out
                    elif out.dim() == 3:
                        # Flatten [B, N, D] to [B*N, D]
                        node_h = out.view(-1, out.size(-1))
                    else:
                        node_h = out
                elif hasattr(out, "last_hidden_state"):
                    node_h = out.last_hidden_state
                    if node_h.dim() == 3:
                        node_h = node_h.view(-1, node_h.size(-1))
                elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    cand = out[0]
                    if cand.dim() == 2 and cand.size(0) == z.size(0):
                        node_h = cand
                    elif cand.dim() == 3:
                        node_h = cand.view(-1, cand.size(-1))
            except Exception as e:
                raise RuntimeError("Failed to obtain node-level embeddings from PyG SchNet.") from e

        # Ensure node_h is 2D [N_total, D]
        if node_h.dim() == 3:
            node_h = node_h.view(-1, node_h.size(-1))

        bsize = int(batch.max().item()) + 1 if z.numel() > 0 else 1
        pooled = torch.zeros((bsize, node_h.size(1)), device=node_h.device)
        for i in range(bsize):
            mask = batch == i
            if mask.sum() == 0:
                continue
            pooled[i] = node_h[mask].mean(dim=0)
        return self.pool_proj(pooled)

class FingerprintEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE_FP, hidden_dim=256, seq_len=FP_LENGTH, num_layers=4, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seq_len = seq_len
        self.token_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(input_ids.device)
        else:
            key_padding_mask = None
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        if attention_mask is None:
            pooled = out.mean(dim=1)
        else:
            am = attention_mask.to(out.device).float().unsqueeze(-1)
            pooled = (out * am).sum(dim=1) / (am.sum(dim=1).clamp(min=1.0))
        return self.pool_proj(pooled)

class PSMILESDebertaEncoder(nn.Module):
    def __init__(self, model_dir_or_name: Optional[str] = None):
        super().__init__()
        try:
            if model_dir_or_name is not None and os.path.isdir(model_dir_or_name):
                self.model = DebertaV2ForMaskedLM.from_pretrained(model_dir_or_name)
            else:
                self.model = DebertaV2ForMaskedLM.from_pretrained("microsoft/deberta-v2-xlarge")
        except Exception as e:
            print("Warning: couldn't load DebertaV2 pretrained weights; initializing randomly.", e)
            from transformers import DebertaV2Config
            cfg = DebertaV2Config(vocab_size=300, hidden_size=DEBERTA_HIDDEN, num_attention_heads=12, num_hidden_layers=12, intermediate_size=512)
            self.model = DebertaV2ForMaskedLM(cfg)
        self.pool_proj = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            am = attention_mask.unsqueeze(-1).to(last_hidden.device).float()
            pooled = (last_hidden * am).sum(dim=1) / (am.sum(dim=1).clamp(min=1.0))
        return self.pool_proj(pooled)

class MultimodalContrastiveModel(nn.Module):
    """The four-encoder CL system from multimodal_pretrain_streaming.py"""
    def __init__(self,
                 gine_encoder: Optional[GineEncoder],
                 schnet_encoder: Optional[NodeSchNetWrapper],
                 fp_encoder: Optional[FingerprintEncoder],
                 psmiles_encoder: Optional[PSMILESDebertaEncoder],
                 emb_dim: int = CL_EMB_DIM):
        super().__init__()
        self.gine = gine_encoder
        self.schnet = schnet_encoder
        self.fp = fp_encoder
        self.psmiles = psmiles_encoder
        self.proj_gine = nn.Linear(getattr(self.gine, "pool_proj").out_features if self.gine is not None else emb_dim, emb_dim) if self.gine is not None else None
        self.proj_schnet = nn.Linear(getattr(self.schnet, "pool_proj").out_features if self.schnet is not None else emb_dim, emb_dim) if self.schnet is not None else None
        self.proj_fp = nn.Linear(getattr(self.fp, "pool_proj").out_features if self.fp is not None else emb_dim, emb_dim) if self.fp is not None else None
        self.proj_psmiles = nn.Linear(getattr(self.psmiles, "pool_proj").out_features if self.psmiles is not None else emb_dim, emb_dim) if self.psmiles is not None else None

    def encode(self, batch_mods: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        embs = {}
        B = None
        if 'gine' in batch_mods and self.gine is not None:
            g = batch_mods['gine']
            emb_g = self.gine(g['z'], g['chirality'], g['formal_charge'], g['edge_index'], g['edge_attr'], g.get('batch', None))
            embs['gine'] = F.normalize(self.proj_gine(emb_g), dim=-1)
            B = embs['gine'].size(0) if B is None else B
        if 'schnet' in batch_mods and self.schnet is not None:
            s = batch_mods['schnet']
            emb_s = self.schnet(s['z'], s['pos'], s.get('batch', None))
            embs['schnet'] = F.normalize(self.proj_schnet(emb_s), dim=-1)
            B = embs['schnet'].size(0) if B is None else B
        if 'fp' in batch_mods and self.fp is not None:
            f = batch_mods['fp']
            emb_f = self.fp(f['input_ids'], f.get('attention_mask', None))
            embs['fp'] = F.normalize(self.proj_fp(emb_f), dim=-1)
            B = embs['fp'].size(0) if B is None else B
        if 'psmiles' in batch_mods and self.psmiles is not None:
            p = batch_mods['psmiles']
            emb_p = self.psmiles(p['input_ids'], p.get('attention_mask', None))
            embs['psmiles'] = F.normalize(self.proj_psmiles(emb_p), dim=-1)
            B = embs['psmiles'].size(0) if B is None else B
        return embs

# ---------------- Simple Char-level Tokenizer ----------------
class SimpleCharTokenizer:
    def __init__(self, vocab_chars: List[str], special_tokens=("<pad>", "<s>", "</s>", "<unk>")):
        self.special_tokens = list(special_tokens)
        chars = [c for c in vocab_chars if c not in self.special_tokens]
        self.vocab = list(self.special_tokens) + chars
        self.piece_to_id = {p: i for i, p in enumerate(self.vocab)}
        self.id_to_piece = {i: p for i, p in enumerate(self.vocab)}

    @classmethod
    def from_texts(cls, texts: List[str], min_freq: int = 1):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            for ch in t:
                cnt[ch] += 1
        chars = [c for c, f in cnt.items() if f >= min_freq]
        return cls(chars)

    def encode(self, text: str, out_type=int):
        ids = []
        for ch in text:
            ids.append(self.piece_to_id.get(ch, self.piece_to_id.get("<unk>")))
        return ids

    def decode(self, ids: List[int]) -> str:
        pieces = [self.id_to_piece.get(i, "") for i in ids]
        out = "".join([p for p in pieces if p not in self.special_tokens])
        return out

    def PieceToId(self, piece: str) -> Optional[int]:
        return self.piece_to_id.get(piece, None)

    def IdToPiece(self, idx: int) -> str:
        return self.id_to_piece.get(idx, "")

    def get_piece_size(self) -> int:
        return len(self.vocab)

# ---------------- Tokenization utilities ----------------
def preprocess_psmiles(s: str) -> str:
    if not isinstance(s, str):
        return s
    import re
    s = re.sub(r'\[\*\]', r'*', s)
    return s

def encode_sequence(text: str, max_len: int) -> List[int]:
    ids = sp.encode(text, out_type=int)
    ids = ids[: max_len - 2]
    return [BOS_ID] + ids + [EOS_ID]

def pad_to_maxlen(ids: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(ids) < max_len:
        return ids + [pad_id] * (max_len - len(ids))
    return ids[:max_len]

# ---------------- Dataset + Collation for Generation ----------------
class GenCLDataset(Dataset):
    """Dataset that loads all four modalities for CL encoder."""
    def __init__(self, df: pd.DataFrame, prop_col: str, max_len: int, psmiles_tokenizer):
        self.df = df.reset_index(drop=True)
        self.prop_col = prop_col
        self.max_len = max_len
        self.psmiles_tokenizer = psmiles_tokenizer

        self.texts = [preprocess_psmiles(s) for s in self.df["psmiles"].astype(str).tolist()]
        self.props = self.df[prop_col].astype(float).values if prop_col in self.df.columns else np.zeros(len(self.df), dtype=float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prop_val = float(self.props[idx])
        text = self.texts[idx]
        ids = encode_sequence(text, self.max_len)
        ids = pad_to_maxlen(ids, self.max_len, PAD_ID)
        attn = [1 if t != PAD_ID else 0 for t in ids]

        row = self.df.iloc[idx]

        # Parse graph
        graph_data = parse_graph_data(row["graph"]) if "graph" in row and pd.notna(row["graph"]) else None

        # Parse geometry for SchNet
        geom_data = parse_geometry_data(row["geometry"]) if "geometry" in row and pd.notna(row["geometry"]) else None
        schnet_z = None
        schnet_pos = None
        if geom_data:
            schnet_z = torch.tensor(geom_data["atomic"], dtype=torch.long)
            schnet_pos = torch.tensor(geom_data["coords"], dtype=torch.float)

        # Parse fingerprint
        fp_bits = parse_fingerprint_data(row["fingerprints"]) if "fingerprints" in row else [0] * FP_LENGTH
        fp_ids = torch.tensor(fp_bits, dtype=torch.long)

        # Tokenize psmiles for DebertaV2
        psmiles_text = str(row["psmiles"]) if "psmiles" in row and pd.notna(row["psmiles"]) else ""
        psmiles_enc = self.psmiles_tokenizer(psmiles_text, truncation=True, padding="max_length", max_length=PSMILES_MAX_LEN)
        psmiles_ids = torch.tensor(psmiles_enc["input_ids"], dtype=torch.long)
        psmiles_attn = torch.tensor(psmiles_enc["attention_mask"], dtype=torch.bool)

        polymer_id = row.get("polymer_id", None) if "polymer_id" in row else None

        return {
            "prop": prop_val,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "graph": graph_data,
            "schnet_z": schnet_z,
            "schnet_pos": schnet_pos,
            "fp_ids": fp_ids,
            "psmiles_ids": psmiles_ids,
            "psmiles_attn": psmiles_attn,
            "polymer_id": polymer_id
        }

class GenCLCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        props = torch.tensor([b["prop"] for b in batch], dtype=torch.float32)
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attn_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        dec_in = input_ids[:, :-1]
        dec_attn = attn_mask[:, :-1]
        labels = input_ids[:, 1:].clone()
        labels[labels == self.pad_id] = -100

        # GINE graph batch
        graph_list = [b["graph"] for b in batch]
        fixed_graphs = []
        for g in graph_list:
            if g is None:
                empty = Data()
                empty.z = torch.tensor([], dtype=torch.long)
                empty.chirality = torch.tensor([], dtype=torch.float)
                empty.formal_charge = torch.tensor([], dtype=torch.float)
                empty.edge_index = torch.tensor([[], []], dtype=torch.long)
                empty.edge_attr = torch.zeros((0, 3), dtype=torch.float)
                empty.num_nodes = 0
                fixed_graphs.append(empty)
            else:
                fixed_graphs.append(g)
        graph_batch = Batch.from_data_list(fixed_graphs)

        # SchNet batch
        schnet_z_list = []
        schnet_pos_list = []
        schnet_batch_idx = []
        for i, b in enumerate(batch):
            if b["schnet_z"] is not None and b["schnet_pos"] is not None:
                schnet_z_list.append(b["schnet_z"])
                schnet_pos_list.append(b["schnet_pos"])
                schnet_batch_idx.append(torch.full((b["schnet_z"].size(0),), i, dtype=torch.long))
        if len(schnet_z_list) > 0:
            schnet_z = torch.cat(schnet_z_list, dim=0)
            schnet_pos = torch.cat(schnet_pos_list, dim=0)
            schnet_batch = torch.cat(schnet_batch_idx, dim=0)
        else:
            schnet_z = torch.tensor([], dtype=torch.long)
            schnet_pos = torch.tensor([], dtype=torch.float)
            schnet_batch = torch.tensor([], dtype=torch.long)

        # Fingerprint batch
        fp_ids = torch.stack([b["fp_ids"] for b in batch], dim=0)
        fp_attn = torch.ones_like(fp_ids, dtype=torch.bool)

        # PSMILES batch
        psmiles_ids = torch.stack([b["psmiles_ids"] for b in batch], dim=0)
        psmiles_attn = torch.stack([b["psmiles_attn"] for b in batch], dim=0)

        polymer_ids = [b.get("polymer_id", None) for b in batch]

        return {
            "prop": props,
            "decoder_input_ids": dec_in,
            "decoder_attention_mask": dec_attn,
            "labels": labels,
            "graph_batch": graph_batch,
            "schnet_z": schnet_z,
            "schnet_pos": schnet_pos,
            "schnet_batch": schnet_batch,
            "fp_ids": fp_ids,
            "fp_attn": fp_attn,
            "psmiles_ids": psmiles_ids,
            "psmiles_attn": psmiles_attn,
            "polymer_ids": polymer_ids
        }

# ---------------- Transformer Decoder ----------------
class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 8,
                 nhead: int = 10, ff_mult: int = 4, dropout: float = 0.1,
                 tie_embeddings: Optional[nn.Embedding] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_emb = tie_embeddings if tie_embeddings is not None else nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(4096, hidden_size)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_embeddings is not None:
            self.lm_head.weight = tie_embeddings.weight

    def _make_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(self, decoder_input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor,
                decoder_attention_mask: Optional[torch.Tensor] = None):
        B, Ld = decoder_input_ids.size()
        device = decoder_input_ids.device
        pos_ids = torch.arange(Ld, device=device).unsqueeze(0).expand(B, Ld)
        x = self.token_emb(decoder_input_ids) + self.pos_emb(pos_ids)
        tgt_mask = self._make_causal_mask(Ld, device)

        if decoder_attention_mask is not None:
            tgt_key_padding_mask = decoder_attention_mask == 0
        else:
            tgt_key_padding_mask = None
        y = self.decoder(
            tgt=x,
            memory=encoder_hidden_states,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None
        )
        y = self.ln_f(y)
        logits = self.lm_head(y)
        return logits

# ---------------- CL Conditional Generator ----------------
class CLConditionalGenerator(nn.Module):
    """
    Uses the pretrained four-encoder CL system to encode multimodal inputs,
    then conditions a transformer decoder on the fused CL embeddings.
    """
    def __init__(self, cl_encoder: MultimodalContrastiveModel, vocab_size: int,
                 hidden_size: int = 600, num_memory_tokens: int = 8, decoder_layers: int = 8):
        super().__init__()
        self.cl_encoder = cl_encoder
        self.hidden_size = hidden_size
        self.num_memory_tokens = num_memory_tokens

        # Project CL embedding (600-d) to memory tokens
        self.memory_proj = nn.Sequential(
            nn.Linear(CL_EMB_DIM, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * num_memory_tokens)
        )

        self.decoder = TransformerDecoderOnly(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            nhead=10,
            ff_mult=4,
            dropout=0.1,
            tie_embeddings=None
        )

    def encode_memory(self, batch_mods: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode all available modalities through CL encoder and fuse them.
        Returns memory tensor of shape [B, num_memory_tokens, hidden_size].
        """
        device = next(self.cl_encoder.parameters()).device

        # Get embeddings from all available modalities
        embs = self.cl_encoder.encode(batch_mods)

        if len(embs) == 0:
            # No valid modalities, return zero memory
            B = 1
            if 'gine' in batch_mods and 'batch' in batch_mods['gine']:
                B = int(batch_mods['gine']['batch'].max().item()) + 1 if batch_mods['gine']['batch'].numel() > 0 else 1
            memory = torch.zeros((B, self.num_memory_tokens, self.hidden_size), device=device)
            return memory

        # Average all available modality embeddings
        emb_list = [embs[k] for k in embs.keys()]
        fused_emb = torch.stack(emb_list, dim=0).mean(dim=0)  # [B, CL_EMB_DIM]

        # Project to memory tokens
        memory_flat = self.memory_proj(fused_emb)  # [B, hidden_size * num_memory_tokens]
        memory = memory_flat.view(fused_emb.size(0), self.num_memory_tokens, self.hidden_size)

        return memory

    def forward(self, batch_mods: Dict[str, torch.Tensor],
                decoder_input_ids: torch.Tensor,
                decoder_attention_mask: Optional[torch.Tensor]):
        memory = self.encode_memory(batch_mods)
        logits = self.decoder(decoder_input_ids, memory, decoder_attention_mask)
        return logits

# ---------------- Loss, training & evaluation utilities ----------------
def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    return loss

def train_one_epoch(model, loader, optimizer, scheduler, scaler=None):
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        dec_in = batch["decoder_input_ids"].to(DEVICE)
        dec_attn = batch["decoder_attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Prepare batch_mods for CL encoder
        batch_mods = {}

        # GINE
        graph_batch = batch["graph_batch"]
        batch_mods['gine'] = {
            'z': graph_batch.z,
            'chirality': graph_batch.chirality,
            'formal_charge': graph_batch.formal_charge,
            'edge_index': graph_batch.edge_index,
            'edge_attr': graph_batch.edge_attr,
            'batch': graph_batch.batch if hasattr(graph_batch, 'batch') else None
        }

        # SchNet
        if batch["schnet_z"].numel() > 0:
            batch_mods['schnet'] = {
                'z': batch["schnet_z"],
                'pos': batch["schnet_pos"],
                'batch': batch["schnet_batch"]
            }

        # Fingerprint
        batch_mods['fp'] = {
            'input_ids': batch["fp_ids"],
            'attention_mask': batch["fp_attn"]
        }

        # PSMILES
        batch_mods['psmiles'] = {
            'input_ids': batch["psmiles_ids"],
            'attention_mask': batch["psmiles_attn"]
        }

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(batch_mods, dec_in, dec_attn)
                loss = compute_loss(logits, labels) / max(1, GRAD_ACCUM_STEPS)
            scaler.scale(loss).backward()
        else:
            logits = model(batch_mods, dec_in, dec_attn)
            loss = compute_loss(logits, labels) / max(1, GRAD_ACCUM_STEPS)
            loss.backward()

        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * max(1, GRAD_ACCUM_STEPS)
        steps += 1
    return total_loss / max(steps, 1)

@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        dec_in = batch["decoder_input_ids"].to(DEVICE)
        dec_attn = batch["decoder_attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Prepare batch_mods
        batch_mods = {}
        graph_batch = batch["graph_batch"]
        batch_mods['gine'] = {
            'z': graph_batch.z,
            'chirality': graph_batch.chirality,
            'formal_charge': graph_batch.formal_charge,
            'edge_index': graph_batch.edge_index,
            'edge_attr': graph_batch.edge_attr,
            'batch': graph_batch.batch if hasattr(graph_batch, 'batch') else None
        }
        if batch["schnet_z"].numel() > 0:
            batch_mods['schnet'] = {
                'z': batch["schnet_z"],
                'pos': batch["schnet_pos"],
                'batch': batch["schnet_batch"]
            }
        batch_mods['fp'] = {
            'input_ids': batch["fp_ids"],
            'attention_mask': batch["fp_attn"]
        }
        batch_mods['psmiles'] = {
            'input_ids': batch["psmiles_ids"],
            'attention_mask': batch["psmiles_attn"]
        }

        logits = model(batch_mods, dec_in, dec_attn)
        loss = compute_loss(logits, labels)
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)

# ---------------- Sampling / Decoding functions ----------------
@torch.no_grad()
def sample_autoregressive(model: CLConditionalGenerator,
                          graph_data: Data,
                          schnet_z: Optional[torch.Tensor],
                          schnet_pos: Optional[torch.Tensor],
                          fp_ids: torch.Tensor,
                          psmiles_ids: torch.Tensor,
                          psmiles_attn: torch.Tensor,
                          max_len: int = MAX_LENGTH,
                          top_p: float = TOP_P,
                          temperature: float = TEMPERATURE) -> List[int]:
    model.eval()

    # Prepare batch_mods for single sample
    batch_mods = {}
    graph_batch = Batch.from_data_list([graph_data]).to(DEVICE)
    batch_mods['gine'] = {
        'z': graph_batch.z,
        'chirality': graph_batch.chirality,
        'formal_charge': graph_batch.formal_charge,
        'edge_index': graph_batch.edge_index,
        'edge_attr': graph_batch.edge_attr,
        'batch': graph_batch.batch if hasattr(graph_batch, 'batch') else None
    }

    if schnet_z is not None and schnet_pos is not None:
        # Ensure proper shapes for single sample
        if schnet_z.dim() == 1:
            schnet_z_batch = schnet_z
        else:
            schnet_z_batch = schnet_z.view(-1)

        if schnet_pos.dim() == 2:
            schnet_pos_batch = schnet_pos
        else:
            schnet_pos_batch = schnet_pos.view(-1, 3)

        batch_mods['schnet'] = {
            'z': schnet_z_batch.to(DEVICE),
            'pos': schnet_pos_batch.to(DEVICE),
            'batch': torch.zeros(schnet_z_batch.size(0), dtype=torch.long, device=DEVICE)
        }

    batch_mods['fp'] = {
        'input_ids': fp_ids.unsqueeze(0).to(DEVICE),
        'attention_mask': torch.ones_like(fp_ids.unsqueeze(0), dtype=torch.bool, device=DEVICE)
    }

    batch_mods['psmiles'] = {
        'input_ids': psmiles_ids.unsqueeze(0).to(DEVICE),
        'attention_mask': psmiles_attn.unsqueeze(0).to(DEVICE)
    }

    memory = model.encode_memory(batch_mods)
    cur = torch.tensor([[BOS_ID]], dtype=torch.long, device=DEVICE)

    for _ in range(max_len - 1):
        logits_all = model.decoder(cur, memory, None)
        pos_logits = logits_all[:, -1, :]
        logits = pos_logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        denom = sorted_probs.sum(dim=-1, keepdim=True)
        denom[denom == 0.0] = 1.0
        sorted_probs = sorted_probs / denom

        next_id_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(dim=-1, index=next_id_in_sorted)
        cur = torch.cat([cur, next_token], dim=1)
        if next_token.item() == EOS_ID:
            break
        if cur.size(1) >= max_len:
            break
    return cur.squeeze(0).tolist()

def decode_ids(ids: List[int]) -> str:
    tokens = []
    for t in ids:
        if t == BOS_ID:
            continue
        if t == EOS_ID:
            break
        tokens.append(t)
    try:
        text = sp.decode(tokens)
    except Exception:
        try:
            text = "".join([sp.IdToPiece(i) for i in tokens])
        except Exception:
            text = ""
    return text

# ---------------- Generation metrics ----------------
def rdkit_mol_from_smiles(s: str):
    try:
        return Chem.MolFromSmiles(s, sanitize=True)
    except Exception:
        return None

def is_valid_smiles(s: str) -> bool:
    if RDKit_AVAILABLE:
        return rdkit_mol_from_smiles(s) is not None
    return len(s) > 0 and s.count("[") == s.count("]")

def canonicalize_smiles(s: str) -> str:
    if RDKit_AVAILABLE:
        m = rdkit_mol_from_smiles(s)
        if m is None:
            return ""
        try:
            return Chem.MolToSmiles(m, isomericSmiles=True)
        except Exception:
            return ""
    return s

def morgan_fp(s: str):
    if RDKit_AVAILABLE:
        m = rdkit_mol_from_smiles(s)
        if m is None:
            return None
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
    return None

def internal_diversity(smiles_list: List[str]) -> float:
    if RDKit_AVAILABLE:
        fps = [morgan_fp(s) for s in smiles_list]
        fps = [f for f in fps if f is not None]
        if len(fps) < 2:
            return 0.0
        n = len(fps)
        total = 0.0
        cnt = 0
        for _ in range(min(COMPUTE_DIVERSITY_ON_N, n*(n-1)//2)):
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total += (1.0 - sim)
            cnt += 1
        return (total / cnt) if cnt > 0 else 0.0
    if len(smiles_list) < 2:
        return 0.0
    total = 0.0
    cnt = 0
    for _ in range(min(COMPUTE_DIVERSITY_ON_N, len(smiles_list)*(len(smiles_list)-1)//2)):
        a, b = random.sample(smiles_list, 2)
        L = max(len(a), len(b))
        if L == 0:
            continue
        mismatches = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        total += mismatches / L
        cnt += 1
    return (total / cnt) if cnt > 0 else 0.0

def compute_gen_metrics(generated: List[str], train_canon_set: set) -> Dict[str, float]:
    if len(generated) == 0:
        return {"validity": 0.0, "uniqueness": 0.0, "novelty": 0.0, "diversity": 0.0}
    valid = [s for s in generated if is_valid_smiles(s)]
    validity = 100.0 * len(valid) / len(generated)
    unique_valid = list(set(valid))
    uniqueness = 100.0 * len(unique_valid) / max(len(valid), 1)
    novel = []
    for s in unique_valid:
        can = canonicalize_smiles(s)
        if can and can not in train_canon_set:
            novel.append(s)
    novelty = 100.0 * len(novel) / max(len(unique_valid), 1)
    diversity = internal_diversity(unique_valid)
    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "diversity": diversity,
    }

# ---------------- Main generation pipeline ----------------
def main(smoke: bool = False):
    global sp, BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE

    os.makedirs(GEN_OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load dataset
    df_raw = None
    if os.path.exists(POLYINFO_CSV_PATH):
        print(f"Loading CSV dataset from {POLYINFO_CSV_PATH} ...")
        df_raw = pd.read_csv(POLYINFO_CSV_PATH)
    elif os.path.exists(POLYINFO_XLSX_PATH):
        print(f"Loading XLSX dataset from {POLYINFO_XLSX_PATH} ...")
        df_raw = pd.read_excel(POLYINFO_XLSX_PATH)
    else:
        raise FileNotFoundError(f"No dataset found. Please provide either {POLYINFO_CSV_PATH} or {POLYINFO_XLSX_PATH}.")

    print(f"Loaded dataset with {len(df_raw)} rows")

    # Verify required columns
    required_cols = ['graph', 'geometry', 'fingerprints', 'psmiles']
    for col in required_cols:
        if col not in df_raw.columns:
            raise ValueError(f"Dataset must contain '{col}' column for CL encoder.")

    structure_cols = [c for c in df_raw.columns if c.lower() in ['psmiles', 'smiles', 'bigsmiles']]
    if not structure_cols:
        possible = [c for c in df_raw.columns if 'smiles' in c.lower() or 'smile' in c.lower()]
        if possible:
            structure_cols = possible
    if not structure_cols:
        raise ValueError("No SMILES-like column found. Expected one of ['psmiles','smiles','bigsmiles'].")

    structure_col = structure_cols[0]
    print(f"Using structure column: {structure_col}")

    df_raw['psmiles'] = df_raw[structure_col].astype(str)

    # Build simple char tokenizer
    psmiles_texts = df_raw['psmiles'].astype(str).tolist()
    sp = SimpleCharTokenizer.from_texts(psmiles_texts)
    BOS_ID = sp.PieceToId("<s>") if sp.PieceToId("<s>") is not None else sp.PieceToId("<unk>")
    EOS_ID = sp.PieceToId("</s>") if sp.PieceToId("</s>") is not None else sp.PieceToId("<unk>")
    PAD_ID = sp.PieceToId("<pad>") if sp.PieceToId("<pad>") is not None else 0
    VOCAB_SIZE = sp.get_piece_size()

    print(f"Tokenizer vocab size: {VOCAB_SIZE} | BOS {BOS_ID} | EOS {EOS_ID} | PAD {PAD_ID}")

    # Load or create DebertaV2 tokenizer for psmiles
    try:
        SPM_MODEL = "spm.model"
        if os.path.exists(SPM_MODEL):
            psmiles_tokenizer = DebertaV2Tokenizer(vocab_file=SPM_MODEL, do_lower_case=False)
            psmiles_tokenizer.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})
            psmiles_tokenizer.pad_token = "<pad>"
            psmiles_tokenizer.mask_token = "<mask>"
        else:
            psmiles_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge", use_fast=False)
            psmiles_tokenizer.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})
            psmiles_tokenizer.pad_token = "<pad>"
            psmiles_tokenizer.mask_token = "<mask>"
    except Exception as e:
        print("Warning: Deberta tokenizer creation failed:", e)
        # Fallback to simple tokenizer
        class SimplePSMILESTokenizer:
            def __init__(self, max_len=PSMILES_MAX_LEN):
                chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-=#()[]@+/\\.")
                self.vocab = {c: i + 5 for i, c in enumerate(chars)}
                self.vocab["<pad>"] = 0
                self.vocab["<mask>"] = 1
                self.vocab["<unk>"] = 2
                self.vocab["<cls>"] = 3
                self.vocab["<sep>"] = 4
                self.mask_token = "<mask>"
                self.mask_token_id = self.vocab[self.mask_token]
                self.vocab_size = len(self.vocab)
                self.max_len = max_len

            def __call__(self, s, truncation=True, padding="max_length", max_length=None):
                max_len = max_length or self.max_len
                toks = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in list(s)][:max_len]
                attn = [1] * len(toks)
                if len(toks) < max_len:
                    pad = [self.vocab["<pad>"]] * (max_len - len(toks))
                    toks = toks + pad
                    attn = attn + [0] * (max_len - len(attn))
                return {"input_ids": toks, "attention_mask": attn}
        psmiles_tokenizer = SimplePSMILESTokenizer()

    # Load pretrained CL encoders
    print("\nLoading pretrained CL encoders...")
    gine_encoder = GineEncoder(node_emb_dim=NODE_EMB_DIM, edge_emb_dim=EDGE_EMB_DIM, num_layers=NUM_GNN_LAYERS, max_atomic_z=MAX_ATOMIC_Z)
    if os.path.exists(os.path.join(BEST_GINE_DIR, "pytorch_model.bin")):
        try:
            gine_encoder.load_state_dict(torch.load(os.path.join(BEST_GINE_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
            print(f"Loaded GINE weights from {BEST_GINE_DIR}")
        except Exception as e:
            print(f"Could not load GINE weights: {e}")
    gine_encoder.to(DEVICE)

    schnet_encoder = NodeSchNetWrapper(hidden_channels=SCHNET_HIDDEN, num_interactions=SCHNET_NUM_INTERACTIONS, num_gaussians=SCHNET_NUM_GAUSSIANS, cutoff=SCHNET_CUTOFF, max_num_neighbors=SCHNET_MAX_NEIGHBORS)
    if os.path.exists(os.path.join(BEST_SCHNET_DIR, "pytorch_model.bin")):
        try:
            schnet_encoder.load_state_dict(torch.load(os.path.join(BEST_SCHNET_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
            print(f"Loaded SchNet weights from {BEST_SCHNET_DIR}")
        except Exception as e:
            print(f"Could not load SchNet weights: {e}")
    schnet_encoder.to(DEVICE)

    fp_encoder = FingerprintEncoder(vocab_size=VOCAB_SIZE_FP, hidden_dim=256, seq_len=FP_LENGTH, num_layers=4, nhead=8, dim_feedforward=1024, dropout=0.1)
    if os.path.exists(os.path.join(BEST_FP_DIR, "pytorch_model.bin")):
        try:
            fp_encoder.load_state_dict(torch.load(os.path.join(BEST_FP_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
            print(f"Loaded fingerprint encoder weights from {BEST_FP_DIR}")
        except Exception as e:
            print(f"Could not load fingerprint weights: {e}")
    fp_encoder.to(DEVICE)

    psmiles_encoder = None
    if os.path.isdir(BEST_PSMILES_DIR):
        try:
            psmiles_encoder = PSMILESDebertaEncoder(model_dir_or_name=BEST_PSMILES_DIR)
            print(f"Loaded Deberta (PSMILES) from {BEST_PSMILES_DIR}")
        except Exception as e:
            print(f"Failed to load Deberta from saved directory: {e}")
    if psmiles_encoder is None:
        try:
            psmiles_encoder = PSMILESDebertaEncoder(model_dir_or_name=None)
        except Exception as e:
            print(f"Failed to instantiate Deberta encoder: {e}")
    if psmiles_encoder is not None:
        psmiles_encoder.to(DEVICE)

    # Build CL model
    cl_model = MultimodalContrastiveModel(gine_encoder, schnet_encoder, fp_encoder, psmiles_encoder, emb_dim=CL_EMB_DIM)
    if os.path.exists(os.path.join(BEST_MULTIMODAL_DIR, "pytorch_model.bin")):
        try:
            cl_model.load_state_dict(torch.load(os.path.join(BEST_MULTIMODAL_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
            print(f"Loaded CL model weights from {BEST_MULTIMODAL_DIR}")
        except Exception as e:
            print(f"Could not load CL model weights: {e}")
    cl_model.to(DEVICE)

    # Freeze CL encoder (only train decoder)
    for p in cl_model.parameters():
        p.requires_grad = False

    # Properties to iterate
    desired_properties = [
        "Density",
        "Glass transition temperature",
        "Melting temperature",
        "Specific volume",
        "Thermal decomposition temperature"
    ]
    numeric_cols = [c for c in desired_properties if c in df_raw.columns]
    if len(numeric_cols) == 0:
        df_raw['_dummy_prop_'] = 0.0
        numeric_cols = ['_dummy_prop_']

    results = []
    with open(RESULTS_TXT, "w") as f:
        f.write("CL-Encoder-to-Decoder Generation Results - Using Pretrained Four-Encoder CL System\n")
        f.write("="*80 + "\n\n")

    for prop in numeric_cols:
        print("\n" + "="*60)
        print(f"Processing generation for property: {prop}")
        print("="*60)

        df_prop = df_raw[['graph', 'geometry', 'fingerprints', 'psmiles', prop] + (['polymer_id'] if 'polymer_id' in df_raw.columns else [])].dropna(subset=['graph', 'psmiles', prop]).reset_index(drop=True)
        nsamples = len(df_prop)
        print(f"  Available samples: {nsamples}")
        if nsamples < 100:
            print(f"  Skipping {prop} (insufficient data: {nsamples} < 100)")
            with open(RESULTS_TXT, "a") as f:
                f.write(f"Property: {prop} - SKIPPED (insufficient data: {nsamples})\n\n")
            continue

        scaler = RobustScaler()
        prop_vals = df_prop[prop].astype(float).values.reshape(-1, 1)
        if len(prop_vals) > 0:
            scaler.fit(prop_vals)
            df_prop["_prop_std"] = scaler.transform(prop_vals).flatten()
        else:
            df_prop["_prop_std"] = 0.0

        train_df, temp_df = train_test_split(df_prop, test_size=0.2, random_state=SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

        train_canon_set = set()
        if RDKit_AVAILABLE:
            for s in train_df["psmiles"].astype(str):
                can = canonicalize_smiles(preprocess_psmiles(s))
                if can:
                    train_canon_set.add(can)
        else:
            train_canon_set = set(preprocess_psmiles(s) for s in train_df["psmiles"].astype(str))

        train_ds = GenCLDataset(train_df.rename(columns={"_prop_std": "prop"}), "prop", MAX_LENGTH, psmiles_tokenizer)
        val_ds = GenCLDataset(val_df.rename(columns={"_prop_std": "prop"}), "prop", MAX_LENGTH, psmiles_tokenizer)
        test_ds = GenCLDataset(test_df.rename(columns={"_prop_std": "prop"}), "prop", MAX_LENGTH, psmiles_tokenizer)

        collator = GenCLCollator(PAD_ID)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collator, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collator, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collator, pin_memory=True)

        model = CLConditionalGenerator(
            cl_encoder=cl_model,
            vocab_size=VOCAB_SIZE,
            hidden_size=600,
            num_memory_tokens=8,
            decoder_layers=8
        ).to(DEVICE)

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = max(1, ((len(train_dl) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS) * NUM_EPOCHS)
        def lr_lambda(step):
            if step < WARMUP_STEPS:
                return float(step) / float(max(1, WARMUP_STEPS))
            return max(0.0, (total_steps - step) / max(1, total_steps - WARMUP_STEPS))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        best_val = float("inf")
        best_state = None
        no_improve = 0
        print("Starting decoder training loop...")
        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, scaler=scaler_amp)
            val_loss = evaluate_loss(model, val_dl)
            dt = time.time() - t0
            print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | {dt:.1f}s")
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                no_improve = 0
                best_state = {
                    "model_state": model.state_dict(),
                    "scaler": scaler,
                    "prop": prop,
                    "best_val": best_val
                }
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print("Early stopping.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state["model_state"])
        model.eval()

        save_dir = os.path.join(GEN_OUTPUT_DIR, f"{prop.replace(' ', '_')}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "prop_name": prop,
            "scaler": scaler,
            "config": {
                "MAX_LENGTH": MAX_LENGTH,
                "TOP_P": TOP_P,
                "TEMPERATURE": TEMPERATURE,
            }
        }, os.path.join(save_dir, "checkpoint.pt"))

        # --- NEW: save tokenizer(s), decoder weights, CL encoder state and metadata ---
        try:
            # Save simple char tokenizer (sp) as pickle
            with open(os.path.join(save_dir, "sp_tokenizer.pkl"), "wb") as fh:
                pickle.dump(sp, fh)
        except Exception as e:
            print(f"Warning: failed to pickle sp tokenizer for {prop}: {e}")
            try:
                # fallback: save vocab mapping and special tokens
                meta_sp = {
                    "vocab": getattr(sp, "vocab", None),
                    "piece_to_id": getattr(sp, "piece_to_id", None),
                    "id_to_piece": getattr(sp, "id_to_piece", None),
                    "special_tokens": getattr(sp, "special_tokens", None)
                }
                with open(os.path.join(save_dir, "sp_tokenizer_meta.json"), "w") as fh:
                    json.dump(meta_sp, fh)
            except Exception as e2:
                print(f"Warning: failed to save sp tokenizer meta for {prop}: {e2}")

        try:
            # Save simple tokenizer metadata for easy loading
            sp_meta = {
                "VOCAB_SIZE": VOCAB_SIZE,
                "BOS_ID": BOS_ID,
                "EOS_ID": EOS_ID,
                "PAD_ID": PAD_ID
            }
            with open(os.path.join(save_dir, "sp_tokenizer_ids.json"), "w") as fh:
                json.dump(sp_meta, fh)
        except Exception as e:
            print(f"Warning: failed to save sp tokenizer ids for {prop}: {e}")

        try:
            # Save PSMILES tokenizer: prefer HuggingFace save_pretrained, fallback to pickle
            psmiles_dir = os.path.join(save_dir, "psmiles_tokenizer")
            if hasattr(psmiles_tokenizer, "save_pretrained"):
                try:
                    psmiles_tokenizer.save_pretrained(psmiles_dir)
                except Exception as e:
                    # fallback to pickle
                    with open(os.path.join(save_dir, "psmiles_tokenizer.pkl"), "wb") as fh:
                        pickle.dump(psmiles_tokenizer, fh)
            else:
                with open(os.path.join(save_dir, "psmiles_tokenizer.pkl"), "wb") as fh:
                    pickle.dump(psmiles_tokenizer, fh)
        except Exception as e:
            print(f"Warning: failed to save psmiles tokenizer for {prop}: {e}")

        try:
            # Save decoder weights separately for convenience
            torch.save(model.decoder.state_dict(), os.path.join(save_dir, "decoder_state.pt"))
        except Exception as e:
            print(f"Warning: failed to save decoder state for {prop}: {e}")

        try:
            # Save CL encoder state dict (so generation as a tool can load CL encoders easily)
            torch.save({
                "cl_state_dict": cl_model.state_dict()
            }, os.path.join(save_dir, "cl_encoder_state.pt"))
        except Exception as e:
            print(f"Warning: failed to save CL encoder state for {prop}: {e}")

        # --- end NEW saving block ---

        all_gens = []
        gen_records = []
        sample_rows = []
        for idx, row in test_df.iterrows():
            prop_val_raw = float(row[prop]) if prop in row and row[prop] is not None else 0.0
            prop_val_std = float(row["_prop_std"]) if "_prop_std" in row else prop_val_raw

            graph_data = parse_graph_data(row["graph"])
            if graph_data is None:
                continue

            geom_data = parse_geometry_data(row["geometry"]) if "geometry" in row and pd.notna(row["geometry"]) else None
            schnet_z = torch.tensor(geom_data["atomic"], dtype=torch.long) if geom_data else None
            schnet_pos = torch.tensor(geom_data["coords"], dtype=torch.float) if geom_data else None

            fp_bits = parse_fingerprint_data(row["fingerprints"]) if "fingerprints" in row else [0] * FP_LENGTH
            fp_ids = torch.tensor(fp_bits, dtype=torch.long)

            psmiles_text = str(row["psmiles"]) if "psmiles" in row and pd.notna(row["psmiles"]) else ""
            psmiles_enc = psmiles_tokenizer(psmiles_text, truncation=True, padding="max_length", max_length=PSMILES_MAX_LEN)
            psmiles_ids = torch.tensor(psmiles_enc["input_ids"], dtype=torch.long)
            psmiles_attn = torch.tensor(psmiles_enc["attention_mask"], dtype=torch.bool)

            gens = []
            for _ in range(NUM_SAMPLES_PER_CONDITION):
                ids = sample_autoregressive(model, graph_data, schnet_z, schnet_pos, fp_ids, psmiles_ids, psmiles_attn, max_len=MAX_LENGTH, top_p=TOP_P, temperature=TEMPERATURE)
                text = decode_ids(ids)
                text = preprocess_psmiles(text)
                gens.append(text)
                all_gens.append(text)
            gen_records.append({
                "polymer_id": row.get("polymer_id", None),
                "property_value": prop_val_raw,
                "generated": gens
            })
            if len(sample_rows) < 200:
                row_entry = {
                    "polymer_id": row.get("polymer_id", None),
                    "property_value": prop_val_raw
                }
                for k in range(len(gens)):
                    row_entry[f"gen_{k+1}"] = gens[k]
                sample_rows.append(row_entry)

        metrics = compute_gen_metrics(all_gens, train_canon_set)

        with open(os.path.join(save_dir, "generations.jsonl"), "w") as fw:
            for rec in gen_records:
                fw.write(json.dumps(rec) + "\n")
        if len(sample_rows) > 0:
            sample_df = pd.DataFrame(sample_rows)
            if 'polymer_id' not in sample_df.columns:
                sample_df['polymer_id'] = None
            sample_df.to_csv(os.path.join(save_dir, "sample_generations.csv"), index=False)
        else:
            pd.DataFrame(columns=['polymer_id', 'property_value']).to_csv(os.path.join(save_dir, "sample_generations.csv"), index=False)

        with open(RESULTS_TXT, "a") as f:
            f.write(f"Property: {prop}\n")
            f.write(f"  Samples: train={len(train_df)} val={len(val_df)} test={len(test_df)}\n")
            f.write(f"  Validity:  {metrics['validity']:.2f}%\n")
            f.write(f"  Uniqueness: {metrics['uniqueness']:.2f}%\n")
            f.write(f"  Novelty:    {metrics['novelty']:.2f}%\n")
            f.write(f"  Diversity:  {metrics['diversity']:.4f}\n")
            f.write("-" * 50 + "\n")

        print(f"[{prop}] Validity {metrics['validity']:.2f}% | Uniq {metrics['uniqueness']:.2f}% | Novel {metrics['novelty']:.2f}% | Div {metrics['diversity']:.4f}")

        del model, optimizer, scheduler, scaler_amp
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "="*80)
    print("All properties processed. Results saved to", RESULTS_TXT)
    print("="*80)

if __name__ == "__main__":
    main()
