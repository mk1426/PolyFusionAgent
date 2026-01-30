from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Optional / guarded imports (kept graceful)
try:
    from torch_geometric.nn import GINEConv
    from torch_geometric.nn.models import SchNet as PyGSchNet
    from torch_geometric.nn import radius_graph
except Exception:  # pragma: no cover
    GINEConv = None
    PyGSchNet = None
    radius_graph = None

# Transformers (guarded)
try:
    from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer, DebertaV2Config
except Exception:  # pragma: no cover
    DebertaV2ForMaskedLM = None
    DebertaV2Tokenizer = None
    DebertaV2Config = None

# =============================
# Hyperparameters / Config
# =============================
P_MASK: float = 0.15
MAX_ATOMIC_Z: int = 85
MASK_ATOM_ID: int = MAX_ATOMIC_Z + 1

# GINE
NODE_EMB_DIM: int = 300
EDGE_EMB_DIM: int = 300
NUM_GNN_LAYERS: int = 5

# SchNet
SCHNET_NUM_GAUSSIANS: int = 50
SCHNET_NUM_INTERACTIONS: int = 6
SCHNET_CUTOFF: float = 10.0
SCHNET_MAX_NEIGHBORS: int = 64
SCHNET_HIDDEN: int = 600

# Fingerprints (MLM over {0,1,mask})
FP_LENGTH: int = 2048
MASK_TOKEN_ID_FP: int = 2
VOCAB_SIZE_FP: int = 3

# PSMILES / DeBERTa
DEBERTA_HIDDEN: int = 600
PSMILES_MAX_LEN: int = 128

# Contrastive
TEMPERATURE: float = 0.07
REC_LOSS_WEIGHT: float = 1.0

# =============================
# Helpers
# =============================

def safe_get(d: dict, key: str, default=None):
    return d[key] if (isinstance(d, dict) and key in d) else default


def match_edge_attr_to_index(edge_index: torch.Tensor, edge_attr: torch.Tensor, target_dim: int = 3):
    """Robustly align edge_attr rows with edge_index columns.
    Pads/truncates features to target_dim.
    """
    # choose device
    dev = (
        edge_attr.device
        if (edge_attr is not None and hasattr(edge_attr, "device"))
        else (edge_index.device if (edge_index is not None and hasattr(edge_index, "device")) else torch.device("cpu"))
    )

    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((0, target_dim), dtype=torch.float, device=dev)
    E_idx = edge_index.size(1)
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.zeros((E_idx, target_dim), dtype=torch.float, device=dev)

    E_attr = edge_attr.size(0)
    if E_attr == E_idx:
        out = edge_attr
    elif E_attr * 2 == E_idx:
        try:
            out = torch.cat([edge_attr, edge_attr], dim=0)
        except Exception:
            reps = (E_idx + E_attr - 1) // E_attr
            out = edge_attr.repeat(reps, 1)[:E_idx]
    else:
        reps = (E_idx + E_attr - 1) // E_attr
        out = edge_attr.repeat(reps, 1)[:E_idx]

    if out.size(1) == target_dim:
        return out
    D = out.size(1)
    if D < target_dim:
        pad = torch.zeros((E_idx, target_dim - D), dtype=torch.float, device=dev)
        return torch.cat([out, pad], dim=1)
    return out[:, :target_dim]


# =============================
# Data handling
# =============================
class LazyMultimodalDataset(Dataset):
    """Loads each per-sample .pt (or .json) lazily on __getitem__.
    Expects keys per sample:
      'gine':   {node/edge lists}
      'schnet': {atomic, coords}
      'fp':     list[int] length L
      'psmiles_raw': str
    """

    def __init__(self, sample_file_list: List[str], tokenizer, fp_length: int = FP_LENGTH, psmiles_max_len: int = PSMILES_MAX_LEN):
        self.files = sample_file_list
        self.tokenizer = tokenizer
        self.fp_length = fp_length
        self.psmiles_max_len = psmiles_max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_path = self.files[idx]
        if sample_path.endswith(".pt"):
            sample = torch.load(sample_path, map_location="cpu")
        else:
            with open(sample_path, "r") as f:
                sample = json.load(f)

        # ---- GINE ----
        gine_raw = sample.get("gine", None)
        if gine_raw:
            node_atomic = torch.tensor(gine_raw.get("node_atomic", []), dtype=torch.long)
            node_chirality = torch.tensor(gine_raw.get("node_chirality", []), dtype=torch.float)
            node_charge = torch.tensor(gine_raw.get("node_charge", []), dtype=torch.float)
            edge_index = torch.tensor(gine_raw.get("edge_index", [[], []]), dtype=torch.long)
            ea_raw = gine_raw.get("edge_attr", None)
            edge_attr = torch.tensor(ea_raw, dtype=torch.float) if ea_raw else torch.zeros((edge_index.size(1), 3), dtype=torch.float)
            gine_item = {
                "z": node_atomic,
                "chirality": node_chirality,
                "formal_charge": node_charge,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "num_nodes": int(node_atomic.size(0)) if node_atomic.numel() > 0 else 0,
            }
        else:
            gine_item = {
                "z": torch.tensor([], dtype=torch.long),
                "chirality": torch.tensor([], dtype=torch.float),
                "formal_charge": torch.tensor([], dtype=torch.float),
                "edge_index": torch.tensor([[], []], dtype=torch.long),
                "edge_attr": torch.zeros((0, 3), dtype=torch.float),
                "num_nodes": 0,
            }

        # ---- SchNet ----
        schnet_raw = sample.get("schnet", None)
        if schnet_raw:
            s_z = torch.tensor(schnet_raw.get("atomic", []), dtype=torch.long)
            s_pos = torch.tensor(schnet_raw.get("coords", []), dtype=torch.float)
            schnet_item = {"z": s_z, "pos": s_pos}
        else:
            schnet_item = {"z": torch.tensor([], dtype=torch.long), "pos": torch.tensor([], dtype=torch.float)}

        # ---- FP ----
        fp_raw = sample.get("fp", None)
        if fp_raw is None:
            fp_vec = torch.zeros((self.fp_length,), dtype=torch.long)
        else:
            if isinstance(fp_raw, torch.Tensor):
                fp_vec = fp_raw.clone().to(torch.long)
            else:
                arr = list(fp_raw)[: self.fp_length]
                if len(arr) < self.fp_length:
                    arr += [0] * (self.fp_length - len(arr))
                fp_vec = torch.tensor(arr, dtype=torch.long)

        # ---- PSMILES ----
        psm_raw = sample.get("psmiles_raw", "") or ""
        enc = self.tokenizer(psm_raw, truncation=True, padding="max_length", max_length=self.psmiles_max_len)
        p_input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        p_attn = torch.tensor(enc["attention_mask"], dtype=torch.bool)

        return {
            "gine": gine_item,
            "schnet": schnet_item,
            "fp": {"input_ids": fp_vec},
            "psmiles": {"input_ids": p_input_ids, "attention_mask": p_attn},
        }


def multimodal_collate(batch_list: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Builds a batch for all modalities.
    Returns a dict with sub-dicts for 'gine', 'schnet', 'fp', 'psmiles'.
    """
    # ---- GINE batching ----
    all_z, all_ch, all_fc, all_ei, all_ea, batch_map = [], [], [], [], [], []
    node_offset = 0
    for i, item in enumerate(batch_list):
        g = item["gine"]
        z = g["z"]
        n = z.size(0)
        all_z.append(z)
        all_ch.append(g["chirality"])
        all_fc.append(g["formal_charge"])
        batch_map.append(torch.full((n,), i, dtype=torch.long))
        if g["edge_index"].numel() > 0:
            ei_off = g["edge_index"] + node_offset
            all_ei.append(ei_off)
            all_ea.append(match_edge_attr_to_index(g["edge_index"], g["edge_attr"], target_dim=3))
        node_offset += n

    if len(all_z) == 0:
        z_batch = torch.tensor([], dtype=torch.long)
        ch_batch = torch.tensor([], dtype=torch.float)
        fc_batch = torch.tensor([], dtype=torch.float)
        b_batch = torch.tensor([], dtype=torch.long)
        ei_batch = torch.empty((2, 0), dtype=torch.long)
        ea_batch = torch.zeros((0, 3), dtype=torch.float)
    else:
        z_batch = torch.cat(all_z, dim=0)
        ch_batch = torch.cat(all_ch, dim=0)
        fc_batch = torch.cat(all_fc, dim=0)
        b_batch = torch.cat(batch_map, dim=0)
        ei_batch = torch.cat(all_ei, dim=1) if len(all_ei) > 0 else torch.empty((2, 0), dtype=torch.long)
        ea_batch = torch.cat(all_ea, dim=0) if len(all_ea) > 0 else torch.zeros((0, 3), dtype=torch.float)

    # ---- SchNet batching ----
    all_sz, all_pos, s_batch = [], [], []
    for i, item in enumerate(batch_list):
        s = item["schnet"]
        if s["z"].numel() == 0:
            continue
        all_sz.append(s["z"])
        all_pos.append(s["pos"])
        s_batch.append(torch.full((s["z"].size(0),), i, dtype=torch.long))

    s_z_batch = torch.cat(all_sz, dim=0) if len(all_sz) > 0 else torch.tensor([], dtype=torch.long)
    s_pos_batch = torch.cat(all_pos, dim=0) if len(all_pos) > 0 else torch.tensor([], dtype=torch.float)
    s_b_batch = torch.cat(s_batch, dim=0) if len(s_batch) > 0 else torch.tensor([], dtype=torch.long)

    # ---- FP batching ----
    fp_ids = torch.stack([item["fp"]["input_ids"] for item in batch_list], dim=0)
    fp_attn = torch.ones_like(fp_ids, dtype=torch.bool)

    # ---- PSMILES batching ----
    p_ids = torch.stack([item["psmiles"]["input_ids"] for item in batch_list], dim=0)
    p_attn = torch.stack([item["psmiles"]["attention_mask"] for item in batch_list], dim=0)

    return {
        "gine": {"z": z_batch, "chirality": ch_batch, "formal_charge": fc_batch, "edge_index": ei_batch, "edge_attr": ea_batch, "batch": b_batch},
        "schnet": {"z": s_z_batch, "pos": s_pos_batch, "batch": s_b_batch},
        "fp": {"input_ids": fp_ids, "attention_mask": fp_attn},
        "psmiles": {"input_ids": p_ids, "attention_mask": p_attn},
    }


# =============================
# Masking utilities (reconstruction labels)
# =============================

def mask_batch_for_modality(batch: dict, modality: str, p_mask: float = P_MASK) -> dict:
    """Create masked variants + reconstruction labels for the chosen modality.
    Returns a batch-shaped dict mirroring the input structure, but with
    'labels' for the masked modality.
    """
    b = {}

    # GINE
    if "gine" in batch:
        z = batch["gine"]["z"].clone()
        chir = batch["gine"]["chirality"].clone()
        fc = batch["gine"]["formal_charge"].clone()
        edge_index = batch["gine"]["edge_index"]
        edge_attr = batch["gine"]["edge_attr"]
        batch_map = batch["gine"].get("batch", None)
        n_nodes = z.size(0)
        dev = z.device

        labels_z = torch.full_like(z, -100)
        if modality == "gine" and n_nodes > 0:
            sel = torch.rand(n_nodes, device=dev) < p_mask
            if sel.all():
                sel[torch.randint(0, n_nodes, (1,), device=dev)] = False
            labels_z[sel] = z[sel]
            probs = torch.rand(sel.sum().item(), device=dev)
            idx = torch.nonzero(sel).squeeze(-1)
            if idx.numel() > 0:
                mask_choice = probs < 0.8
                rand_choice = (probs >= 0.8) & (probs < 0.9)
                if mask_choice.any():
                    z[idx[mask_choice]] = MASK_ATOM_ID
                if rand_choice.any():
                    z[idx[rand_choice]] = torch.randint(1, MAX_ATOMIC_Z + 1, (int(rand_choice.sum().item()),), device=dev)
        b["gine"] = {"z": z, "chirality": chir, "formal_charge": fc, "edge_index": edge_index, "edge_attr": edge_attr, "batch": batch_map, "labels": labels_z}

    # SchNet
    if "schnet" in batch:
        z = batch["schnet"]["z"].clone()
        pos = batch["schnet"]["pos"].clone()
        batch_map = batch["schnet"].get("batch", None)
        n_nodes = z.size(0)
        dev = z.device
        labels_z = torch.full((n_nodes,), -100, dtype=torch.long, device=dev)
        if modality == "schnet" and n_nodes > 0:
            sel = torch.rand(n_nodes, device=dev) < p_mask
            if sel.all():
                sel[torch.randint(0, n_nodes, (1,), device=dev)] = False
            labels_z[sel] = z[sel]
            # Add noise or randomize positions (data augmentation-like masking)
            idx = torch.nonzero(sel).squeeze(-1)
            if idx.numel() > 0:
                probs = torch.rand(idx.size(0), device=dev)
                noisy_choice = probs < 0.8
                randpos_choice = (probs >= 0.8) & (probs < 0.9)
                if noisy_choice.any():
                    nidx = idx[noisy_choice]
                    pos[nidx] = pos[nidx] + torch.randn((nidx.size(0), 3), device=dev) * 0.5
                if randpos_choice.any():
                    ridx = idx[randpos_choice]
                    mins = pos.min(dim=0).values
                    maxs = pos.max(dim=0).values
                    pos[ridx] = (torch.rand((ridx.size(0), 3), device=dev) * (maxs - mins)) + mins
        b["schnet"] = {"z": z, "pos": pos, "batch": batch_map, "labels": labels_z}

    # Fingerprints
    if "fp" in batch:
        ids = batch["fp"]["input_ids"].clone()
        attn = batch["fp"].get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
        B, L = ids.shape
        labels = torch.full_like(ids, -100)
        if modality == "fp":
            for i in range(B):
                sel = torch.rand(L, device=ids.device) < p_mask
                if sel.all():
                    sel[torch.randint(0, L, (1,), device=ids.device)] = False
                labels[i, sel] = ids[i, sel]
                probs = torch.rand(sel.sum().item(), device=ids.device)
                idx = torch.nonzero(sel).squeeze(-1)
                if idx.numel() > 0:
                    mask_choice = probs < 0.8
                    rand_choice = (probs >= 0.8) & (probs < 0.9)
                    if mask_choice.any():
                        ids[i, idx[mask_choice]] = MASK_TOKEN_ID_FP
                    if rand_choice.any():
                        ids[i, idx[rand_choice]] = torch.randint(0, 2, (int(rand_choice.sum().item()),), device=ids.device)
        b["fp"] = {"input_ids": ids, "attention_mask": attn, "labels": labels}

    # PSMILES
    if "psmiles" in batch:
        ids = batch["psmiles"]["input_ids"].clone()
        attn = batch["psmiles"]["attention_mask"].clone()
        B, L = ids.shape
        labels = torch.full_like(ids, -100)
        if modality == "psmiles":
            # Resolve mask token id if possible
            mask_id = None
            # if a tokenizer-like object is passed via closure, user can set later
            # here we just use a common default if needed
            mask_id = 1  # fallback (will commonly map to <mask>)
            for i in range(B):
                sel = torch.rand(L, device=ids.device) < p_mask
                if sel.all():
                    sel[torch.randint(0, L, (1,), device=ids.device)] = False
                labels[i, sel] = ids[i, sel]
                probs = torch.rand(sel.sum().item(), device=ids.device)
                idx = torch.nonzero(sel).squeeze(-1)
                if idx.numel() > 0:
                    mask_choice = probs < 0.8
                    rand_choice = (probs >= 0.8) & (probs < 0.9)
                    if mask_choice.any():
                        ids[i, idx[mask_choice]] = mask_id
                    if rand_choice.any():
                        ids[i, idx[rand_choice]] = torch.randint(0, 300, (int(rand_choice.sum().item()),), device=ids.device)
        b["psmiles"] = {"input_ids": ids, "attention_mask": attn, "labels": labels}

    return b


def mm_batch_to_model_input(masked_batch: dict) -> dict:
    out = {}
    if "gine" in masked_batch:
        gm = masked_batch["gine"]
        out["gine"] = {
            "z": gm["z"],
            "chirality": gm["chirality"],
            "formal_charge": gm["formal_charge"],
            "edge_index": gm["edge_index"],
            "edge_attr": gm["edge_attr"],
            "batch": gm.get("batch", None),
            "labels": gm.get("labels", None),
        }
    if "schnet" in masked_batch:
        sm = masked_batch["schnet"]
        out["schnet"] = {"z": sm["z"], "pos": sm["pos"], "batch": sm.get("batch", None), "labels": sm.get("labels", None)}
    if "fp" in masked_batch:
        fm = masked_batch["fp"]
        out["fp"] = {"input_ids": fm["input_ids"], "attention_mask": fm.get("attention_mask", None), "labels": fm.get("labels", None)}
    if "psmiles" in masked_batch:
        pm = masked_batch["psmiles"]
        out["psmiles"] = {"input_ids": pm["input_ids"], "attention_mask": pm.get("attention_mask", None), "labels": pm.get("labels", None)}
    return out


# =============================
# Tokenizer helper (PSMILES)
# =============================

def build_psmiles_tokenizer(spm_path: Optional[str] = "spm.model", fallback_model: str = "microsoft/deberta-v2-xlarge", max_len: int = PSMILES_MAX_LEN):
    """Try to build a DeBERTa tokenizer; fallback to a simple char tokenizer.
    Returns an object with __call__(), .mask_token_id and .vocab_size when possible.
    """
    # Attempt HF DeBERTa tokenizer
    if DebertaV2Tokenizer is not None:
        try:
            if spm_path and Path(spm_path).exists():
                tok = DebertaV2Tokenizer(vocab_file=spm_path, do_lower_case=False)
            else:
                tok = DebertaV2Tokenizer.from_pretrained(fallback_model, use_fast=False)
            tok.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})
            tok.pad_token = "<pad>"
            tok.mask_token = "<mask>"
            return tok
        except Exception:
            pass

    # Fallback simple tokenizer
    class SimplePSMILESTokenizer:
        def __init__(self, max_len=max_len):
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
            L = max_length or self.max_len
            toks = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in list(s)][:L]
            attn = [1] * len(toks)
            if len(toks) < L:
                pad = [self.vocab["<pad>"]] * (L - len(toks))
                toks = toks + pad
                attn = attn + [0] * (L - len(attn))
            return {"input_ids": toks, "attention_mask": attn}

    return SimplePSMILESTokenizer()


# =============================
# Encoders
# =============================
class GineBlock(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        if GINEConv is None:
            raise RuntimeError("GINEConv is not available. Install torch_geometric.")
        self.mlp = nn.Sequential(nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim))
        self.conv = GINEConv(self.mlp)
        self.bn = nn.BatchNorm1d(node_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        return self.act(x)


class GineEncoder(nn.Module):
    def __init__(self, node_emb_dim=NODE_EMB_DIM, edge_emb_dim=EDGE_EMB_DIM, num_layers=NUM_GNN_LAYERS):
        super().__init__()
        self.atom_emb = nn.Embedding(num_embeddings=MASK_ATOM_ID + 1, embedding_dim=node_emb_dim)
        self.node_attr_proj = nn.Sequential(nn.Linear(2, node_emb_dim), nn.ReLU(), nn.Linear(node_emb_dim, node_emb_dim))
        self.edge_encoder = nn.Sequential(nn.Linear(3, edge_emb_dim), nn.ReLU(), nn.Linear(edge_emb_dim, edge_emb_dim))
        self._edge_to_node_proj = nn.Linear(edge_emb_dim, node_emb_dim) if edge_emb_dim != node_emb_dim else None
        self.gnn_layers = nn.ModuleList([GineBlock(node_emb_dim) for _ in range(num_layers)])
        self.pool_proj = nn.Linear(node_emb_dim, node_emb_dim)
        self.node_classifier = nn.Linear(node_emb_dim, MASK_ATOM_ID + 1)

    def _compute_node_reps(self, z, chirality, formal_charge, edge_index, edge_attr):
        atom_embedding = self.atom_emb(z)
        node_attr = torch.stack([chirality, formal_charge], dim=1) if chirality.numel() > 0 else torch.zeros((z.size(0), 2), device=z.device)
        node_attr_emb = self.node_attr_proj(node_attr)
        x = atom_embedding + node_attr_emb
        edge_emb = self.edge_encoder(edge_attr) if (edge_attr is not None and edge_attr.numel() > 0) else torch.zeros((0, EDGE_EMB_DIM), device=x.device)
        edge_for_conv = self._edge_to_node_proj(edge_emb) if (self._edge_to_node_proj is not None and edge_emb.numel() > 0) else edge_emb
        h = x
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_for_conv)
        return h

    def forward(self, z, chirality, formal_charge, edge_index, edge_attr, batch=None):
        h = self._compute_node_reps(z, chirality, formal_charge, edge_index, edge_attr)
        if batch is None or batch.numel() == 0:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            bsz = int(batch.max().item()) + 1
            pooled = torch.zeros((bsz, h.size(1)), device=h.device)
            for i in range(bsz):
                m = batch == i
                if m.any():
                    pooled[i] = h[m].mean(dim=0)
        return self.pool_proj(pooled)

    def node_logits(self, z, chirality, formal_charge, edge_index, edge_attr):
        h = self._compute_node_reps(z, chirality, formal_charge, edge_index, edge_attr)
        return self.node_classifier(h)


class NodeSchNetWrapper(nn.Module):
    def __init__(self, hidden_channels=SCHNET_HIDDEN, num_interactions=SCHNET_NUM_INTERACTIONS, num_gaussians=SCHNET_NUM_GAUSSIANS, cutoff=SCHNET_CUTOFF, max_num_neighbors=SCHNET_MAX_NEIGHBORS):
        super().__init__()
        if PyGSchNet is None:
            raise RuntimeError("PyG SchNet is not available. Install torch_geometric[full].")
        self.schnet = PyGSchNet(hidden_channels=hidden_channels, num_filters=hidden_channels, num_interactions=num_interactions, num_gaussians=num_gaussians, cutoff=cutoff, max_num_neighbors=max_num_neighbors)
        self.pool_proj = nn.Linear(hidden_channels, hidden_channels)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.node_classifier = nn.Linear(hidden_channels, MASK_ATOM_ID + 1)

    def forward(self, z, pos, batch=None):
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        try:
            ei = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        except Exception:
            ei = None

        node_h = None
        try:
            node_h = self.schnet.embedding(z)
        except Exception:
            node_h = None

        if node_h is not None and ei is not None and ei.numel() > 0:
            row, col = ei
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
            if getattr(self.schnet, "interactions", None) is not None:
                for inter in self.schnet.interactions:
                    try:
                        node_h = node_h + inter(node_h, ei, edge_weight, edge_attr)
                    except TypeError:
                        node_h = node_h + inter(node_h, ei, edge_weight)

        if node_h is None:
            out = self.schnet(z=z, pos=pos, batch=batch)
            if isinstance(out, torch.Tensor) and out.dim() == 2 and out.size(0) == z.size(0):
                node_h = out
            elif hasattr(out, "last_hidden_state"):
                node_h = out.last_hidden_state
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                node_h = out[0]
            else:
                raise RuntimeError("SchNet forward did not return node-level embeddings")

        bsz = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled = torch.zeros((bsz, node_h.size(1)), device=node_h.device)
        for i in range(bsz):
            m = batch == i
            if m.any():
                pooled[i] = node_h[m].mean(dim=0)
        return self.pool_proj(pooled)

    def node_logits(self, z, pos, batch=None):
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        try:
            ei = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        except Exception:
            ei = None

        node_h = None
        try:
            node_h = self.schnet.embedding(z)
        except Exception:
            node_h = None

        if node_h is not None and ei is not None and ei.numel() > 0:
            row, col = ei
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
            if getattr(self.schnet, "interactions", None) is not None:
                for inter in self.schnet.interactions:
                    try:
                        node_h = node_h + inter(node_h, ei, edge_weight, edge_attr)
                    except TypeError:
                        node_h = node_h + inter(node_h, ei, edge_weight)
        if node_h is None:
            out = self.schnet(z=z, pos=pos, batch=batch)
            if isinstance(out, torch.Tensor):
                node_h = out
            elif hasattr(out, "last_hidden_state"):
                node_h = out.last_hidden_state
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                node_h = out[0]
            else:
                raise RuntimeError("Unable to obtain node embeddings for SchNet node_logits")
        return self.node_classifier(node_h)


class FingerprintEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE_FP, hidden_dim=256, seq_len=FP_LENGTH, num_layers=4, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seq_len = seq_len
        self.token_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        kpm = ~attention_mask if attention_mask is not None else None
        out = self.transformer(x, src_key_padding_mask=kpm)
        if attention_mask is None:
            pooled = out.mean(dim=1)
        else:
            am = attention_mask.float().unsqueeze(-1)
            pooled = (out * am).sum(dim=1) / am.sum(dim=1).clamp(min=1.0)
        return self.pool_proj(pooled)

    def token_logits(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        kpm = ~attention_mask if attention_mask is not None else None
        out = self.transformer(x, src_key_padding_mask=kpm)
        return self.token_proj(out)


class PSMILESDebertaEncoder(nn.Module):
    def __init__(self, model_dir_or_name: Optional[str] = None, hidden_fallback: int = DEBERTA_HIDDEN, vocab_fallback: int = 300):
        super().__init__()
        if DebertaV2ForMaskedLM is None:
            # minimal random init fallback
            self.model = nn.Identity()
            self._hidden = hidden_fallback
            self._is_fallback = True
        else:
            try:
                if model_dir_or_name and os.path.isdir(model_dir_or_name):
                    self.model = DebertaV2ForMaskedLM.from_pretrained(model_dir_or_name)
                else:
                    self.model = DebertaV2ForMaskedLM.from_pretrained("microsoft/deberta-v2-xlarge")
                self._hidden = self.model.config.hidden_size
                self._is_fallback = False
            except Exception:
                cfg = DebertaV2Config(vocab_size=vocab_fallback, hidden_size=hidden_fallback, num_attention_heads=12, num_hidden_layers=12, intermediate_size=512)
                self.model = DebertaV2ForMaskedLM(cfg)
                self._hidden = hidden_fallback
                self._is_fallback = False
        self.pool_proj = nn.Linear(self._hidden, self._hidden)

    def forward(self, input_ids, attention_mask=None):
        if isinstance(self.model, nn.Identity):  # ultra-minimal fallback
            # Embed as one-hot projection to a fixed hidden size
            emb = F.one_hot(input_ids, num_classes=512).float()
            pooled = emb.mean(dim=1)
            return self.pool_proj(pooled)
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            am = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * am).sum(dim=1) / am.sum(dim=1).clamp(min=1.0)
        return self.pool_proj(pooled)

    def token_logits(self, input_ids, attention_mask=None, labels=None):
        if isinstance(self.model, nn.Identity):
            # No MLM when using Identity; return zeros loss/logits
            if labels is not None:
                return torch.zeros((), device=input_ids.device)
            return torch.zeros((*input_ids.shape, 1), device=input_ids.device)
        if labels is not None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            return outputs.loss
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.logits


# =============================
# Multimodal wrapper
# =============================
class MultimodalContrastiveModel(nn.Module):
    def __init__(self, gine_encoder: Optional[GineEncoder], schnet_encoder: Optional[NodeSchNetWrapper], fp_encoder: Optional[FingerprintEncoder], psmiles_encoder: Optional[PSMILESDebertaEncoder], emb_dim: int = 600):
        super().__init__()
        self.gine = gine_encoder
        self.schnet = schnet_encoder
        self.fp = fp_encoder
        self.psmiles = psmiles_encoder
        self.proj_gine = nn.Linear(getattr(self.gine, "pool_proj").out_features if self.gine is not None else emb_dim, emb_dim) if self.gine is not None else None
        self.proj_schnet = nn.Linear(getattr(self.schnet, "pool_proj").out_features if self.schnet is not None else emb_dim, emb_dim) if self.schnet is not None else None
        self.proj_fp = nn.Linear(getattr(self.fp, "pool_proj").out_features if self.fp is not None else emb_dim, emb_dim) if self.fp is not None else None
        self.proj_psmiles = nn.Linear(getattr(self.psmiles, "pool_proj").out_features if self.psmiles is not None else emb_dim, emb_dim) if self.psmiles is not None else None
        self.temperature = TEMPERATURE
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    def encode(self, batch_mods: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embs = {}
        if "gine" in batch_mods and self.gine is not None:
            g = batch_mods["gine"]
            emb_g = self.gine(g["z"], g["chirality"], g["formal_charge"], g["edge_index"], g["edge_attr"], g.get("batch", None))
            embs["gine"] = F.normalize(self.proj_gine(emb_g), dim=-1)
        if "schnet" in batch_mods and self.schnet is not None:
            s = batch_mods["schnet"]
            emb_s = self.schnet(s["z"], s["pos"], s.get("batch", None))
            embs["schnet"] = F.normalize(self.proj_schnet(emb_s), dim=-1)
        if "fp" in batch_mods and self.fp is not None:
            f = batch_mods["fp"]
            emb_f = self.fp(f["input_ids"], f.get("attention_mask", None))
            embs["fp"] = F.normalize(self.proj_fp(emb_f), dim=-1)
        if "psmiles" in batch_mods and self.psmiles is not None:
            p = batch_mods["psmiles"]
            emb_p = self.psmiles(p["input_ids"], p.get("attention_mask", None))
            embs["psmiles"] = F.normalize(self.proj_psmiles(emb_p), dim=-1)
        return embs

    def forward(self, batch_mods: Dict[str, torch.Tensor], mask_target: str):
        embs = self.encode(batch_mods)
        if mask_target not in embs:
            return torch.tensor(0.0, device=next(self.parameters()).device), {"batch_size": 0}
        target = embs[mask_target]
        other = [k for k in embs.keys() if k != mask_target]
        if len(other) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device), {"batch_size": target.size(0)}
        anchor = torch.stack([embs[k] for k in other], dim=0).mean(dim=0)
        logits = anchor @ target.T / self.temperature
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        info_nce_loss = F.cross_entropy(logits, labels)

        rec_losses = []
        info = {"info_nce_loss": float(info_nce_loss.detach().cpu().item())}

        try:
            if "gine" in batch_mods and self.gine is not None:
                gm = batch_mods["gine"]
                labels_nodes = gm.get("labels", None)
                if labels_nodes is not None:
                    node_logits = self.gine.node_logits(gm["z"], gm["chirality"], gm["formal_charge"], gm["edge_index"], gm["edge_attr"])
                    if labels_nodes.dim() == 1 and node_logits.size(0) == labels_nodes.size(0):
                        rec_losses.append(self.ce_loss(node_logits, labels_nodes.to(node_logits.device)))
        except Exception:
            pass

        try:
            if "schnet" in batch_mods and self.schnet is not None:
                sm = batch_mods["schnet"]
                labels_nodes = sm.get("labels", None)
                if labels_nodes is not None:
                    node_logits = self.schnet.node_logits(sm["z"], sm["pos"], sm.get("batch", None))
                    if labels_nodes.dim() == 1 and node_logits.size(0) == labels_nodes.size(0):
                        rec_losses.append(self.ce_loss(node_logits, labels_nodes.to(node_logits.device)))
        except Exception:
            pass

        try:
            if "fp" in batch_mods and self.fp is not None:
                fm = batch_mods["fp"]
                labels_fp = fm.get("labels", None)
                if labels_fp is not None:
                    tlog = self.fp.token_logits(fm["input_ids"], fm.get("attention_mask", None))
                    V = tlog.size(-1)
                    loss_fp = self.ce_loss(tlog.view(-1, V), labels_fp.view(-1).to(tlog.device))
                    rec_losses.append(loss_fp)
        except Exception:
            pass

        try:
            if "psmiles" in batch_mods and self.psmiles is not None:
                pm = batch_mods["psmiles"]
                labels_ps = pm.get("labels", None)
                if labels_ps is not None:
                    loss_ps = self.psmiles.token_logits(pm["input_ids"], pm.get("attention_mask", None), labels=labels_ps)
                    if isinstance(loss_ps, torch.Tensor):
                        rec_losses.append(loss_ps)
        except Exception:
            pass

        if rec_losses:
            rec_loss_total = sum(rec_losses) / len(rec_losses)
            total_loss = info_nce_loss + REC_LOSS_WEIGHT * rec_loss_total
            info.update({"reconstruction_loss": float(rec_loss_total.detach().cpu().item()), "total_loss": float(total_loss.detach().cpu().item())})
        else:
            total_loss = info_nce_loss
            info.update({"reconstruction_loss": 0.0, "total_loss": float(total_loss.detach().cpu().item())})
        return total_loss, info


# Public API
__all__ = [
    # config
    "P_MASK",
    "MAX_ATOMIC_Z",
    "MASK_ATOM_ID",
    "NODE_EMB_DIM",
    "EDGE_EMB_DIM",
    "NUM_GNN_LAYERS",
    "SCHNET_NUM_GAUSSIANS",
    "SCHNET_NUM_INTERACTIONS",
    "SCHNET_CUTOFF",
    "SCHNET_MAX_NEIGHBORS",
    "SCHNET_HIDDEN",
    "FP_LENGTH",
    "MASK_TOKEN_ID_FP",
    "VOCAB_SIZE_FP",
    "DEBERTA_HIDDEN",
    "PSMILES_MAX_LEN",
    "TEMPERATURE",
    "REC_LOSS_WEIGHT",
    # data
    "LazyMultimodalDataset",
    "multimodal_collate",
    "mask_batch_for_modality",
    "mm_batch_to_model_input",
    "build_psmiles_tokenizer",
    # encoders
    "GineEncoder",
    "NodeSchNetWrapper",
    "FingerprintEncoder",
    "PSMILESDebertaEncoder",
    # model
    "MultimodalContrastiveModel",
]
