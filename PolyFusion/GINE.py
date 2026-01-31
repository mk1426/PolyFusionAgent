"""
GINE-based masked pretraining on polymer graphs.
"""

import os
import json
import time
import sys
import csv
import argparse
from typing import Any, Dict, List, Optional, Tuple

# Increase max CSV field size limit
csv.field_size_limit(sys.maxsize)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

from torch_geometric.nn import GINEConv

# ---------------------------
# Configuration / Constants
# ---------------------------
P_MASK = 0.15
MAX_ATOMIC_Z = 85
MASK_ATOM_ID = MAX_ATOMIC_Z + 1

USE_LEARNED_WEIGHTING = True

NODE_EMB_DIM = 300
EDGE_EMB_DIM = 300
NUM_GNN_LAYERS = 5

K_ANCHORS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GINE masked pretraining (graphs).")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/path/to/polymer_structures_unified_processed.csv",
        help="Processed CSV containing a JSON 'graph' column.",
    )
    parser.add_argument("--target_rows", type=int, default=5_000_000, help="Max rows to parse.")
    parser.add_argument("--chunksize", type=int, default=50_000, help="CSV chunksize.")
    parser.add_argument("--output_dir", type=str, default="/path/to/gin_output_5M", help="Training output directory.")
    parser.add_argument("--num_workers", type=int, default=4, help="PyTorch DataLoader num workers.")
    return parser.parse_args()


# ---------------------------
# Helper functions
# ---------------------------

def safe_get(d: dict, key: str, default=None):
    return d[key] if (isinstance(d, dict) and key in d) else default


def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Adjacency list for BFS shortest paths."""
    adj = [[] for _ in range(num_nodes)]
    if edge_index is None or edge_index.numel() == 0:
        return adj
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
    return adj


def shortest_path_lengths_hops(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """
    All-pairs shortest path lengths in hops using BFS per node.
    Unreachable pairs get distance INF=num_nodes+1.
    """
    adj = build_adj_list(edge_index, num_nodes)
    INF = num_nodes + 1
    dist_mat = np.full((num_nodes, num_nodes), INF, dtype=np.int32)
    for s in range(num_nodes):
        q = [s]
        dist_mat[s, s] = 0
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj[u]:
                if dist_mat[s, v] == INF:
                    dist_mat[s, v] = dist_mat[s, u] + 1
                    q.append(v)
    return dist_mat


def match_edge_attr_to_index(edge_index: torch.Tensor, edge_attr: torch.Tensor, target_dim: int = 3) -> torch.Tensor:
    """
    Ensure edge_attr has shape [E_index, target_dim], handling common mismatches.
    """
    E_idx = edge_index.size(1) if (edge_index is not None and edge_index.numel() > 0) else 0
    if E_idx == 0:
        return torch.zeros((0, target_dim), dtype=torch.float)
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.zeros((E_idx, target_dim), dtype=torch.float)

    E_attr = edge_attr.size(0)
    if E_attr == E_idx:
        if edge_attr.size(1) != target_dim:
            D = edge_attr.size(1)
            if D < target_dim:
                pad = torch.zeros((E_attr, target_dim - D), dtype=torch.float, device=edge_attr.device)
                return torch.cat([edge_attr, pad], dim=1)
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


def parse_graphs_from_csv(csv_path: str, target_rows: int, chunksize: int):
    """
    Stream CSV and parse the JSON 'graph' field into graph tensors needed by the model.
    Returns lists of per-graph tensors.
    """
    node_atomic_lists = []
    node_chirality_lists = []
    node_charge_lists = []
    edge_index_lists = []
    edge_attr_lists = []
    num_nodes_list = []

    rows_read = 0

    for chunk in pd.read_csv(csv_path, engine="python", chunksize=chunksize):
        for _, row in chunk.iterrows():
            graph_field = None
            if "graph" in row and not pd.isna(row["graph"]):
                try:
                    graph_field = json.loads(row["graph"]) if isinstance(row["graph"], str) else row["graph"]
                except Exception:
                    graph_field = None
            else:
                continue

            if graph_field is None:
                continue

            node_features = safe_get(graph_field, "node_features", None)
            if not node_features:
                continue

            atomic_nums = []
            chirality_vals = []
            formal_charges = []
            for nf in node_features:
                an = safe_get(nf, "atomic_num", safe_get(nf, "atomic_number", 0))
                ch = safe_get(nf, "chirality", 0)
                fc = safe_get(nf, "formal_charge", 0)
                atomic_nums.append(int(an))
                chirality_vals.append(float(ch))
                formal_charges.append(float(fc))

            n_nodes = len(atomic_nums)

            edge_indices_raw = safe_get(graph_field, "edge_indices", None)
            edge_features_raw = safe_get(graph_field, "edge_features", None)

            if edge_indices_raw is None:
                adj_mat = safe_get(graph_field, "adjacency_matrix", None)
                if adj_mat:
                    srcs, dsts = [], []
                    for i, row_adj in enumerate(adj_mat):
                        for j, val in enumerate(row_adj):
                            if val:
                                srcs.append(i)
                                dsts.append(j)
                    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
                    E = edge_index.size(1)
                    edge_attr = torch.zeros((E, 3), dtype=torch.float)
                else:
                    continue
            else:
                srcs, dsts = [], []
                if isinstance(edge_indices_raw, list) and len(edge_indices_raw) > 0 and isinstance(edge_indices_raw[0], list):
                    if all(len(pair) == 2 and isinstance(pair[0], int) for pair in edge_indices_raw):
                        srcs = [int(p[0]) for p in edge_indices_raw]
                        dsts = [int(p[1]) for p in edge_indices_raw]
                    elif isinstance(edge_indices_raw[0][0], int):
                        try:
                            srcs = [int(x) for x in edge_indices_raw[0]]
                            dsts = [int(x) for x in edge_indices_raw[1]]
                        except Exception:
                            srcs, dsts = [], []
                if len(srcs) == 0:
                    continue

                edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

                if edge_features_raw and isinstance(edge_features_raw, list):
                    bond_types, stereos, is_conjs = [], [], []
                    for ef in edge_features_raw:
                        bt = safe_get(ef, "bond_type", 0)
                        st = safe_get(ef, "stereo", 0)
                        ic = safe_get(ef, "is_conjugated", False)
                        bond_types.append(float(bt))
                        stereos.append(float(st))
                        is_conjs.append(float(1.0 if ic else 0.0))
                    edge_attr = torch.tensor(np.stack([bond_types, stereos, is_conjs], axis=1), dtype=torch.float)
                else:
                    E = edge_index.size(1)
                    edge_attr = torch.zeros((E, 3), dtype=torch.float)

            edge_attr = match_edge_attr_to_index(edge_index, edge_attr, target_dim=3)

            node_atomic_lists.append(torch.tensor(atomic_nums, dtype=torch.long))
            node_chirality_lists.append(torch.tensor(chirality_vals, dtype=torch.float))
            node_charge_lists.append(torch.tensor(formal_charges, dtype=torch.float))
            edge_index_lists.append(edge_index)
            edge_attr_lists.append(edge_attr)
            num_nodes_list.append(n_nodes)

            rows_read += 1
            if rows_read >= target_rows:
                break
        if rows_read >= target_rows:
            break

    if len(node_atomic_lists) == 0:
        raise RuntimeError("No graphs were parsed from the CSV 'graph' column. Check input file and format.")

    print(f"Parsed {len(node_atomic_lists)} graphs (using 'graph' column). Using manual max atomic Z = {MAX_ATOMIC_Z}")
    return (
        node_atomic_lists,
        node_chirality_lists,
        node_charge_lists,
        edge_index_lists,
        edge_attr_lists,
        num_nodes_list,
    )


def compute_class_weights(train_atomic: List[torch.Tensor]) -> torch.Tensor:
    """Compute inverse-frequency class weights for atomic number prediction."""
    num_classes = MASK_ATOM_ID + 1
    counts = np.ones((num_classes,), dtype=np.float64)
    for z in train_atomic:
        vals = z.cpu().numpy().astype(int)
        for v in vals:
            if 0 <= v < num_classes:
                counts[v] += 1.0
    freq = counts / counts.sum()
    inv_freq = 1.0 / (freq + 1e-12)
    class_weights = inv_freq / inv_freq.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights[MASK_ATOM_ID] = 1.0
    return class_weights


class PolymerDataset(Dataset):
    """Holds per-graph tensors; collation builds a single batched graph with masking targets."""
    def __init__(self, atomic_list, chirality_list, charge_list, edge_index_list, edge_attr_list, num_nodes_list):
        self.atomic_list = atomic_list
        self.chirality_list = chirality_list
        self.charge_list = charge_list
        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list
        self.num_nodes_list = num_nodes_list

    def __len__(self):
        return len(self.atomic_list)

    def __getitem__(self, idx):
        return {
            "z": self.atomic_list[idx],
            "chirality": self.chirality_list[idx],
            "formal_charge": self.charge_list[idx],
            "edge_index": self.edge_index_list[idx],
            "edge_attr": self.edge_attr_list[idx],
            "num_nodes": int(self.num_nodes_list[idx]),
        }


def collate_batch(batch):
    """
    Build a single batched graph (node-concatenation with edge index offsets) and create:
      - masked node labels (labels_z)
      - hop-distance anchor targets (labels_dists) for masked nodes
    """
    all_z, all_ch, all_fc = [], [], []
    all_labels_z, all_labels_dists, all_labels_dists_mask = [], [], []
    batch_idx = []

    edge_index_list_batched = []
    edge_attr_list_batched = []
    node_offset = 0

    for i, g in enumerate(batch):
        z = g["z"]
        n = z.size(0)
        if n == 0:
            continue

        chir = g["chirality"]
        fc = g["formal_charge"]
        edge_index = g["edge_index"]
        edge_attr = g["edge_attr"]

        is_selected = torch.rand(n) < P_MASK
        if is_selected.all():
            is_selected[torch.randint(0, n, (1,))] = False

        labels_z = torch.full((n,), -100, dtype=torch.long)
        labels_dists = torch.zeros((n, K_ANCHORS), dtype=torch.float)
        labels_dists_mask = torch.zeros((n, K_ANCHORS), dtype=torch.bool)
        labels_z[is_selected] = z[is_selected]

        # BERT-style corruption on atomic numbers
        z_masked = z.clone()
        if is_selected.any():
            sel_idx = torch.nonzero(is_selected).squeeze(-1)
            rand_atomic = torch.randint(1, MAX_ATOMIC_Z + 1, (sel_idx.size(0),), dtype=torch.long)
            probs = torch.rand(sel_idx.size(0))
            mask_choice = probs < 0.8
            rand_choice = (probs >= 0.8) & (probs < 0.9)
            if mask_choice.any():
                z_masked[sel_idx[mask_choice]] = MASK_ATOM_ID
            if rand_choice.any():
                z_masked[sel_idx[rand_choice]] = rand_atomic[rand_choice]

        # Hop-distance targets for masked atoms (anchors = nearest visible nodes in hop distance)
        visible_idx = torch.nonzero(~is_selected).squeeze(-1)
        if visible_idx.numel() == 0:
            visible_idx = torch.arange(n, dtype=torch.long)

        dist_mat = shortest_path_lengths_hops(edge_index.clone(), n)
        for a in torch.nonzero(is_selected).squeeze(-1).tolist():
            vis = visible_idx.numpy()
            if vis.size == 0:
                continue
            dists = dist_mat[a, vis].astype(np.float32)
            valid_mask = dists <= n
            if not valid_mask.any():
                continue
            dists_valid = dists[valid_mask]
            k = min(K_ANCHORS, dists_valid.size)
            idx_sorted = np.argsort(dists_valid)[:k]
            labels_dists[a, :k] = torch.tensor(dists_valid[idx_sorted], dtype=torch.float)
            labels_dists_mask[a, :k] = True

        all_z.append(z_masked)
        all_ch.append(chir)
        all_fc.append(fc)
        all_labels_z.append(labels_z)
        all_labels_dists.append(labels_dists)
        all_labels_dists_mask.append(labels_dists_mask)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))

        if edge_index is not None and edge_index.numel() > 0:
            ei_offset = edge_index + node_offset
            edge_index_list_batched.append(ei_offset)
            edge_attr_matched = match_edge_attr_to_index(edge_index, edge_attr, target_dim=3)
            edge_attr_list_batched.append(edge_attr_matched)

        node_offset += n

    if len(all_z) == 0:
        return {
            "z": torch.tensor([], dtype=torch.long),
            "chirality": torch.tensor([], dtype=torch.float),
            "formal_charge": torch.tensor([], dtype=torch.float),
            "edge_index": torch.tensor([[], []], dtype=torch.long),
            "edge_attr": torch.tensor([], dtype=torch.float).reshape(0, 3),
            "batch": torch.tensor([], dtype=torch.long),
            "labels_z": torch.tensor([], dtype=torch.long),
            "labels_dists": torch.tensor([], dtype=torch.float).reshape(0, K_ANCHORS),
            "labels_dists_mask": torch.tensor([], dtype=torch.bool).reshape(0, K_ANCHORS),
        }

    z_batch = torch.cat(all_z, dim=0)
    chir_batch = torch.cat(all_ch, dim=0)
    fc_batch = torch.cat(all_fc, dim=0)
    labels_z_batch = torch.cat(all_labels_z, dim=0)
    labels_dists_batch = torch.cat(all_labels_dists, dim=0)
    labels_dists_mask_batch = torch.cat(all_labels_dists_mask, dim=0)
    batch_batch = torch.cat(batch_idx, dim=0)

    if len(edge_index_list_batched) > 0:
        edge_index_batched = torch.cat(edge_index_list_batched, dim=1)
        edge_attr_batched = torch.cat(edge_attr_list_batched, dim=0)
    else:
        edge_index_batched = torch.tensor([[], []], dtype=torch.long)
        edge_attr_batched = torch.tensor([], dtype=torch.float).reshape(0, 3)

    return {
        "z": z_batch,
        "chirality": chir_batch,
        "formal_charge": fc_batch,
        "edge_index": edge_index_batched,
        "edge_attr": edge_attr_batched,
        "batch": batch_batch,
        "labels_z": labels_z_batch,
        "labels_dists": labels_dists_batch,
        "labels_dists_mask": labels_dists_mask_batch,
    }


class GineBlock(nn.Module):
    """One GINEConv block (MLP + BN + ReLU)."""
    def __init__(self, node_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim))
        self.conv = GINEConv(self.mlp)
        self.bn = nn.BatchNorm1d(node_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = self.act(x)
        return x


class MaskedGINE(nn.Module):
    """
    Masked GNN objective:
      - predict masked atomic numbers (classification head)
      - predict hop-distance anchors for masked nodes (regression head)
      - optionally learned uncertainty weighting across the two losses
    """
    def __init__(self, node_emb_dim=NODE_EMB_DIM, edge_emb_dim=EDGE_EMB_DIM, num_layers=NUM_GNN_LAYERS,
                 max_atomic_z=MAX_ATOMIC_Z, class_weights=None):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim
        self.max_atomic_z = max_atomic_z

        num_embeddings = MASK_ATOM_ID + 1
        self.atom_emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=node_emb_dim, padding_idx=None)

        self.node_attr_proj = nn.Sequential(nn.Linear(2, node_emb_dim), nn.ReLU(), nn.Linear(node_emb_dim, node_emb_dim))
        self.edge_encoder = nn.Sequential(nn.Linear(3, edge_emb_dim), nn.ReLU(), nn.Linear(edge_emb_dim, edge_emb_dim))

        self._edge_to_node_proj = nn.Linear(edge_emb_dim, node_emb_dim) if edge_emb_dim != node_emb_dim else None

        self.gnn_layers = nn.ModuleList([GineBlock(node_emb_dim) for _ in range(num_layers)])

        self.atom_head = nn.Linear(node_emb_dim, MASK_ATOM_ID + 1)
        self.coord_head = nn.Linear(node_emb_dim, K_ANCHORS)

        if USE_LEARNED_WEIGHTING:
            self.log_var_z = nn.Parameter(torch.zeros(1))
            self.log_var_pos = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_z = None
            self.log_var_pos = None

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, z, chirality, formal_charge, edge_index, edge_attr, batch=None,
                labels_z=None, labels_dists=None, labels_dists_mask=None):
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        atom_embedding = self.atom_emb(z)
        node_attr = torch.stack([chirality, formal_charge], dim=1)
        node_attr_emb = self.node_attr_proj(node_attr.to(atom_embedding.device))
        x = atom_embedding + node_attr_emb

        if edge_attr is None or edge_attr.numel() == 0:
            edge_emb = torch.zeros((0, self.edge_emb_dim), dtype=torch.float, device=x.device)
        else:
            edge_emb = self.edge_encoder(edge_attr.to(x.device))

        edge_for_conv = self._edge_to_node_proj(edge_emb) if self._edge_to_node_proj is not None else edge_emb

        h = x
        for layer in self.gnn_layers:
            h = layer(h, edge_index.to(h.device), edge_for_conv)

        logits = self.atom_head(h)
        dists_pred = self.coord_head(h)

        if labels_z is not None and labels_dists is not None and labels_dists_mask is not None:
            mask = labels_z != -100
            if mask.sum() == 0:
                return torch.tensor(0.0, device=z.device)

            logits_masked = logits[mask]
            dists_pred_masked = dists_pred[mask]
            labels_z_masked = labels_z[mask]
            labels_dists_masked = labels_dists[mask]
            labels_dists_mask_mask = labels_dists_mask[mask]

            if self.class_weights is not None:
                loss_z = F.cross_entropy(
                    logits_masked, labels_z_masked.to(logits_masked.device), weight=self.class_weights.to(logits_masked.device)
                )
            else:
                loss_z = F.cross_entropy(logits_masked, labels_z_masked.to(logits_masked.device))

            if labels_dists_mask_mask.any():
                preds = dists_pred_masked[labels_dists_mask_mask]
                trues = labels_dists_masked[labels_dists_mask_mask].to(preds.device)
                loss_pos = F.mse_loss(preds, trues, reduction="mean")
            else:
                loss_pos = torch.tensor(0.0, device=z.device)

            if USE_LEARNED_WEIGHTING:
                lz = torch.exp(-self.log_var_z) * loss_z + self.log_var_z
                lp = torch.exp(-self.log_var_pos) * loss_pos + self.log_var_pos
                return 0.5 * (lz + lp)

            return loss_z + loss_pos

        return logits, dists_pred


class ValLossCallback(TrainerCallback):
    """Evaluation callback: prints metrics, saves best model, and early-stops on val loss."""
    def __init__(self, best_model_dir: str, val_loader: DataLoader, patience: int = 10, trainer_ref=None):
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.patience = patience
        self.best_epoch = None
        self.trainer_ref = trainer_ref
        self.best_model_dir = best_model_dir
        self.val_loader = val_loader

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num = int(state.epoch)
        train_loss = next((x["loss"] for x in reversed(state.log_history) if "loss" in x), None)
        print(f"\n=== Epoch {epoch_num}/{args.num_train_epochs} ===")
        if train_loss is not None:
            print(f"Train Loss: {train_loss:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch_num = int(state.epoch) + 1

        if self.trainer_ref is None:
            print(f"[Eval] Epoch {epoch_num} - metrics (trainer_ref missing): {metrics}")
            return

        metric_val_loss = metrics.get("eval_loss") if metrics is not None else None

        model_eval = self.trainer_ref.model
        model_eval.eval()
        device_local = next(model_eval.parameters()).device

        preds_z_all, true_z_all = [], []
        pred_dists_all, true_dists_all = [], []
        total_loss, n_batches = 0.0, 0

        logits_masked_list, labels_masked_list = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                z = batch["z"].to(device_local)
                chir = batch["chirality"].to(device_local)
                fc = batch["formal_charge"].to(device_local)
                edge_index = batch["edge_index"].to(device_local)
                edge_attr = batch["edge_attr"].to(device_local)
                batch_idx = batch["batch"].to(device_local)
                labels_z = batch["labels_z"].to(device_local)
                labels_dists = batch["labels_dists"].to(device_local)
                labels_dists_mask = batch["labels_dists_mask"].to(device_local)

                try:
                    loss = model_eval(z, chir, fc, edge_index, edge_attr, batch_idx, labels_z, labels_dists, labels_dists_mask)
                except Exception:
                    loss = None

                if isinstance(loss, torch.Tensor):
                    total_loss += loss.item()
                    n_batches += 1

                logits, dists_pred = model_eval(z, chir, fc, edge_index, edge_attr, batch_idx)
                mask = labels_z != -100
                if mask.sum().item() == 0:
                    continue

                logits_masked_list.append(logits[mask])
                labels_masked_list.append(labels_z[mask])

                pred_z = torch.argmax(logits[mask], dim=-1)
                true_z = labels_z[mask]

                pred_d = dists_pred[mask][labels_dists_mask[mask]]
                true_d = labels_dists[mask][labels_dists_mask[mask]]

                if pred_d.numel() > 0:
                    pred_dists_all.extend(pred_d.cpu().tolist())
                    true_dists_all.extend(true_d.cpu().tolist())

                preds_z_all.extend(pred_z.cpu().tolist())
                true_z_all.extend(true_z.cpu().tolist())

        avg_val_loss = metric_val_loss if metric_val_loss is not None else ((total_loss / n_batches) if n_batches > 0 else float("nan"))

        accuracy = accuracy_score(true_z_all, preds_z_all) if len(true_z_all) > 0 else 0.0
        f1 = f1_score(true_z_all, preds_z_all, average="weighted") if len(true_z_all) > 0 else 0.0
        rmse = np.sqrt(mean_squared_error(true_dists_all, pred_dists_all)) if len(true_dists_all) > 0 else 0.0
        mae = mean_absolute_error(true_dists_all, pred_dists_all) if len(true_dists_all) > 0 else 0.0

        if len(logits_masked_list) > 0:
            all_logits_masked = torch.cat(logits_masked_list, dim=0)
            all_labels_masked = torch.cat(labels_masked_list, dim=0)
            cw = getattr(model_eval, "class_weights", None)
            if cw is not None:
                try:
                    loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked, weight=cw.to(device_local))
                except Exception:
                    loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked)
            else:
                loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked)
            try:
                perplexity = float(torch.exp(loss_z_all).cpu().item())
            except Exception:
                perplexity = float(np.exp(float(loss_z_all.cpu().item())))
        else:
            perplexity = float("nan")

        print(f"\n--- Evaluation after Epoch {epoch_num} ---")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 (weighted): {f1:.4f}")
        print(f"Validation RMSE (distances): {rmse:.4f}")
        print(f"Validation MAE  (distances): {mae:.4f}")
        print(f"Validation Perplexity (classification head): {perplexity:.4f}")

        if avg_val_loss is not None and not (isinstance(avg_val_loss, float) and np.isnan(avg_val_loss)) and avg_val_loss < self.best_val_loss - 1e-6:
            self.best_val_loss = avg_val_loss
            self.best_epoch = int(state.epoch)
            self.epochs_no_improve = 0
            os.makedirs(self.best_model_dir, exist_ok=True)
            try:
                torch.save(self.trainer_ref.model.state_dict(), os.path.join(self.best_model_dir, "pytorch_model.bin"))
                print(f"Saved new best model (epoch {epoch_num}) to {os.path.join(self.best_model_dir, 'pytorch_model.bin')}")
            except Exception as e:
                print(f"Failed to save best model at epoch {epoch_num}: {e}")
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping after {self.patience} epochs with no improvement.")
            control.should_training_stop = True


def build_datasets_and_loaders(parsed, batch_train: int = 16, batch_val: int = 8, num_workers: int = 4):
    """Split indices into train/val and construct Dataset/DataLoader."""
    (node_atomic_lists, node_chirality_lists, node_charge_lists, edge_index_lists, edge_attr_lists, num_nodes_list) = parsed

    indices = list(range(len(node_atomic_lists)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    def subset(l, idxs):
        return [l[i] for i in idxs]

    train_atomic = subset(node_atomic_lists, train_idx)
    train_chirality = subset(node_chirality_lists, train_idx)
    train_charge = subset(node_charge_lists, train_idx)
    train_edge_index = subset(edge_index_lists, train_idx)
    train_edge_attr = subset(edge_attr_lists, train_idx)
    train_num_nodes = subset(num_nodes_list, train_idx)

    val_atomic = subset(node_atomic_lists, val_idx)
    val_chirality = subset(node_chirality_lists, val_idx)
    val_charge = subset(node_charge_lists, val_idx)
    val_edge_index = subset(edge_index_lists, val_idx)
    val_edge_attr = subset(edge_attr_lists, val_idx)
    val_num_nodes = subset(num_nodes_list, val_idx)

    train_dataset = PolymerDataset(train_atomic, train_chirality, train_charge, train_edge_index, train_edge_attr, train_num_nodes)
    val_dataset = PolymerDataset(val_atomic, val_chirality, val_charge, val_edge_index, val_edge_attr, val_num_nodes)

    train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True, collate_fn=collate_batch, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_val, shuffle=False, collate_fn=collate_batch, num_workers=num_workers)
    return train_dataset, val_dataset, train_loader, val_loader, train_atomic


def train_and_evaluate(args: argparse.Namespace) -> None:
    """Main run: parse data, build model, train, reload best, final eval printout."""
    output_dir = args.output_dir
    best_model_dir = os.path.join(output_dir, "best")
    os.makedirs(output_dir, exist_ok=True)

    parsed = parse_graphs_from_csv(args.csv_path, args.target_rows, args.chunksize)
    train_dataset, val_dataset, train_loader, val_loader, train_atomic = build_datasets_and_loaders(
        parsed, batch_train=16, batch_val=8, num_workers=args.num_workers
    )

    class_weights = compute_class_weights(train_atomic)

    model = MaskedGINE(
        node_emb_dim=NODE_EMB_DIM,
        edge_emb_dim=EDGE_EMB_DIM,
        num_layers=NUM_GNN_LAYERS,
        max_atomic_z=MAX_ATOMIC_Z,
        class_weights=class_weights,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=25,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        disable_tqdm=False,
        logging_first_step=True,
        report_to=[],
        dataloader_num_workers=args.num_workers,
    )

    callback = ValLossCallback(best_model_dir=best_model_dir, val_loader=val_loader, patience=10)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_batch,
        callbacks=[callback],
    )
    callback.trainer_ref = trainer

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    best_model_path = os.path.join(best_model_dir, "pytorch_model.bin")
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"\nLoaded best model from {best_model_path}")
        except Exception as e:
            print(f"\nFailed to load best model from {best_model_path}: {e}")

    # Final evaluation
    model.eval()
    preds_z_all, true_z_all = [], []
    pred_dists_all, true_dists_all = [], []
    logits_masked_list_final, labels_masked_list_final = [], []

    with torch.no_grad():
        for batch in val_loader:
            z = batch["z"].to(device)
            chir = batch["chirality"].to(device)
            fc = batch["formal_charge"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_idx = batch["batch"].to(device)
            labels_z = batch["labels_z"].to(device)
            labels_dists = batch["labels_dists"].to(device)
            labels_dists_mask = batch["labels_dists_mask"].to(device)

            logits, dists_pred = model(z, chir, fc, edge_index, edge_attr, batch_idx)

            mask = labels_z != -100
            if mask.sum().item() == 0:
                continue

            logits_masked_list_final.append(logits[mask])
            labels_masked_list_final.append(labels_z[mask])

            pred_z = torch.argmax(logits[mask], dim=-1)
            true_z = labels_z[mask]

            pred_d = dists_pred[mask][labels_dists_mask[mask]]
            true_d = labels_dists[mask][labels_dists_mask[mask]]

            if pred_d.numel() > 0:
                pred_dists_all.extend(pred_d.cpu().tolist())
                true_dists_all.extend(true_d.cpu().tolist())

            preds_z_all.extend(pred_z.cpu().tolist())
            true_z_all.extend(true_z.cpu().tolist())

    accuracy = accuracy_score(true_z_all, preds_z_all) if len(true_z_all) > 0 else 0.0
    f1 = f1_score(true_z_all, preds_z_all, average="weighted") if len(true_z_all) > 0 else 0.0
    rmse = np.sqrt(mean_squared_error(true_dists_all, pred_dists_all)) if len(true_dists_all) > 0 else 0.0
    mae = mean_absolute_error(true_dists_all, pred_dists_all) if len(true_dists_all) > 0 else 0.0

    if len(logits_masked_list_final) > 0:
        all_logits_masked_final = torch.cat(logits_masked_list_final, dim=0)
        all_labels_masked_final = torch.cat(labels_masked_list_final, dim=0)
        cw_final = getattr(model, "class_weights", None)
        if cw_final is not None:
            try:
                loss_z_final = F.cross_entropy(all_logits_masked_final, all_labels_masked_final, weight=cw_final.to(device))
            except Exception:
                loss_z_final = F.cross_entropy(all_logits_masked_final, all_labels_masked_final)
        else:
            loss_z_final = F.cross_entropy(all_logits_masked_final, all_labels_masked_final)
        try:
            perplexity_final = float(torch.exp(loss_z_final).cpu().item())
        except Exception:
            perplexity_final = float(np.exp(float(loss_z_final.cpu().item())))
    else:
        perplexity_final = float("nan")

    best_val_loss = callback.best_val_loss if hasattr(callback, "best_val_loss") else float("nan")
    best_epoch_num = (int(callback.best_epoch) + 1) if callback.best_epoch is not None else None

    print(f"\n=== Final Results (evaluated on best saved model) ===")
    print(f"Total Training Time (s): {total_time:.2f}")
    print(f"Best Epoch (1-based): {best_epoch_num}" if best_epoch_num is not None else "Best Epoch: (none saved)")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 (weighted): {f1:.4f}")
    print(f"Validation RMSE (distances): {rmse:.4f}")
    print(f"Validation MAE  (distances): {mae:.4f}")
    print(f"Validation Perplexity (classification head): {perplexity_final:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-trainable Parameters: {non_trainable_params}")


def main():
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
