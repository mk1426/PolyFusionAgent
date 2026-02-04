"""
SchNet.py
SchNet-based masked pretraining on polymer conformer geometry.
"""

from __future__ import annotations

import os
import json
import time
import sys
import csv
import argparse
from typing import List, Optional

# Increase max CSV field size limit
csv.field_size_limit(sys.maxsize)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import SchNet as BaseSchNet
from torch_geometric.nn import radius_graph

from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

# ---------------------------
# Configuration / Constants
# ---------------------------
P_MASK = 0.15
MAX_ATOMIC_Z = 85
MASK_ATOM_ID = MAX_ATOMIC_Z + 1

COORD_NOISE_SIGMA = 0.5
USE_LEARNED_WEIGHTING = True

SCHNET_NUM_GAUSSIANS = 50
SCHNET_NUM_INTERACTIONS = 6
SCHNET_CUTOFF = 10.0
SCHNET_MAX_NEIGHBORS = 64

K_ANCHORS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SchNet masked pretraining (geometry).")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/path/to/polymer_structures_unified_processed.csv",
        help="Processed CSV containing a JSON 'geometry' column.",
    )
    parser.add_argument("--target_rows", type=int, default=5_000_000, help="Max rows to read.")
    parser.add_argument("--chunksize", type=int, default=50_000, help="CSV chunksize.")
    parser.add_argument("--output_dir", type=str, default="/path/to/schnet_output_5M", help="Training output directory.")
    parser.add_argument("--num_workers", type=int, default=4, help="PyTorch DataLoader num workers.")
    return parser.parse_args()


def load_geometry_from_csv(csv_path: str, target_rows: int, chunksize: int):
    """
    Stream the processed CSV and extract:
      - atomic_numbers
      - coordinates
    from geometry['best_conformer'] for each row.
    """
    atomic_lists = []
    coord_lists = []
    rows_read = 0

    for chunk in pd.read_csv(csv_path, engine="python", chunksize=chunksize):
        geoms_chunk = chunk["geometry"].apply(json.loads)
        for geom in geoms_chunk:
            conf = geom["best_conformer"]
            atomic_lists.append(conf["atomic_numbers"])
            coord_lists.append(conf["coordinates"])

        rows_read += len(chunk)
        if rows_read >= target_rows:
            break

    print(f"Using manual max atomic number: {MAX_ATOMIC_Z} (MASK_ATOM_ID={MASK_ATOM_ID})")
    return atomic_lists, coord_lists


def compute_class_weights(train_z: List[torch.Tensor]) -> torch.Tensor:
    """Inverse-frequency class weights for atomic number classification."""
    num_classes = MASK_ATOM_ID + 1
    counts = np.ones((num_classes,), dtype=np.float64)

    for z in train_z:
        if z.numel() > 0:
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
    """Pairs of (z, pos) per polymer conformer."""

    def __init__(self, zs: List[torch.Tensor], pos_list: List[torch.Tensor]):
        self.zs = zs
        self.pos_list = pos_list

    def __len__(self):
        return len(self.zs)

    def __getitem__(self, idx):
        return {"z": self.zs[idx], "pos": self.pos_list[idx]}


def collate_batch(batch):
    """
    Collate conformers into a concatenated node set with a 'batch' vector, while applying:
      - atomic number masking (MLM-style)
      - coordinate corruption for masked atoms
      - invariant distance targets to nearest visible anchors (K_ANCHORS)
    """
    all_z, all_pos = [], []
    all_labels_z, all_labels_dists, all_labels_dists_mask = [], [], []
    batch_idx = []

    for i, data in enumerate(batch):
        z = data["z"]
        pos = data["pos"]
        n_atoms = z.size(0)
        if n_atoms == 0:
            continue

        is_selected = torch.rand(n_atoms) < P_MASK
        if is_selected.all():
            is_selected[torch.randint(0, n_atoms, (1,))] = False

        labels_z = torch.full((n_atoms,), -100, dtype=torch.long)
        labels_dists = torch.zeros((n_atoms, K_ANCHORS), dtype=torch.float)
        labels_dists_mask = torch.zeros((n_atoms, K_ANCHORS), dtype=torch.bool)
        labels_z[is_selected] = z[is_selected]

        # Atomic number corruption
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

        # Coordinate corruption (noise/random position)
        pos_masked = pos.clone()
        if is_selected.any():
            sel_idx = torch.nonzero(is_selected).squeeze(-1)
            probs_c = torch.rand(sel_idx.size(0))
            noisy_choice = probs_c < 0.8
            randpos_choice = (probs_c >= 0.8) & (probs_c < 0.9)

            if noisy_choice.any():
                idx = sel_idx[noisy_choice]
                noise = torch.randn((idx.size(0), 3)) * COORD_NOISE_SIGMA
                pos_masked[idx] = pos_masked[idx] + noise

            if randpos_choice.any():
                idx = sel_idx[randpos_choice]
                mins = pos.min(dim=0).values
                maxs = pos.max(dim=0).values
                randpos = (torch.rand((idx.size(0), 3)) * (maxs - mins)) + mins
                pos_masked[idx] = randpos

        # Anchor-distance targets
        visible_idx = torch.nonzero(~is_selected).squeeze(-1)
        if visible_idx.numel() == 0:
            visible_idx = torch.arange(n_atoms, dtype=torch.long)

        visible_pos = pos[visible_idx]
        for a in torch.nonzero(is_selected).squeeze(-1).tolist():
            dists = torch.sqrt(((pos[a].unsqueeze(0) - visible_pos) ** 2).sum(dim=1) + 1e-12)
            if dists.numel() > 0:
                k = min(K_ANCHORS, dists.numel())
                nearest_vals, _ = torch.topk(dists, k, largest=False)
                labels_dists[a, :k] = nearest_vals
                labels_dists_mask[a, :k] = True

        all_z.append(z_masked)
        all_pos.append(pos_masked)
        all_labels_z.append(labels_z)
        all_labels_dists.append(labels_dists)
        all_labels_dists_mask.append(labels_dists_mask)
        batch_idx.append(torch.full((n_atoms,), i, dtype=torch.long))

    if len(all_z) == 0:
        return {
            "z": torch.tensor([], dtype=torch.long),
            "pos": torch.tensor([], dtype=torch.float).reshape(0, 3),
            "batch": torch.tensor([], dtype=torch.long),
            "labels_z": torch.tensor([], dtype=torch.long),
            "labels_dists": torch.tensor([], dtype=torch.float).reshape(0, K_ANCHORS),
            "labels_dists_mask": torch.tensor([], dtype=torch.bool).reshape(0, K_ANCHORS),
        }

    return {
        "z": torch.cat(all_z, dim=0),
        "pos": torch.cat(all_pos, dim=0),
        "batch": torch.cat(batch_idx, dim=0),
        "labels_z": torch.cat(all_labels_z, dim=0),
        "labels_dists": torch.cat(all_labels_dists, dim=0),
        "labels_dists_mask": torch.cat(all_labels_dists_mask, dim=0),
    }


class NodeSchNet(nn.Module):
    """SchNet variant that returns node embeddings (no readout)."""

    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        max_num_neighbors=32,
        readout="add",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.base_schnet = BaseSchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
        )

    def forward(self, z, pos, batch=None):
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        h = self.base_schnet.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.base_schnet.distance_expansion(edge_weight)

        for interaction in self.base_schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h


# =============================================================================
# Wrapper used by MaskedSchNet
# =============================================================================

class NodeSchNetWrapper(nn.Module):
    """
    - Produces pooled embedding (mean pooling + pool_proj)
    - Provides node_logits(...) for reconstruction
    """

    def __init__(
        self,
        hidden_channels=600,
        num_interactions=SCHNET_NUM_INTERACTIONS,
        num_gaussians=SCHNET_NUM_GAUSSIANS,
        cutoff=SCHNET_CUTOFF,
        max_num_neighbors=SCHNET_MAX_NEIGHBORS,
        emb_dim: int = 600,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.schnet = NodeSchNet(
            hidden_channels=hidden_channels,
            num_filters=hidden_channels,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )

        self.atom_head = nn.Linear(hidden_channels, MASK_ATOM_ID + 1)
        self.pool_proj = nn.Linear(hidden_channels, emb_dim)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def encode_nodes(self, z, pos, batch=None):
        return self.schnet(z=z, pos=pos, batch=batch)

    def node_logits(self, z, pos, batch=None):
        h = self.encode_nodes(z, pos, batch=batch)
        return self.atom_head(h)

    def forward(self, z, pos, batch=None):
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        h = self.encode_nodes(z, pos, batch=batch)
        if h.size(0) == 0:
            B = int(batch.max().item() + 1) if batch.numel() > 0 else 0
            return torch.zeros((B, self.pool_proj.out_features), device=z.device)

        B = int(batch.max().item() + 1) if batch.numel() > 0 else 1
        pooled = torch.zeros((B, h.size(1)), device=h.device)
        counts = torch.zeros((B,), device=h.device)

        pooled.index_add_(0, batch, h)
        counts.index_add_(0, batch, torch.ones((h.size(0),), device=h.device))
        pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
        return self.pool_proj(pooled)


class MaskedSchNet(nn.Module):
    """Masked objectives on top of node embeddings from SchNet."""

    def __init__(
        self,
        hidden_channels=600,
        num_interactions=SCHNET_NUM_INTERACTIONS,
        num_gaussians=SCHNET_NUM_GAUSSIANS,
        cutoff=SCHNET_CUTOFF,
        max_atomic_z=MAX_ATOMIC_Z,
        max_num_neighbors=SCHNET_MAX_NEIGHBORS,
        class_weights=None,
    ):
        super().__init__()

        self.wrapper = NodeSchNetWrapper(
            hidden_channels=hidden_channels,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            emb_dim=600,
            class_weights=class_weights,
        )

        self.coord_head = nn.Linear(hidden_channels, K_ANCHORS)

        if USE_LEARNED_WEIGHTING:
            self.log_var_z = nn.Parameter(torch.zeros(1))
            self.log_var_pos = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_z = None
            self.log_var_pos = None

        self.class_weights = getattr(self.wrapper, "class_weights", None)

    def forward(self, z, pos, batch, labels_z=None, labels_dists=None, labels_dists_mask=None):
        h = self.wrapper.encode_nodes(z=z, pos=pos, batch=batch)
        logits = self.wrapper.atom_head(h)
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
                loss_z = F.cross_entropy(logits_masked, labels_z_masked, weight=self.class_weights)
            else:
                loss_z = F.cross_entropy(logits_masked, labels_z_masked)

            if labels_dists_mask_mask.any():
                preds = dists_pred_masked[labels_dists_mask_mask]
                trues = labels_dists_masked[labels_dists_mask_mask]
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
    """Evaluation callback: computes metrics on val_loader, saves best, early-stops on val loss."""

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
                pos = batch["pos"].to(device_local)
                batch_idx = batch["batch"].to(device_local)
                labels_z = batch["labels_z"].to(device_local)
                labels_dists = batch["labels_dists"].to(device_local)
                labels_dists_mask = batch["labels_dists_mask"].to(device_local)

                try:
                    loss = model_eval(z, pos, batch_idx, labels_z, labels_dists, labels_dists_mask)
                except Exception:
                    loss = None

                if isinstance(loss, torch.Tensor):
                    total_loss += loss.item()
                    n_batches += 1

                logits, dists_pred = model_eval(z, pos, batch_idx)

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


def train_and_eval(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    best_model_dir = os.path.join(output_dir, "best")
    os.makedirs(output_dir, exist_ok=True)

    atomic_lists, coord_lists = load_geometry_from_csv(args.csv_path, args.target_rows, args.chunksize)

    train_idx, val_idx = train_test_split(list(range(len(atomic_lists))), test_size=0.2, random_state=42)
    train_z = [torch.tensor(atomic_lists[i], dtype=torch.long) for i in train_idx]
    train_pos = [torch.tensor(coord_lists[i], dtype=torch.float) for i in train_idx]
    val_z = [torch.tensor(atomic_lists[i], dtype=torch.long) for i in val_idx]
    val_pos = [torch.tensor(coord_lists[i], dtype=torch.float) for i in val_idx]

    class_weights = compute_class_weights(train_z)

    train_dataset = PolymerDataset(train_z, train_pos)
    val_dataset = PolymerDataset(val_z, val_pos)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch, num_workers=args.num_workers)

    model = MaskedSchNet(
        hidden_channels=600,
        num_interactions=SCHNET_NUM_INTERACTIONS,
        num_gaussians=SCHNET_NUM_GAUSSIANS,
        cutoff=SCHNET_CUTOFF,
        max_atomic_z=MAX_ATOMIC_Z,
        max_num_neighbors=SCHNET_MAX_NEIGHBORS,
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
            pos = batch["pos"].to(device)
            batch_idx = batch["batch"].to(device)
            labels_z = batch["labels_z"].to(device)
            labels_dists = batch["labels_dists"].to(device)
            labels_dists_mask = batch["labels_dists_mask"].to(device)

            logits, dists_pred = model(z, pos, batch_idx)

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
    train_and_eval(args)


if __name__ == "__main__":
    main()
