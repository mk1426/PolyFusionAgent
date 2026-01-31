"""
Fingerprint masked language modeling (MLM) using a Transformer encoder.
"""

import os
import json
import time
import sys
import csv
import argparse
from typing import List

# Increase max CSV field size limit (fingerprints can be long)
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
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# Configuration / Constants
# ---------------------------
P_MASK = 0.15

FINGERPRINT_KEY = "morgan_r3_bits"
FP_LENGTH = 2048

MASK_TOKEN_ID = 2
VOCAB_SIZE = 3

HIDDEN_DIM = 256
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_NHEAD = 8
TRANSFORMER_FF = 1024
DROPOUT = 0.1

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fingerprint MLM pretraining (Transformer).")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/path/to/polymer_structures_unified_processed.csv",
        help="Processed CSV containing a JSON 'fingerprints' column.",
    )
    parser.add_argument("--target_rows", type=int, default=5_000_000, help="Max rows to parse.")
    parser.add_argument("--chunksize", type=int, default=50_000, help="CSV chunksize.")
    parser.add_argument("--output_dir", type=str, default="/path/to/fingerprint_mlm_output_5M", help="Training output directory.")
    parser.add_argument("--num_workers", type=int, default=0, help="PyTorch DataLoader num workers (kept default 0).")
    return parser.parse_args()


def load_fingerprints(csv_path: str, target_rows: int, chunksize: int) -> List[List[int]]:
    """Stream CSV and parse fingerprint bits into fixed-length vectors of ints."""
    fp_lists: List[List[int]] = []
    rows_read = 0

    for chunk in pd.read_csv(csv_path, engine="python", chunksize=chunksize):
        fps_chunk = chunk["fingerprints"]
        for fpval in fps_chunk:
            if pd.isna(fpval):
                fp_lists.append([0] * FP_LENGTH)
                continue

            if isinstance(fpval, str):
                try:
                    fp_json = json.loads(fpval)
                except Exception:
                    try:
                        fp_json = json.loads(fpval.replace("'", '"'))
                    except Exception:
                        parts = [p.strip().strip('"').strip("'") for p in fpval.split(",")]
                        bits = [1 if p in ("1", "True", "true") else 0 for p in parts[:FP_LENGTH]]
                        if len(bits) < FP_LENGTH:
                            bits += [0] * (FP_LENGTH - len(bits))
                        fp_lists.append(bits)
                        continue
            elif isinstance(fpval, dict):
                fp_json = fpval
            else:
                fp_lists.append([0] * FP_LENGTH)
                continue

            bits = fp_json.get(FINGERPRINT_KEY, None)
            if bits is None:
                if isinstance(fp_json, list):
                    bits = fp_json
                else:
                    bits = [0] * FP_LENGTH

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

            fp_lists.append(normalized[:FP_LENGTH])

        rows_read += len(chunk)
        if rows_read >= target_rows:
            break

    print(f"Loaded {len(fp_lists)} fingerprint vectors (using FP_LENGTH={FP_LENGTH}).")
    return fp_lists


class FingerprintDataset(Dataset):
    """Dataset of fixed-length fingerprint bit vectors (stored as torch.long tensors)."""
    def __init__(self, fps: List[torch.Tensor]):
        self.fps = fps

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        return self.fps[idx]


def collate_batch(batch):
    """
    MLM-style collation:
      - Select positions with P_MASK
      - Labels are true bits only on selected positions, else -100
      - Inputs are corrupted with 80/10/10 mask/random/keep policy
    """
    B = len(batch)
    if B == 0:
        return {
            "z": torch.zeros((0, FP_LENGTH), dtype=torch.long),
            "labels_z": torch.zeros((0, FP_LENGTH), dtype=torch.long),
            "attention_mask": torch.zeros((0, FP_LENGTH), dtype=torch.bool),
        }

    tensors = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, dict):
            if "fp" in item:
                val = item["fp"]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.long)
                tensors.append(val)
            else:
                found = None
                for v in item.values():
                    if isinstance(v, torch.Tensor):
                        found = v
                        break
                    elif isinstance(v, np.ndarray):
                        found = torch.tensor(v, dtype=torch.long)
                        break
                    elif isinstance(v, list):
                        try:
                            found = torch.tensor(v, dtype=torch.long)
                            break
                        except Exception:
                            continue
                if found is None:
                    raise KeyError(f"collate_batch: couldn't find tensor-like fp in item keys: {list(item.keys())}")
                tensors.append(found)
        else:
            tensors.append(torch.tensor(item, dtype=torch.long))

    all_inputs = torch.stack(tensors, dim=0).long()
    labels_z = torch.full_like(all_inputs, fill_value=-100, dtype=torch.long)
    z_masked = all_inputs.clone()

    for i in range(B):
        z = all_inputs[i]
        n_positions = z.size(0)
        is_selected = torch.rand(n_positions) < P_MASK
        if is_selected.all():
            is_selected[torch.randint(0, n_positions, (1,))] = False

        sel_idx = torch.nonzero(is_selected).squeeze(-1)
        if sel_idx.numel() > 0:
            labels_z[i, sel_idx] = z[sel_idx]

            probs = torch.rand(sel_idx.size(0))
            mask_choice = probs < 0.8
            rand_choice = (probs >= 0.8) & (probs < 0.9)

            if mask_choice.any():
                z_masked[i, sel_idx[mask_choice]] = MASK_TOKEN_ID
            if rand_choice.any():
                rand_bits = torch.randint(0, 2, (rand_choice.sum().item(),), dtype=torch.long)
                z_masked[i, sel_idx[rand_choice]] = rand_bits

    attention_mask = torch.ones_like(all_inputs, dtype=torch.bool)
    return {"z": z_masked, "labels_z": labels_z, "attention_mask": attention_mask}


class FingerprintEncoder(nn.Module):
    """Transformer encoder over a length-FP_LENGTH token sequence with small vocab {0,1,MASK}."""
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=FP_LENGTH,
                 num_layers=TRANSFORMER_NUM_LAYERS, nhead=TRANSFORMER_NHEAD, dim_feedforward=TRANSFORMER_FF,
                 dropout=DROPOUT):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)

        key_padding_mask = (~attention_mask) if attention_mask is not None else None
        return self.transformer(x, src_key_padding_mask=key_padding_mask)


class MaskedFingerprintModel(nn.Module):
    """Encoder + token classification head; returns scalar loss when labels_z provided."""
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.encoder = FingerprintEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, attention_mask=None, labels_z=None):
        embeddings = self.encoder(z, attention_mask=attention_mask)
        logits = self.mlm_head(embeddings)

        if labels_z is not None:
            mask = labels_z != -100
            if mask.sum() == 0:
                return torch.tensor(0.0, device=z.device)

            logits_masked = logits[mask]
            labels_masked = labels_z[mask].long()
            return F.cross_entropy(logits_masked, labels_masked)

        return logits


class ValLossCallback(TrainerCallback):
    """Tracks best eval loss, prints metrics, saves best model, early-stops."""
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

        preds_bits, true_bits = [], []
        total_loss, n_batches = 0.0, 0
        logits_masked_list, labels_masked_list = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                z = batch["z"].to(device_local)
                labels_z = batch["labels_z"].to(device_local)
                attention_mask = batch.get("attention_mask", torch.ones_like(z, dtype=torch.bool)).to(device_local)

                try:
                    loss = model_eval(z, attention_mask=attention_mask, labels_z=labels_z)
                except Exception:
                    loss = None

                if isinstance(loss, torch.Tensor):
                    total_loss += loss.item()
                    n_batches += 1

                logits = model_eval(z, attention_mask=attention_mask)

                mask = labels_z != -100
                if mask.sum().item() == 0:
                    continue

                logits_masked_list.append(logits[mask])
                labels_masked_list.append(labels_z[mask])

                pred_bits = torch.argmax(logits[mask], dim=-1)
                true_b = labels_z[mask]

                preds_bits.extend(pred_bits.cpu().tolist())
                true_bits.extend(true_b.cpu().tolist())

        avg_val_loss = metric_val_loss if metric_val_loss is not None else ((total_loss / n_batches) if n_batches > 0 else float("nan"))
        accuracy = accuracy_score(true_bits, preds_bits) if len(true_bits) > 0 else 0.0
        f1 = f1_score(true_bits, preds_bits, average="weighted") if len(true_bits) > 0 else 0.0

        if len(logits_masked_list) > 0:
            all_logits_masked = torch.cat(logits_masked_list, dim=0)
            all_labels_masked = torch.cat(labels_masked_list, dim=0)
            loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked.long())
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

    fp_lists = load_fingerprints(args.csv_path, args.target_rows, args.chunksize)

    train_idx, val_idx = train_test_split(list(range(len(fp_lists))), test_size=0.2, random_state=42)
    train_fps = [torch.tensor(fp_lists[i], dtype=torch.long) for i in train_idx]
    val_fps = [torch.tensor(fp_lists[i], dtype=torch.long) for i in val_idx]

    # Compute class weights
    counts = np.ones((2,), dtype=np.float64)
    for fp in train_fps:
        vals = fp.cpu().numpy().astype(int)
        counts[0] += np.sum(vals == 0)
        counts[1] += np.sum(vals == 1)
    freq = counts / counts.sum()
    inv_freq = 1.0 / (freq + 1e-12)
    class_weights_arr = inv_freq / inv_freq.mean()
    class_weights = torch.tensor(class_weights_arr, dtype=torch.float)
    print("Class weights (for bit 0 and bit 1):", class_weights.numpy())

    train_dataset = FingerprintDataset(train_fps)
    val_dataset = FingerprintDataset(val_fps)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_batch, drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_batch, drop_last=False, num_workers=args.num_workers)

    model = MaskedFingerprintModel(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        eval_accumulation_steps=1000,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="epoch",
        logging_steps=500,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
    preds_bits_all, true_bits_all = [], []
    logits_masked_final, labels_masked_final = [], []

    with torch.no_grad():
        for batch in val_loader:
            z = batch["z"].to(device)
            labels_z = batch["labels_z"].to(device)
            attention_mask = batch.get("attention_mask", torch.ones_like(z, dtype=torch.bool)).to(device)

            logits = model(z, attention_mask=attention_mask)

            mask = labels_z != -100
            if mask.sum().item() == 0:
                continue

            logits_masked_final.append(logits[mask])
            labels_masked_final.append(labels_z[mask])

            pred_bits = torch.argmax(logits[mask], dim=-1)
            true_b = labels_z[mask]

            preds_bits_all.extend(pred_bits.cpu().tolist())
            true_bits_all.extend(true_b.cpu().tolist())

    accuracy = accuracy_score(true_bits_all, preds_bits_all) if len(true_bits_all) > 0 else 0.0
    f1 = f1_score(true_bits_all, preds_bits_all, average="weighted") if len(true_bits_all) > 0 else 0.0

    if len(logits_masked_final) > 0:
        all_logits_masked_final = torch.cat(logits_masked_final, dim=0)
        all_labels_masked_final = torch.cat(labels_masked_final, dim=0)
        loss_z_final = F.cross_entropy(all_logits_masked_final, all_labels_masked_final.long())
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
