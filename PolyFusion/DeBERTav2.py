
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import random
import json
import torch
import pandas as pd
import numpy as np
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, Trainer, TrainingArguments
from transformers import DebertaV2Tokenizer, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback
import shutil
import sentencepiece as spm

# === 1. Load Data ===
df = pd.read_csv("Polymer_Foundational_Model/Datasets/polymer_structures_unified.csv", nrows=5000000, engine='python')
psmiles_list = df["psmiles"].astype(str).tolist()

# === 2. Train/Val Split ===
train_psmiles, val_psmiles = train_test_split(psmiles_list, test_size=0.2, random_state=42)

# === 3. SentencePiece Tokenizer Training ===
# Write training text required by SentencePiece
train_txt = "generated_polymer_smiles_5M.txt"
with open(train_txt, "w", encoding="utf-8") as f:
    for s in train_psmiles:
        f.write(s.strip() + "\n")

# Special tokens and element tokens as provided
elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
small_elements = [i.lower() for i in elements]

special_tokens = [
    "<pad>",
    "<mask>",
    "[*]",
    "(", ")", "=", "@", "#",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "-", "+",
    "/", "\\",
    "%", "[", "]",
]
special_tokens += elements + small_elements

spm_model_prefix = "spm_5M"
if not os.path.isfile(spm_model_prefix + ".model"):
    spm.SentencePieceTrainer.train(
        input=train_txt,
        model_prefix=spm_model_prefix,
        vocab_size=265,
        input_sentence_size=5_000_000,
        character_coverage=1.0,
        user_defined_symbols=special_tokens
    )

# Note: this produces spm.model and spm.vocab in the working directory

# === 4. Load HuggingFace Tokenizer (expects tokenizer files in './') ===
# Use the SentencePiece model we produced as the vocab file for DebertaV2Tokenizer.
# (This keeps DebertaV2Tokenizer usage while explicitly referencing the spm model file.)
tokenizer = DebertaV2Tokenizer(vocab_file=spm_model_prefix + ".model", do_lower_case=False)

# Ensure special tokens are set (if they already exist this will be a no-op)
tokenizer.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})

# === 5. Create HF datasets and tokenize (batched) ===
hf_train = Dataset.from_dict({"text": train_psmiles})
hf_val = Dataset.from_dict({"text": val_psmiles})

def tokenize_batch(examples):
    # Tokenize text -> return input_ids and attention_mask
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Batched tokenization with provided params (kept num_proc and batch_size as originally used)
train_tok = hf_train.map(tokenize_batch, batched=True, batch_size=10_000, num_proc=10)
val_tok = hf_val.map(tokenize_batch, batched=True, batch_size=10_000, num_proc=10)

dataset_dict = DatasetDict({"train": train_tok, "test": val_tok})
dataset_dict.save_to_disk("dataset_tokenized_all")

# === 6. Load tokenized dataset for training and set format for PyTorch ===
dataset_all = DatasetDict.load_from_disk("dataset_tokenized_all")
dataset_train = dataset_all["train"]
dataset_test = dataset_all["test"]

# Keep only input_ids and attention_mask for Trainer; DataCollator will create labels
dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

# === 7. Data collator for MLM ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# === 8. Model Config and Model ===
# Use tokenizer length for vocab_size and pad token id from tokenizer
vocab_size = len(tokenizer)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

config = DebertaV2Config(
    vocab_size=vocab_size,
    hidden_size=600,
    num_attention_heads=12,
    num_hidden_layers=12,
    intermediate_size=512,
    pad_token_id=pad_token_id
)

model = DebertaV2ForMaskedLM(config)
# Resize token embeddings to match tokenizer (in case add_special_tokens added tokens)
model.resize_token_embeddings(len(tokenizer))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# === 9. Training Arguments ===
# NOTE: per your instruction, leaving `eval_strategy` as originally written (not changing to evaluation_strategy).
training_args = TrainingArguments(
    output_dir="./polybert_output_5M",
    overwrite_output_dir=True,
    num_train_epochs=25,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=1000,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",  # kept as in your original code (you asked to keep suggestion 4 unchanged)
    logging_strategy="steps",
    logging_steps=500,
    logging_first_step=True,
    save_strategy="no",
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to=[],
    disable_tqdm=False,
)

# === 10. Callback (printing/metrics logic to exactly match the first file) ===
class EpochMetricsCallback(TrainerCallback):
    def __init__(self, tokenizer_model_path):
        super().__init__()
        # Load SentencePiece processor properly
        self.sp = SentencePieceProcessor()
        self.sp.Load(tokenizer_model_path)
        self.tokenizer_model_path = tokenizer_model_path
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.patience = 10
        self.all_epochs = []
        self.best_val_f1 = None
        self.best_val_accuracy = None
        self.best_perplexity = None
        # trainer reference will be set after Trainer instantiation
        self.trainer_ref = None
        # temporary storage for train loss captured at epoch end
        self._last_train_loss = None

    def token_id_to_str(self, token_id):
        try:
            if token_id == self.sp.pad_id():
                return "[PAD]"
            if token_id == self.sp.unk_id():
                return "[UNK]"
        except Exception:
            pass
        return self.sp.id_to_piece(token_id)

    def save_model(self, trainer_obj, suffix):
        if trainer_obj is None:
            return
        model_dir = f"./polybert_output_5M/{suffix}"
        os.makedirs(model_dir, exist_ok=True)
        trainer_obj.model.save_pretrained(model_dir)
        try:
            shutil.copyfile(self.tokenizer_model_path, f"{model_dir}/tokenizer.model")
        except Exception:
            pass

    # Keep on_epoch_end minimal: only capture the most recent train loss
    def on_epoch_end(self, args, state, control, **kwargs):
        train_loss = None
        for log in reversed(state.log_history):
            if "loss" in log and float(log.get("loss", 0)) != 0.0:
                # pick the most recent loss entry
                train_loss = log["loss"]
                break
        self._last_train_loss = train_loss
        # DO NOT print eval values here — evaluation hasn't necessarily run yet.

    # Called AFTER Trainer runs evaluation; metrics is provided by Trainer
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_metrics = metrics or {}
        eval_loss = eval_metrics.get("eval_loss")
        eval_f1 = eval_metrics.get("eval_f1")
        eval_accuracy = eval_metrics.get("eval_accuracy", None)

        # retrieve most recent train loss captured at epoch end (could be None)
        train_loss = self._last_train_loss

        epoch_data = {
            "epoch": state.epoch,
            "train_loss": train_loss,
            "val_loss": eval_loss,
            "val_f1": eval_f1,
            "val_accuracy": eval_accuracy,
            "perplexity": np.exp(eval_loss) if eval_loss is not None else None,
        }
        self.all_epochs.append(epoch_data)

        # Save best model (use stored trainer_ref if available)
        if eval_loss is not None and eval_loss < self.best_val_loss - 1e-6:
            self.best_val_loss = eval_loss
            self.best_epoch = state.epoch
            self.epochs_no_improve = 0
            self.best_val_f1 = eval_f1
            self.best_val_accuracy = eval_accuracy
            self.best_perplexity = np.exp(eval_loss) if eval_loss is not None else None
            self.save_model(self.trainer_ref, "best")
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping: no improvement in val_loss for {self.patience} epochs.")
            control.should_training_stop = True

        total_params = sum(p.numel() for p in self.trainer_ref.model.parameters()) if self.trainer_ref is not None else sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in (self.trainer_ref.model.parameters() if self.trainer_ref is not None else model.parameters()) if p.requires_grad)
        print(f"\n=== Epoch {int(state.epoch)}/{args.num_train_epochs} ===")
        print(f"Train Loss: {train_loss:.4f}" if train_loss is not None else "Train Loss: None")
        print(f"Validation Loss: {eval_loss:.4f}" if eval_loss is not None else "Validation Loss: None")
        print(f"Validation F1: {eval_f1:.4f}" if eval_f1 is not None else "Validation F1: None")
        if eval_accuracy is not None:
            print(f"Validation Accuracy:{eval_accuracy:.4f}")
        if eval_loss is not None:
            print(f"Perplexity: {np.exp(eval_loss):.2f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f} (epoch {int(self.best_epoch)})")
        print(f"Total Params: {total_params}")
        print(f"Trainable Params: {trainable_params}")
        print(f"No improvement count:{self.epochs_no_improve}/{self.patience}")

    def on_train_end(self, args, state, control, **kwargs):
        print("\n=== Model saved ===")
        print(f"Best model (epoch {int(self.best_epoch)}, val_loss={self.best_val_loss:.4f}): ./polybert_output_5M/best/")

# === 11. Metrics function (fixed for MLM shapes and -100 masking) ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (batch, seq_len, vocab_size), labels: (batch, seq_len)
    # Flatten
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100
    if mask.sum() == 0:
        return {"eval_f1": 0.0, "eval_accuracy": 0.0}
    masked_logits = flat_logits[mask]
    masked_labels = flat_labels[mask]
    preds = np.argmax(masked_logits, axis=-1)
    f1 = f1_score(masked_labels, preds, average="weighted")
    accuracy = np.mean(masked_labels == preds)
    return {"eval_f1": f1, "eval_accuracy": accuracy}

# === 12. Trainer and training ===
start_time = time.time()
callback = EpochMetricsCallback(tokenizer_model_path=spm_model_prefix + ".model")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,   # fixed name
    eval_dataset=dataset_test,     # fixed name
    data_collator=data_collator,   # ensure MLM labels are created
    compute_metrics=compute_metrics,
    callbacks=[callback],
)

# Attach trainer reference to callback so it can save model
callback.trainer_ref = trainer

train_output = trainer.train()
total_time = time.time() - start_time

# === 9. Final Results Report ===
print(f"\n=== Final Results ===")
print(f"Total Training Time (s): {total_time:.2f}")
print(f"Best Validation Loss: {callback.best_val_loss:.4f}")
if callback.best_val_f1 is not None:
    print(f"Best Validation F1: {callback.best_val_f1:.4f}")
else:
    print("Best Validation F1: None")
if callback.best_val_accuracy is not None:
    print(f"Best Validation Accuracy: {callback.best_val_accuracy:.4f}")
else:
    print("Best Validation Accuracy: None")
if callback.best_perplexity is not None:
    print(f"Best Perplexity: {callback.best_perplexity:.2f}")
else:
    print("Best Perplexity: None")
print(f"Best Model Epoch: {int(callback.best_epoch)}")
print(f"Final Training Loss: {train_output.training_loss:.4f}")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Non-trainable Parameters: {non_trainable_params}")
