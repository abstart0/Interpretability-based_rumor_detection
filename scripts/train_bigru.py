"""
Training and evaluation pipeline for the BiGRU rumor detection model.
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.bigru_config import (
    TRAIN_PATH, VAL_PATH, MODEL_SAVE_PATH, VOCAB_SAVE_PATH,
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, LR_FACTOR, EARLY_STOP_PATIENCE,
    DEVICE, SEED, DATALOADER_WORKERS,
)
from models.bigru import BiGRU
from utils.data_utils import create_dataloaders, save_vocab


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader, criterion=None):
    """
    Evaluate the model on a given dataloader.

    Returns:
        dict with keys: loss, accuracy, precision, recall, f1
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.long().cpu().tolist())

    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
    }
    if criterion is not None:
        metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def train():
    """Main training procedure with scheduler and early stopping."""
    set_seed(SEED)
    # CPU threading is handled via OMP_NUM_THREADS/MKL_NUM_THREADS env vars.
    # Do NOT call torch.set_num_threads() on Windows — it breaks OpenMP scheduling.
    print(f"Device: {DEVICE}, DataLoader workers: {DATALOADER_WORKERS}")

    print(f"Loading data from: {TRAIN_PATH}")

    # ---------- Load data ----------
    train_loader, val_loader, vocab = create_dataloaders(
        TRAIN_PATH, VAL_PATH, BATCH_SIZE
    )
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---------- Save vocab ----------
    save_vocab(vocab, VOCAB_SAVE_PATH)

    # ---------- Model, optimizer, loss ----------
    model = BiGRU(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    # ---------- Print model info ----------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}")
    print(f"Model: BiGRU")
    print(f"  Embedding dim: {EMBEDDING_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"  Layers: {NUM_LAYERS}, Dropout: {DROPOUT}")
    print(f"  Params: {total_params:,} ({trainable_params:,} trainable)")
    print(f"  Train config: lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"{'='*55}\n")

    # ---------- Training loop ----------
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- Training phase ----
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Validation phase ----
        val_metrics = evaluate(model, val_loader, criterion)
        current_val_acc = val_metrics["accuracy"]

        # ---- LR scheduler step (based on val accuracy) ----
        scheduler.step(current_val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Print progress ----
        lr_indicator = ""
        if current_lr < LEARNING_RATE * 0.5:
            lr_indicator = f" [lr reduced to {current_lr:.2e}]"
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {current_val_acc:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
            f"{lr_indicator}"
        )

        # ---- Save best model ----
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Best model saved (epoch {epoch}, acc={best_val_acc:.4f}, f1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1

        # ---- Early stopping ----
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping: no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    # ---------- Final evaluation ----------
    print(f"\n{'='*55}")
    print(f"Training complete. Best epoch: {best_epoch}")
    print(f"Loading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))

    final_metrics = evaluate(model, val_loader)
    print(f"\n  Best Val Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Best Val Precision: {final_metrics['precision']:.4f}")
    print(f"  Best Val Recall:    {final_metrics['recall']:.4f}")
    print(f"  Best Val F1:        {final_metrics['f1']:.4f}")

    # Full classification report
    print(f"\n{'='*55}")
    print("Detailed Classification Report")
    print(f"{'='*55}")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.long().cpu().tolist())
    print(classification_report(all_labels, all_preds,
        target_names=["Non-rumor", "Rumor"], digits=4))

    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Vocab saved to: {VOCAB_SAVE_PATH}")

    return model, vocab


if __name__ == "__main__":
    train()
