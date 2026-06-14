"""
Training and evaluation pipeline for the BiGRU rumor detection model.
"""

import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Allow running as script from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    TRAIN_PATH, VAL_PATH, CHECKPOINT_DIR, MODEL_SAVE_PATH, VOCAB_SAVE_PATH,
    EMBEDDING_DIM, HIDDEN_DIM, MAX_LEN, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, DEVICE, SEED, MIN_WORD_FREQ,
)
from models.bigru import BiGRU
from utils.data_utils import create_dataloaders, build_vocab, save_vocab, load_vocab


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
    """Main training procedure."""
    set_seed(SEED)

    print(f"Device: {DEVICE}")
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
    model = BiGRU(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nModel: BiGRU (emb={EMBEDDING_DIM}, hidden={HIDDEN_DIM})")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total_params:,}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")

    # ---------- Training loop ----------
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
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
        val_metrics = evaluate(model, val_loader, criterion)

        # Print progress
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  → Best model saved (acc={best_val_acc:.4f})")

    # ---------- Final evaluation ----------
    print(f"\n{'='*50}")
    print("Training complete. Loading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))

    final_metrics = evaluate(model, val_loader)
    print(f"Best Val Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Best Val Precision: {final_metrics['precision']:.4f}")
    print(f"Best Val Recall:    {final_metrics['recall']:.4f}")
    print(f"Best Val F1:        {final_metrics['f1']:.4f}")

    # Full classification report
    print("\nDetailed classification report:")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.long().cpu().tolist())
    print(classification_report(all_labels, all_preds, target_names=["Non-rumor", "Rumor"], digits=4))

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Vocab saved to: {VOCAB_SAVE_PATH}")

    return model, vocab


if __name__ == "__main__":
    train()
