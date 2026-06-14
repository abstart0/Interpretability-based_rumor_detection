"""
Training pipeline for TextCNN + GloVe rumor detection model.
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

from config.textcnn_config import (
    TRAIN_PATH, VAL_PATH, MODEL_SAVE_PATH, VOCAB_SAVE_PATH,
    GLOVE_DIR, GLOVE_URL, GLOVE_FILE, GLOVE_DIM, GLOVE_FREEZE,
    EMBEDDING_DIM, FILTER_SIZES, N_FILTERS, DROPOUT,
    MAX_LEN, BATCH_SIZE, EPOCHS, MIN_WORD_FREQ,
    LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, LR_FACTOR, EARLY_STOP_PATIENCE,
    DEVICE, SEED, DATALOADER_WORKERS,
)
from models.textcnn import TextCNN
from utils.data_utils import create_dataloaders, build_vocab, save_vocab
from utils.glove_utils import download_and_extract_glove, load_glove_and_matrix


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion=None):
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
    set_seed(SEED)

    # ===== Step 1: Download & extract GloVe =====
    print("=" * 55)
    print("Step 1: Preparing GloVe embeddings")
    print("=" * 55)
    glove_path = download_and_extract_glove(GLOVE_URL, GLOVE_DIR, GLOVE_FILE)

    # ===== Step 2: Load data & build vocab =====
    print(f"\n{'=' * 55}")
    print("Step 2: Loading data")
    print(f"{'=' * 55}")
    print(f"Train: {TRAIN_PATH}")
    print(f"Val:   {VAL_PATH}")

    import pandas as pd
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    vocab = build_vocab(train_df["text"], min_freq=MIN_WORD_FREQ)
    save_vocab(vocab, VOCAB_SAVE_PATH)
    print(f"Vocabulary size: {len(vocab)}")

    train_loader, val_loader, _ = create_dataloaders(
        TRAIN_PATH, VAL_PATH, BATCH_SIZE
    )

    # ===== Step 3: Build GloVe embedding matrix =====
    print(f"\n{'=' * 55}")
    print("Step 3: Building GloVe embedding matrix")
    print(f"{'=' * 55}")
    pretrained_matrix = load_glove_and_matrix(vocab, glove_path, GLOVE_DIM)

    # ===== Step 4: Create model =====
    print(f"\n{'=' * 55}")
    print("Step 4: Creating TextCNN model")
    print(f"{'=' * 55}")
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        filter_sizes=FILTER_SIZES,
        n_filters=N_FILTERS,
        dropout=DROPOUT,
        pretrained=pretrained_matrix,
    ).to(DEVICE)

    if GLOVE_FREEZE:
        model.embedding.weight.requires_grad = False
        print("  Embedding layer: FROZEN")
    else:
        print("  Embedding layer: fine-tunable")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 55}")
    print(f"Model: TextCNN + GloVe {GLOVE_DIM}d")
    print(f"  Filters: {FILTER_SIZES} x {N_FILTERS}, Dropout: {DROPOUT}")
    print(f"  Params: {total_params:,} ({trainable_params:,} trainable)")
    print(f"  Train: lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"{'=' * 55}\n")

    # ===== Step 5: Train =====
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion)
        current_val_acc = val_metrics["accuracy"]

        scheduler.step(current_val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        lr_msg = ""
        if current_lr < LEARNING_RATE * 0.5:
            lr_msg = f" [lr={current_lr:.2e}]"

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {current_val_acc:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
            f"{lr_msg}"
        )

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Best model saved (epoch {epoch}, acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping: no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    # ===== Step 6: Final evaluation =====
    print(f"\n{'=' * 55}")
    print(f"Training complete. Best epoch: {best_epoch}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))

    final_metrics = evaluate(model, val_loader)
    print(f"\n  Best Val Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Best Val Precision: {final_metrics['precision']:.4f}")
    print(f"  Best Val Recall:    {final_metrics['recall']:.4f}")
    print(f"  Best Val F1:        {final_metrics['f1']:.4f}")

    print(f"\n{'=' * 55}")
    print("Detailed Classification Report")
    print(f"{'=' * 55}")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = (torch.sigmoid(model(x)) > 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.long().cpu().tolist())
    print(classification_report(all_labels, all_preds,
        target_names=["Non-rumor", "Rumor"], digits=4))

    print(f"Model saved to: {MODEL_SAVE_PATH}")

    return model, vocab


if __name__ == "__main__":
    train()
