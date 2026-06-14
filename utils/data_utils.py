"""
Data loading, tokenization, and dataset utilities for rumor detection.
"""

import re
import os
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config.config import MAX_LEN


def tokenize(text: str) -> list:
    """Simple whitespace + word character tokenizer, lowercase."""
    return re.findall(r"\w+", text.lower())


def build_vocab(texts, min_freq=2):
    """
    Build vocabulary from a list of texts.

    Args:
        texts: iterable of strings
        min_freq: minimum token frequency to be included

    Returns:
        dict: token -> id mapping
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def encode(text: str, vocab: dict) -> list:
    """
    Encode a text into a list of token ids.
    Padding/truncation to MAX_LEN is applied.
    """
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids


def save_vocab(vocab: dict, path: str):
    """Save vocabulary with torch."""
    torch.save(vocab, path)


def load_vocab(path: str) -> dict:
    """Load vocabulary saved with torch."""
    return torch.load(path, weights_only=False)


class RumorDataset(Dataset):
    """PyTorch Dataset for rumor detection."""

    def __init__(self, df: pd.DataFrame, vocab: dict):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y


def create_dataloaders(train_path: str, val_path: str, batch_size: int):
    """
    Load data, build vocabulary, and create DataLoaders.

    Returns:
        train_loader, val_loader, vocab
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Build vocab from training data only
    vocab = build_vocab(train_df["text"], min_freq=2)

    train_dataset = RumorDataset(train_df, vocab)
    val_dataset = RumorDataset(val_df, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocab
