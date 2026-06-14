"""
Inference: load trained BiGRU model and predict rumor label for input text(s).

This module is designed to be used by the explainability component later.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    MODEL_SAVE_PATH, VOCAB_SAVE_PATH,
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, DEVICE,
)
from models.bigru import BiGRU
from utils.data_utils import encode, load_vocab


class RumorClassifier:
    """
    Wrapper for the trained BiGRU rumor detection model.

    Usage:
        clf = RumorClassifier()
        pred = clf.classify("This is a tweet text.")
        print(pred)  # 0 or 1
    """

    def __init__(self, model_path: str = None, vocab_path: str = None):
        model_path = model_path or MODEL_SAVE_PATH
        vocab_path = vocab_path or VOCAB_SAVE_PATH

        self.vocab = load_vocab(vocab_path)
        self.model = BiGRU(
            len(self.vocab), EMBEDDING_DIM, HIDDEN_DIM,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )
        self.model.to(DEVICE)
        self.model.eval()

    def classify(self, text: str) -> int:
        """
        Classify a single text as rumor (1) or non-rumor (0).

        Args:
            text: input text string

        Returns:
            int: 0 (non-rumor) or 1 (rumor)
        """
        token_ids = encode(text, self.vocab)
        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            pred = (torch.sigmoid(logits) > 0.5).long().item()
        return pred

    def classify_batch(self, texts: list) -> list:
        """
        Classify a batch of texts.

        Args:
            texts: list of input text strings

        Returns:
            list: predicted labels (0 or 1)
        """
        results = [self.classify(t) for t in texts]
        return results


# ================ CLI demo ================
if __name__ == "__main__":
    import sys

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}")
        print("Please run 'python run.py' first to train the model.")
        sys.exit(1)

    clf = RumorClassifier()

    # Interactive demo
    print("=" * 50)
    print("Rumor Detection Demo (BiGRU)")
    print("Enter text to classify, or 'q' to quit.")
    print("=" * 50)

    while True:
        try:
            text = input("\nText: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() == "q":
            break
        if not text:
            continue

        pred = clf.classify(text)
        label = "Rumor" if pred == 1 else "Non-rumor"
        print(f"Prediction: {label} ({pred})")
