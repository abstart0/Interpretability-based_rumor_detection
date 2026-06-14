"""
Unified inference interface for rumor detection models.

This module provides a single entry point for the explainability component
to call the best trained model (TextCNN + GloVe by default).

Usage:
    from predictor import RumorDetector

    detector = RumorDetector()                # loads best model
    label, confidence = detector.predict("some tweet text")
    # label: 0 (non-rumor) or 1 (rumor)
    # confidence: float in [0, 1]
"""

import os
import sys
import re

import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.textcnn_config import (
    MODEL_SAVE_PATH, VOCAB_SAVE_PATH,
    EMBEDDING_DIM, FILTER_SIZES, N_FILTERS, DROPOUT, MAX_LEN,
)
from models.textcnn import TextCNN
from utils.data_utils import encode, load_vocab


class RumorDetector:
    """
    Unified rumor detector using the best trained model.

    Default: TextCNN + GloVe 200d (86.78% val accuracy).

    Usage:
        detector = RumorDetector()
        label, conf = detector.predict("Breaking news: ...")
        # label ∈ {0, 1}, conf ∈ [0.0, 1.0]
    """

    def __init__(self, model_path: str = None, vocab_path: str = None):
        model_path = model_path or MODEL_SAVE_PATH
        vocab_path = vocab_path or VOCAB_SAVE_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run: py -3.11 run_textcnn.py"
            )

        self.vocab = load_vocab(vocab_path)
        self.model = TextCNN(
            vocab_size=len(self.vocab),
            embedding_dim=EMBEDDING_DIM,
            filter_sizes=FILTER_SIZES,
            n_filters=N_FILTERS,
            dropout=DROPOUT,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def predict(self, text: str) -> tuple:
        """
        Classify a single text.

        Args:
            text: input text string (tweet, news headline, etc.)

        Returns:
            tuple[int, float]: (predicted_label, confidence)
                - label: 0 (non-rumor) or 1 (rumor)
                - confidence: sigmoid probability ∈ [0, 1]
        """
        token_ids = encode(text, self.vocab)
        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()

        label = 1 if prob > 0.5 else 0
        return label, prob

    def predict_batch(self, texts: list) -> list:
        """Classify a list of texts. Returns list of (label, confidence)."""
        return [self.predict(t) for t in texts]


# ================ CLI demo ================
if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: No trained model found.")
        print(f"Train first: py -3.11 run_textcnn.py")
        sys.exit(1)

    detector = RumorDetector()

    samples = [
        "BREAKING: Massive explosion reported in downtown area, hundreds feared dead",
        "Just had a lovely coffee with friends at the new cafe downtown",
        "Scientists confirm COVID-19 was artificially created in a Chinese lab",
        "The weather is beautiful today, going for a walk in the park",
        "URGENT: Government secretly planning to impose martial law next week!",
        "#SydneySiege UPDATE: Gunman identified, police on scene http://t.co/abc123",
    ]

    print("=" * 55)
    print("  Rumor Detector Demo (TextCNN + GloVe 200d)")
    print("=" * 55)
    for text in samples:
        label, conf = detector.predict(text)
        tag = "[RUMOR]  " if label == 1 else "[NON-RUM]"
        bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
        print(f"  {tag} conf={conf:.3f} {bar}")
        print(f"          {text[:80]}")
        print()
