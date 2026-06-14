"""
Batch evaluation: run rumor detection + explanation on the entire validation set.

Usage:
    python evaluate.py --model textcnn --val-file data/val.csv
    python evaluate.py --model bigru --val-file data/val.csv
    python evaluate.py --model lr --val-file data/val.csv

Output:
    Prints a summary table with accuracy/precision/recall/F1 for the detection model,
    and saves a JSON file containing all predictions with explanations.
"""

import argparse
import json
import sys
import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation with explanations")
    parser.add_argument(
        "--model", type=str, default="textcnn",
        choices=["textcnn", "bigru", "lr"],
        help="Detection model to use (default: textcnn).",
    )
    parser.add_argument(
        "--val-file", type=str, default="data/val.csv",
        help="Path to the validation CSV file.",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json",
        help="Path to save the JSON output with explanations.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="LLM temperature override.",
    )
    return parser.parse_args()


def load_evaluator(model_name: str):
    """
    Load the appropriate detection model based on the --model argument.

    Returns:
        A callable that takes a string and returns (label, confidence).
    """
    if model_name == "textcnn":
        from predictor import RumorDetector
        detector = RumorDetector()
        return detector.predict

    elif model_name == "bigru":
        from scripts.predict_bigru import RumorClassifier
        clf = RumorClassifier()
        # classify() returns int label only, wrap to also return confidence
        def predict_bigru(text: str):
            label = clf.classify(text)
            # BiGRU doesn't expose confidence, return 0.5 as placeholder
            return int(label), 0.5

        return predict_bigru

    elif model_name == "lr":
        from scripts.predict_lr import LRRumorClassifier
        clf = LRRumorClassifier()

        def predict_lr(text: str):
            label = clf.classify(text)
            return int(label), 0.5

        return predict_lr

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    args = parse_args()

    # Load validation data
    val_df = pd.read_csv(args.val_file)
    texts = val_df["text"].tolist()
    true_labels = val_df["label"].tolist()

    # Load detection model
    predict_fn = load_evaluator(args.model)

    # Run predictions
    logger.info("Running predictions on %d samples...", len(texts))
    predictions = []
    labels = []
    confidences = []

    for i, text in enumerate(texts):
        try:
            label, confidence = predict_fn(text)
            predictions.append(text)
            labels.append(label)
            confidences.append(confidence)
        except Exception as e:
            logger.warning("Failed to predict row %d: %s", i, e)
            predictions.append(text)
            labels.append(None)
            confidences.append(None)

    # Compute detection metrics (skip failed predictions)
    valid_mask = [i for i, l in enumerate(labels) if l is not None]
    if valid_mask:
        valid_preds = [labels[i] for i in valid_mask]
        valid_true = [true_labels[i] for i in valid_mask]

        acc = accuracy_score(valid_true, valid_preds)
        prec = precision_score(valid_true, valid_preds, zero_division=0)
        rec = recall_score(valid_true, valid_preds, zero_division=0)
        f1 = f1_score(valid_true, valid_preds, zero_division=0)

        print(f"\n{'='*50}")
        print(f"  Detection Model: {args.model.upper()}")
        print(f"  Samples: {len(valid_true)}")
        print(f"{'='*50}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"{'='*50}\n")
        print(classification_report(valid_true, valid_preds,
                                    target_names=["Non-rumor", "Rumor"], digits=4))

    # Save results
    results = []
    for i in range(len(texts)):
        entry = {
            "id": i,
            "text": texts[i],
            "true_label": true_labels[i],
            "predicted_label": labels[i],
            "confidence": confidences[i],
        }
        results.append(entry)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
