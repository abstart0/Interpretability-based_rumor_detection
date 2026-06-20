"""
CLI entry point: run rumor detection + explanation on a single text.

Usage:
    python main.py --text "Your tweet text here"
    python main.py --text-file data/val.csv          # explain all rows in CSV
    python main.py --text "Breaking news!" --temperature 0.7

Output format:
    Output 1: 2-class prediction (0=non-rumor, 1=rumor)
    Output 2: natural language explanation text
"""

import argparse
import json
import sys

from explanation.explainer import RumorExplainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rumor Detection + Explanation (Detection Model + LLM)"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Input text to analyze. Either this or --text-file must be provided.",
    )
    parser.add_argument(
        "--text-file", type=str, default=None,
        help="Path to a CSV file containing a 'text' column. Explains all rows.",
    )
    parser.add_argument(
        "--model", type=str, default="textcnn", choices=["textcnn", "bigru", "lr"],
        help="Detection model to use (default: textcnn).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="LLM temperature override (default from config).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="LLM max tokens override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.text and not args.text_file:
        print("Error: please provide --text or --text-file", file=sys.stderr)
        sys.exit(1)

    explainer = RumorExplainer(model=args.model)

    # Single text mode
    if args.text:
        result = explainer.explain(
            args.text,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        # Output 1: 2-class prediction label
        print("=== Prediction ===")
        print(f"label: {result['label']}")
        print(f"confidence: {result['confidence']:.4f}")
        # Output 2: natural language explanation
        print(f"\nexplanation: {result['explanation']}")
        # Also output as JSON for programmatic use
        print(f"\n--- JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Batch mode from CSV
    if args.text_file:
        import pandas as pd
        df = pd.read_csv(args.text_file)

        if "text" not in df.columns:
            print(f"Error: CSV must have a 'text' column, got columns: {df.columns.tolist()}",
                  file=sys.stderr)
            sys.exit(1)

        results = []
        for idx, text in enumerate(df["text"]):
            try:
                result = explainer.explain(
                    text,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                result["id"] = idx
                results.append(result)
            except Exception as e:
                print(f"Warning: failed to explain row {idx}: {e}", file=sys.stderr)
                results.append({
                    "id": idx,
                    "label": None,
                    "confidence": None,
                    "explanation": f"Error: {e}",
                })

        # Print each result with clear separation
        for r in results:
            print(f"\n--- Sample {r['id']} ---")
            print(f"label: {r['predicted_label']}")
            print(f"confidence: {r['confidence']}")
            print(f"explanation: {r['explanation']}")

        # Also save as JSON
        print(f"\n=== Full JSON results ===")
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
