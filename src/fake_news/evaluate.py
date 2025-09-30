from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from .data import load_dataset_csv
from .model import load_model, predict_label, predict_proba
from .config import DEFAULT_MODEL_DIR


def evaluate_main():
    parser = argparse.ArgumentParser(description="Evaluate fake news model")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report_path", type=str, default=None)
    args = parser.parse_args()

    df = load_dataset_csv(Path(args.data_path))
    model = load_model(Path(args.model_dir))

    y_true = df["label"].to_numpy()
    y_prob = np.array(predict_proba(model, df["text"]))
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    results = {"accuracy": acc, "report": report, "confusion_matrix": cm}

    if args.report_path:
        Path(args.report_path).write_text(json.dumps(results, indent=2))
        print(f"Saved report to {args.report_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate_main()
