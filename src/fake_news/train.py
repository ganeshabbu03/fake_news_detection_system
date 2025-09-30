from __future__ import annotations

import argparse
from pathlib import Path
import json

from .config import TrainingConfig, DEFAULT_MODEL_DIR
from .data import load_dataset_csv, train_val_split
from .model import train_model, save_model


def train_main():
    parser = argparse.ArgumentParser(description="Train fake news detector")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=1.5)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    cfg = TrainingConfig(
        model_dir=Path(args.model_dir),
        val_size=args.val_size,
        random_state=args.random_state,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        C=args.C,
    )

    df = load_dataset_csv(Path(args.data_path))
    ds = train_val_split(
        df,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    pipeline, metrics = train_model(
        ds.X_train,
        ds.y_train,
        ds.X_val,
        ds.y_val,
        max_features=cfg.max_features,
        C=cfg.C,
        n_jobs=cfg.n_jobs,
    )

    model_path = save_model(pipeline, cfg.model_dir)
    (cfg.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {cfg.model_dir / 'metrics.json'}")


if __name__ == "__main__":
    train_main()
