from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Dataset:
    X_train: pd.Series
    X_val: pd.Series
    y_train: pd.Series
    y_val: pd.Series


def _normalize_labels(series: pd.Series) -> pd.Series:
    # Map common string labels to integers
    mapping = {
        "FAKE": 1,
        "fake": 1,
        "FALSE": 1,
        "false": 1,
        "REAL": 0,
        "real": 0,
        "TRUE": 0,
        "true": 0,
    }
    if series.dtype == object:
        return series.map(lambda v: mapping.get(v, v)).astype(int)
    return series.astype(int)


def load_dataset_csv(csv_path: Path | str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{text_col}' and '{label_col}' in dataset")
    df = df[[text_col, label_col]].dropna()
    df[label_col] = _normalize_labels(df[label_col])
    df[text_col] = df[text_col].astype(str)
    return df


def train_val_split(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    val_size: float = 0.2,
    random_state: int = 42,
) -> Dataset:
    X = df[text_col]
    y = df[label_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    return Dataset(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
