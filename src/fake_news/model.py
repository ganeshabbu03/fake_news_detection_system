from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from .preprocess import preprocess_texts


def build_pipeline(max_features: int = 50000, C: float = 1.5, n_jobs: int = -1) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
    )
    classifier = LogisticRegression(max_iter=200, n_jobs=n_jobs, C=C)
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def train_model(
    X_train: Iterable[str],
    y_train: Iterable[int],
    X_val: Iterable[str],
    y_val: Iterable[int],
    max_features: int = 50000,
    C: float = 1.5,
    n_jobs: int = -1,
) -> Tuple[Pipeline, dict]:
    X_train_p = preprocess_texts(X_train)
    X_val_p = preprocess_texts(X_val)
    pipeline = build_pipeline(max_features=max_features, C=C, n_jobs=n_jobs)
    pipeline.fit(X_train_p, y_train)
    y_pred = pipeline.predict(X_val_p)
    report = classification_report(y_val, y_pred, output_dict=True)
    acc = accuracy_score(y_val, y_pred)
    metrics = {"accuracy": acc, "report": report}
    return pipeline, metrics


def save_model(pipeline: Pipeline, model_dir: Path | str) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


def load_model(model_dir: Path | str) -> Pipeline:
    model_path = Path(model_dir) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def predict_proba(pipeline: Pipeline, texts: Iterable[str]) -> List[float]:
    processed = preprocess_texts(texts)
    # Probability for class 1 (fake)
    probs = pipeline.predict_proba(processed)[:, 1]
    return probs.tolist()


def predict_label(pipeline: Pipeline, texts: Iterable[str], threshold: float = 0.5) -> List[int]:
    probs = predict_proba(pipeline, texts)
    return [1 if p >= threshold else 0 for p in probs]
