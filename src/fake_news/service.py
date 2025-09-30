from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import load_model, predict_proba, predict_label
from .config import DEFAULT_MODEL_DIR


class PredictRequest(BaseModel):
    texts: List[str]
    threshold: float | None = 0.5


class PredictResponse(BaseModel):
    labels: List[int]
    probabilities: List[float]


def create_app(model_dir: Path | str = DEFAULT_MODEL_DIR) -> FastAPI:
    app = FastAPI(title="Fake News Detector")
    try:
        model = load_model(model_dir)
    except FileNotFoundError as e:
        raise e

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts must be non-empty")
        probs = predict_proba(model, req.texts)
        labels = predict_label(model, req.texts, threshold=req.threshold or 0.5)
        return PredictResponse(labels=labels, probabilities=probs)

    return app
