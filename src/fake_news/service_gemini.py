"""
Enhanced FastAPI service with both traditional ML and Gemini AI models.
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import load_model, predict_proba, predict_label
from .gemini_detector import predict_single_gemini, create_gemini_detector
from .config import DEFAULT_MODEL_DIR


class PredictRequest(BaseModel):
    texts: List[str]
    threshold: float = 0.5
    model_type: str = "traditional"  # "traditional" or "gemini"
    gemini_api_key: Optional[str] = None


class PredictResponse(BaseModel):
    labels: List[int]
    probabilities: List[float]
    model_type: str
    details: Optional[List[dict]] = None


class GeminiPredictRequest(BaseModel):
    texts: List[str]
    gemini_api_key: Optional[str] = None


class GeminiPredictResponse(BaseModel):
    results: List[dict]
    model_type: str = "gemini"


def create_enhanced_app(model_dir: Path | str = DEFAULT_MODEL_DIR) -> FastAPI:
    """Create enhanced FastAPI app with both models."""
    app = FastAPI(
        title="AI-Powered Fake News Detector",
        description="Detect fake news using traditional ML or Google's Gemini AI",
        version="2.0.0"
    )
    
    # Load traditional model
    try:
        traditional_model = load_model(model_dir)
        model_loaded = True
    except FileNotFoundError:
        traditional_model = None
        model_loaded = False

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "traditional_model_loaded": model_loaded,
            "gemini_available": True
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        """Predict using traditional ML model."""
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Traditional model not loaded")
        
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts must be non-empty")
        
        probs = predict_proba(traditional_model, req.texts)
        labels = predict_label(traditional_model, req.texts, threshold=req.threshold)
        
        return PredictResponse(
            labels=labels,
            probabilities=probs,
            model_type="traditional"
        )

    @app.post("/predict/gemini", response_model=GeminiPredictResponse)
    def predict_gemini(req: GeminiPredictRequest):
        """Predict using Gemini AI model."""
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts must be non-empty")
        
        api_key = req.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="Gemini API key required. Set GEMINI_API_KEY env var or pass in request"
            )
        
        try:
            results = []
            for text in req.texts:
                result = predict_single_gemini(text, api_key)
                results.append(result)
            
            return GeminiPredictResponse(
                results=results,
                model_type="gemini"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini prediction failed: {str(e)}")

    @app.post("/predict/compare")
    def predict_compare(req: GeminiPredictRequest):
        """Compare predictions from both models."""
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts must be non-empty")
        
        api_key = req.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="Gemini API key required for comparison"
            )
        
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Traditional model not loaded")
        
        results = []
        
        for text in req.texts:
            # Traditional ML prediction
            traditional_prob = predict_proba(traditional_model, [text])[0]
            traditional_label = 1 if traditional_prob >= 0.5 else 0
            
            # Gemini prediction
            try:
                gemini_result = predict_single_gemini(text, api_key)
                gemini_label = 1 if gemini_result['is_fake'] else 0
                gemini_confidence = gemini_result['confidence']
            except Exception as e:
                gemini_result = {"error": str(e)}
                gemini_label = None
                gemini_confidence = 0.0
            
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "traditional": {
                    "label": traditional_label,
                    "probability": traditional_prob,
                    "prediction": "Fake" if traditional_label == 1 else "Real"
                },
                "gemini": {
                    "label": gemini_label,
                    "confidence": gemini_confidence,
                    "prediction": "Fake" if gemini_label == 1 else "Real" if gemini_label == 0 else "Error",
                    "reasoning": gemini_result.get('reasoning', 'Analysis failed'),
                    "details": gemini_result
                },
                "agreement": traditional_label == gemini_label if gemini_label is not None else None
            })
        
        return {
            "results": results,
            "summary": {
                "total_texts": len(req.texts),
                "agreements": sum(1 for r in results if r['agreement'] is True),
                "disagreements": sum(1 for r in results if r['agreement'] is False),
                "errors": sum(1 for r in results if r['gemini']['label'] is None)
            }
        }

    @app.get("/models")
    def list_models():
        """List available models and their status."""
        return {
            "traditional": {
                "available": model_loaded,
                "type": "TF-IDF + Logistic Regression",
                "accuracy": "99.4% on test data"
            },
            "gemini": {
                "available": True,
                "type": "Google Gemini 1.5 Flash",
                "features": [
                    "Advanced language understanding",
                    "Detailed reasoning",
                    "Source quality assessment",
                    "Factual accuracy analysis"
                ]
            }
        }

    return app
