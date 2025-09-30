"""
FastAPI service using only Gemini AI for fake news detection.
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .gemini_detector import predict_single_gemini, create_gemini_detector


class PredictRequest(BaseModel):
    texts: List[str]
    gemini_api_key: Optional[str] = None


class PredictResponse(BaseModel):
    results: List[dict]
    model_type: str = "gemini"
    total_texts: int


class HealthResponse(BaseModel):
    status: str
    gemini_available: bool
    model_info: dict


def create_gemini_only_app() -> FastAPI:
    """Create FastAPI app with only Gemini AI model."""
    app = FastAPI(
        title="AI Fake News Detector",
        description="Detect fake news using Google's Gemini AI",
        version="2.0.0"
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            gemini_available=True,
            model_info={
                "name": "Google Gemini 2.0 Flash",
                "type": "Large Language Model",
                "features": [
                    "Advanced language understanding",
                    "Detailed reasoning",
                    "Source quality assessment",
                    "Factual accuracy analysis",
                    "Red flag detection"
                ]
            }
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        """Predict fake news using Gemini AI model."""
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
            
            return PredictResponse(
                results=results,
                model_type="gemini",
                total_texts=len(req.texts)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini prediction failed: {str(e)}")

    @app.post("/predict/single")
    def predict_single(text: str, gemini_api_key: Optional[str] = None):
        """Predict fake news for a single text (convenience endpoint)."""
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="Gemini API key required"
            )
        
        try:
            result = predict_single_gemini(text, api_key)
            return {
                "text": text,
                "prediction": "FAKE" if result['is_fake'] else "REAL",
                "confidence": result['confidence'],
                "reasoning": result['reasoning'],
                "source_quality": result['source_quality'],
                "factual_accuracy": result['factual_accuracy'],
                "red_flags": result['red_flags']
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.get("/models")
    def list_models():
        """List available model information."""
        return {
            "gemini": {
                "available": True,
                "name": "Google Gemini 2.0 Flash",
                "type": "Large Language Model",
                "features": [
                    "Advanced language understanding",
                    "Detailed reasoning and analysis",
                    "Source quality assessment",
                    "Factual accuracy evaluation",
                    "Red flag identification",
                    "Context-aware predictions"
                ],
                "capabilities": {
                    "reasoning": True,
                    "source_analysis": True,
                    "red_flag_detection": True,
                    "confidence_scoring": True,
                    "detailed_explanations": True
                }
            }
        }

    @app.get("/")
    def root():
        """Root endpoint with API information."""
        return {
            "message": "AI Fake News Detector API",
            "version": "2.0.0",
            "model": "Google Gemini 2.0 Flash",
            "endpoints": {
                "POST /predict": "Analyze multiple texts",
                "POST /predict/single": "Analyze single text",
                "GET /health": "Health check",
                "GET /models": "Model information",
                "GET /docs": "API documentation"
            }
        }

    return app
