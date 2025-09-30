#!/usr/bin/env python3
"""
Run the enhanced FastAPI service with both traditional ML and Gemini AI models.
"""

import argparse
import uvicorn
from pathlib import Path

from src.fake_news.service_gemini import create_enhanced_app
from src.fake_news.config import DEFAULT_MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description="Run enhanced fake news detection API")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()

    app = create_enhanced_app(Path(args.model_dir))
    
    print("🚀 Starting Enhanced Fake News Detection API")
    print(f"📍 Model directory: {args.model_dir}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\nAvailable endpoints:")
    print("  • POST /predict - Traditional ML model")
    print("  • POST /predict/gemini - Gemini AI model")
    print("  • POST /predict/compare - Compare both models")
    print("  • GET /models - List available models")
    print("  • GET /health - Health check")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
