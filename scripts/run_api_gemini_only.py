#!/usr/bin/env python3
"""
Run the Gemini-only FastAPI service.
"""

import argparse
import uvicorn

from src.fake_news.service_gemini_only import create_gemini_only_app


def main():
    parser = argparse.ArgumentParser(description="Run Gemini-only fake news detection API")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()

    app = create_gemini_only_app()
    
    print("🤖 Starting AI Fake News Detection API")
    print("Powered by Google Gemini 2.0 Flash")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\nAvailable endpoints:")
    print("  • POST /predict - Analyze multiple texts")
    print("  • POST /predict/single - Analyze single text")
    print("  • GET /health - Health check")
    print("  • GET /models - Model information")
    print("  • GET / - API overview")
    print("\nNote: Make sure GEMINI_API_KEY environment variable is set")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
