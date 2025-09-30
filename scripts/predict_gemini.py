#!/usr/bin/env python3
"""
CLI script for testing Gemini-based fake news detection.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.fake_news.gemini_detector import predict_single_gemini


def main():
    parser = argparse.ArgumentParser(description="Detect fake news using Gemini AI")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--api-key", type=str, help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    try:
        result = predict_single_gemini(args.text, args.api_key)
        
        if args.verbose:
            print("=== GEMINI FAKE NEWS ANALYSIS ===")
            print(f"Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
            print(f"Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Source Quality: {result['source_quality']}")
            print(f"Factual Accuracy: {result['factual_accuracy']}")
            if result['red_flags']:
                print(f"Red Flags: {', '.join(result['red_flags'])}")
            print("=" * 40)
        else:
            # Simple output for compatibility
            print({
                "is_fake": result['is_fake'],
                "confidence": result['confidence'],
                "reasoning": result['reasoning']
            })
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
