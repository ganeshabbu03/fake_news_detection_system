#!/usr/bin/env python3
"""
CLI script for AI-powered fake news detection using Gemini.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.fake_news.gemini_detector import predict_single_gemini


def main():
    parser = argparse.ArgumentParser(description="AI-powered fake news detection")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--api-key", type=str, help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed analysis")
    parser.add_argument("--simple", "-s", action="store_true", help="Show only prediction and confidence")
    
    args = parser.parse_args()
    
    try:
        result = predict_single_gemini(args.text, args.api_key)
        
        if args.simple:
            # Simple output
            prediction = "FAKE" if result['is_fake'] else "REAL"
            confidence = f"{result['confidence']:.1%}"
            print(f"{prediction} ({confidence})")
        elif args.verbose:
            # Detailed output
            print("=" * 60)
            print("AI FAKE NEWS ANALYSIS")
            print("=" * 60)
            print(f"Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
            print(f"Prediction: {'FAKE NEWS' if result['is_fake'] else 'REAL NEWS'}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Source Quality: {result['source_quality'].title()}")
            print(f"Factual Accuracy: {result['factual_accuracy'].title()}")
            print("\nReasoning:")
            print(result['reasoning'])
            
            if result['red_flags']:
                print("\nRed Flags Detected:")
                for i, flag in enumerate(result['red_flags'], 1):
                    print(f"  {i}. {flag}")
            else:
                print("\nNo major red flags detected")
            print("=" * 60)
        else:
            # Standard output
            prediction = "FAKE" if result['is_fake'] else "REAL"
            confidence = result['confidence']
            reasoning = result['reasoning']
            
            print({
                "prediction": prediction,
                "confidence": confidence,
                "reasoning": reasoning,
                "source_quality": result['source_quality'],
                "factual_accuracy": result['factual_accuracy']
            })
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
