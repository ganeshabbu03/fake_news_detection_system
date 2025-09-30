"""
Enhanced Streamlit UI with both traditional ML and Gemini AI models.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import streamlit as st

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.fake_news.model import load_model, predict_proba, predict_label
from src.fake_news.gemini_detector import predict_single_gemini, create_gemini_detector
from src.fake_news.config import DEFAULT_MODEL_DIR


def main(model_dir: Path, gemini_api_key: str = None):
    st.set_page_config(
        page_title="Fake News Detector AI", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI-Powered Fake News Detector")
    st.markdown("---")
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_type = st.selectbox(
            "Choose Detection Model:",
            ["Traditional ML (TF-IDF + Logistic Regression)", "Gemini AI (Advanced Analysis)"],
            help="Traditional ML is faster, Gemini provides more detailed analysis"
        )
        
        # Threshold for traditional model
        if "Traditional ML" in model_type:
            threshold = st.slider(
                "Decision threshold (fake if ≥ threshold)", 
                0.0, 1.0, 0.5, 0.01,
                help="Lower values = more sensitive to fake news"
            )
        
        # API key input for Gemini
        if "Gemini AI" in model_type:
            if not gemini_api_key:
                gemini_api_key = st.text_input(
                    "Gemini API Key:",
                    type="password",
                    help="Get your API key from https://makersuite.google.com/app/apikey"
                )
            
            if not gemini_api_key:
                st.warning("⚠️ Please enter your Gemini API key to use the AI model")
                return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Analysis")
        
        # Text input
        text = st.text_area(
            "Enter news text or headline:",
            height=200,
            placeholder="Paste an article, headline, or any text you want to analyze...",
            help="The longer and more detailed the text, the better the analysis"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    if "Traditional ML" in model_type:
                        analyze_traditional_ml(text, model_dir, threshold)
                    else:
                        analyze_gemini_ai(text, gemini_api_key)
    
    with col2:
        st.header("ℹ️ About")
        
        if "Traditional ML" in model_type:
            st.info("""
            **Traditional ML Model:**
            - Fast and efficient
            - Trained on 44,898 news articles
            - 99.4% accuracy on test data
            - Uses TF-IDF + Logistic Regression
            """)
        else:
            st.info("""
            **Gemini AI Model:**
            - Advanced language understanding
            - Detailed reasoning and analysis
            - Considers multiple factors:
              • Source credibility
              • Language patterns
              • Factual claims
              • Emotional manipulation
            """)
        
        st.markdown("---")
        st.markdown("**Tips for better results:**")
        st.markdown("""
        - Use complete sentences
        - Include context when possible
        - Longer text generally gives better results
        - Try both models for comparison
        """)


def analyze_traditional_ml(text: str, model_dir: Path, threshold: float):
    """Analyze text using traditional ML model."""
    try:
        model = load_model(model_dir)
        prob = predict_proba(model, [text])[0]
        label = 1 if prob >= threshold else 0
        
        # Display results
        st.subheader("📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Probability of Fake News",
                f"{prob:.1%}",
                delta=f"{prob-threshold:.1%}" if prob != threshold else None
            )
        
        with col2:
            st.metric(
                "Prediction",
                "🚨 FAKE" if label == 1 else "✅ REAL",
                delta="Above threshold" if label == 1 else "Below threshold"
            )
        
        with col3:
            st.metric(
                "Confidence",
                f"{max(prob, 1-prob):.1%}",
                delta="High" if max(prob, 1-prob) > 0.8 else "Medium" if max(prob, 1-prob) > 0.6 else "Low"
            )
        
        # Additional info
        st.info(f"Threshold used: {threshold:.2f}")
        
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")


def analyze_gemini_ai(text: str, api_key: str):
    """Analyze text using Gemini AI model."""
    try:
        result = predict_single_gemini(text, api_key)
        
        # Display results
        st.subheader("🧠 AI Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Prediction",
                "🚨 FAKE NEWS" if result['is_fake'] else "✅ REAL NEWS",
                delta=f"{result['confidence']:.1%} confidence"
            )
            
            st.metric(
                "Confidence Level",
                f"{result['confidence']:.1%}",
                delta="High" if result['confidence'] > 0.8 else "Medium" if result['confidence'] > 0.6 else "Low"
            )
        
        with col2:
            st.metric(
                "Source Quality",
                result['source_quality'].title(),
                delta="Good" if result['source_quality'] in ['good', 'excellent'] else "Poor" if result['source_quality'] == 'poor' else "Unknown"
            )
            
            st.metric(
                "Factual Accuracy",
                result['factual_accuracy'].title(),
                delta="High" if result['factual_accuracy'] == 'high' else "Low" if result['factual_accuracy'] == 'low' else "Unknown"
            )
        
        # Detailed analysis
        st.subheader("🔍 Detailed Analysis")
        
        st.write("**Reasoning:**")
        st.write(result['reasoning'])
        
        if result['red_flags']:
            st.write("**Red Flags Detected:**")
            for flag in result['red_flags']:
                st.write(f"• {flag}")
        
        # Raw response (collapsible)
        with st.expander("View Raw AI Response"):
            st.code(result['raw_response'], language="json")
        
    except Exception as e:
        st.error(f"Error analyzing text with Gemini: {str(e)}")
        st.info("Make sure you have a valid Gemini API key and internet connection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--gemini_api_key", type=str, help="Gemini API key")
    args = parser.parse_args()
    
    main(Path(args.model_dir), args.gemini_api_key)
