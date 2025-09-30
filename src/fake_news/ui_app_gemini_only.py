"""
Streamlit UI using only Gemini AI for fake news detection.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import streamlit as st

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.fake_news.gemini_detector import predict_single_gemini, create_gemini_detector


def main(gemini_api_key: str = None):
    st.set_page_config(
        page_title="AI Fake News Detector", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI-Powered Fake News Detector")
    st.markdown("Powered by Google's Gemini AI")
    st.markdown("---")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API key input
        if not gemini_api_key:
            gemini_api_key = st.text_input(
                "Gemini API Key:",
                type="password",
                help="Get your API key from https://makersuite.google.com/app/apikey"
            )
        
        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API key to use the AI model")
            return
        
        st.success("✅ Gemini AI Ready!")
        
        st.markdown("---")
        st.markdown("**About Gemini AI:**")
        st.markdown("""
        - Advanced language understanding
        - Detailed reasoning and analysis
        - Considers multiple factors:
          • Source credibility
          • Language patterns
          • Factual claims
          • Emotional manipulation
          • Red flag detection
        """)
        
        st.markdown("---")
        st.markdown("**Tips for better results:**")
        st.markdown("""
        - Use complete sentences
        - Include context when possible
        - Longer text generally gives better results
        - The AI analyzes writing style and content quality
        """)
    
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
        if st.button("🔍 Analyze with AI", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("AI is analyzing..."):
                    analyze_gemini_ai(text, gemini_api_key)
    
    with col2:
        st.header("📊 Quick Stats")
        
        # Placeholder for stats
        st.info("""
        **Analysis Features:**
        - ✅ Source Quality Assessment
        - ✅ Factual Accuracy Analysis
        - ✅ Red Flag Detection
        - ✅ Detailed Reasoning
        - ✅ Confidence Scoring
        """)
        
        st.markdown("---")
        st.markdown("**Sample Analysis:**")
        st.markdown("""
        Try these examples:
        - "Breaking: Scientists discover cure for cancer!"
        - "WASHINGTON (Reuters) - Federal Reserve raises interest rates"
        - "You won't believe this one weird trick!"
        """)


def analyze_gemini_ai(text: str, api_key: str):
    """Analyze text using Gemini AI model."""
    try:
        result = predict_single_gemini(text, api_key)
        
        # Display results
        st.subheader("🧠 AI Analysis Results")
        
        # Main prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_color = "🔴" if result['is_fake'] else "🟢"
            st.metric(
                "Prediction",
                f"{prediction_color} {'FAKE NEWS' if result['is_fake'] else 'REAL NEWS'}",
                delta=f"{result['confidence']:.1%} confidence"
            )
        
        with col2:
            confidence_level = "High" if result['confidence'] > 0.8 else "Medium" if result['confidence'] > 0.6 else "Low"
            st.metric(
                "Confidence Level",
                f"{result['confidence']:.1%}",
                delta=confidence_level
            )
        
        with col3:
            quality_emoji = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}.get(result['source_quality'], "⚪")
            st.metric(
                "Source Quality",
                f"{quality_emoji} {result['source_quality'].title()}",
                delta="Reliable" if result['source_quality'] in ['good', 'excellent'] else "Questionable"
            )
        
        # Detailed analysis
        st.subheader("🔍 Detailed Analysis")
        
        # Reasoning
        with st.expander("📝 AI Reasoning", expanded=True):
            st.write(result['reasoning'])
        
        # Quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Source Quality:**")
            quality_bar = {"poor": 0.2, "fair": 0.4, "good": 0.7, "excellent": 0.9}.get(result['source_quality'], 0.5)
            st.progress(quality_bar)
            st.caption(f"Rated as: {result['source_quality'].title()}")
        
        with col2:
            st.write("**Factual Accuracy:**")
            accuracy_bar = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(result['factual_accuracy'], 0.5)
            st.progress(accuracy_bar)
            st.caption(f"Rated as: {result['factual_accuracy'].title()}")
        
        # Red flags
        if result['red_flags']:
            st.subheader("🚨 Red Flags Detected")
            for i, flag in enumerate(result['red_flags'], 1):
                st.write(f"{i}. {flag}")
        else:
            st.success("✅ No major red flags detected")
        
        # Raw response (collapsible)
        with st.expander("🔧 View Raw AI Response"):
            st.code(result['raw_response'], language="json")
        
    except Exception as e:
        st.error(f"Error analyzing text with Gemini: {str(e)}")
        st.info("Make sure you have a valid Gemini API key and internet connection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini_api_key", type=str, help="Gemini API key")
    args = parser.parse_args()
    
    main(args.gemini_api_key)
