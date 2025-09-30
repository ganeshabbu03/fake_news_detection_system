import argparse
from pathlib import Path

import streamlit as st

from src.fake_news.model import load_model, predict_proba, predict_label
from src.fake_news.config import DEFAULT_MODEL_DIR


def main(model_dir: Path):
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    st.title("📰 Fake News Detector")

    text = st.text_area("Enter news text", height=200, placeholder="Paste an article or headline...")
    threshold = st.slider("Decision threshold (fake if ≥ threshold)", 0.0, 1.0, 0.5, 0.01)

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            model = load_model(model_dir)
            prob = predict_proba(model, [text])[0]
            label = 1 if prob >= threshold else 0
            st.metric("Probability of Fake", f"{prob:.3f}")
            st.write("Prediction:", "Fake" if label == 1 else "Real")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()
    main(Path(args.model_dir))
