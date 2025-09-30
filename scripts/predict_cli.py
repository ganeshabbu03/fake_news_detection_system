import argparse
from pathlib import Path

from src.fake_news.model import load_model, predict_label, predict_proba
from src.fake_news.config import DEFAULT_MODEL_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = load_model(Path(args.model_dir))
    prob = predict_proba(model, [args.text])[0]
    label = 1 if prob >= args.threshold else 0

    print({"probability_fake": prob, "label": label})


if __name__ == "__main__":
    main()
