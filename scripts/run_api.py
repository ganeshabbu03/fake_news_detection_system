import argparse
import uvicorn
from pathlib import Path

from src.fake_news.service import create_app
from src.fake_news.config import DEFAULT_MODEL_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(Path(args.model_dir))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
