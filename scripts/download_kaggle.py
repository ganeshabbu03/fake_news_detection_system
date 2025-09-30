import argparse
from pathlib import Path
import sys

import kagglehub


def list_csvs(base_path: Path):
    return sorted(base_path.rglob("*.csv"))


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle dataset via kagglehub")
    parser.add_argument("--dataset", type=str, default="ghost15/fake-detection-system")
    args = parser.parse_args()

    path = kagglehub.dataset_download(args.dataset)
    print("Path to dataset files:", path)
    csvs = list_csvs(Path(path))
    if not csvs:
        print("No CSV files found under:", path)
        sys.exit(1)
    print("Found CSV files:")
    for p in csvs:
        print(" -", p)


if __name__ == "__main__":
    main()
