import argparse
from pathlib import Path

import kagglehub

from src.fake_news.train import train_main as _train_main


def create_combined_dataset(base_path: Path) -> Path:
    """Create a combined dataset from fake.csv and true.csv files."""
    import pandas as pd
    
    fake_path = base_path / "fake.csv"
    true_path = base_path / "true.csv"
    
    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError("Expected fake.csv and true.csv files in dataset")
    
    # Load both datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add labels
    fake_df['label'] = 1  # Fake news
    true_df['label'] = 0  # Real news
    
    # Combine datasets
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save combined dataset
    output_path = base_path / "combined_dataset.csv"
    combined_df.to_csv(output_path, index=False)
    
    print(f"Created combined dataset with {len(combined_df)} samples")
    print(f"Fake news: {len(fake_df)} samples")
    print(f"Real news: {len(true_df)} samples")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle dataset and train")
    parser.add_argument("--dataset", type=str, default="ghost15/fake-detection-system")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=1.5)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args, unknown = parser.parse_known_args()

    path = kagglehub.dataset_download(args.dataset)
    csv_path = create_combined_dataset(Path(path))
    print("Using CSV:", csv_path)

    # Build argv to reuse train_main
    import sys
    sys.argv = [
        "train_model",
        "--data_path", str(csv_path),
        "--model_dir", args.model_dir,
        "--val_size", str(args.val_size),
        "--random_state", str(args.random_state),
        "--max_features", str(args.max_features),
        "--C", str(args.C),
        "--n_jobs", str(args.n_jobs),
    ]
    _train_main()


if __name__ == "__main__":
    main()
