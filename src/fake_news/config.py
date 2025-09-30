from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    model_dir: Path
    val_size: float = 0.2
    random_state: int = 42
    max_features: int = 50000
    n_jobs: int = -1
    C: float = 1.5


@dataclass(frozen=True)
class ServiceConfig:
    model_dir: Path
    host: str = "0.0.0.0"
    port: int = 8000


DEFAULT_MODEL_DIR = Path("models")
