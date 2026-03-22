import os

import joblib
import pandas as pd

from src.config import load_config
from src.data_processing import validate_inference_dataframe
from src.model_registry import get_model_path


def _load_sample_features(config: dict) -> pd.DataFrame | None:
    """Loads a small feature sample for Deeploi schema inference."""
    train_path = config["data"]["train_path"]
    target_column = config["data"]["target_column"]

    try:
        sample_df = pd.read_csv(train_path, nrows=200)
    except Exception:
        return None

    if target_column in sample_df.columns:
        sample_df = sample_df.drop(columns=[target_column])

    try:
        return validate_inference_dataframe(sample_df)
    except Exception:
        return None


def main() -> None:
    """Serves the current promoted model using Deeploi."""
    try:
        from deeploi import deploy
    except ImportError as exc:
        raise ImportError(
            "Deeploi is required to serve the API. Install it with: pip install deeploi"
        ) from exc

    config = load_config()
    model_path = get_model_path(config["artifacts"]["model_path"])
    model = joblib.load(model_path)
    sample = _load_sample_features(config)

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    deploy(model=model, sample=sample, host=host, port=port)


if __name__ == "__main__":
    main()
