from pathlib import Path
import joblib


def test_current_model_loads():
    model_path = Path("models/current/model.joblib")
    assert model_path.exists(), "No promoted model found"

    model = joblib.load(model_path)
    assert hasattr(model, "predict")
