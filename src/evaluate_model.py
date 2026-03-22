from datetime import datetime
import json
import argparse
import sys
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

from src.config import load_config
from src.data_processing import (
    load_train_data,
    split_features_target,
    validate_training_dataframe,
)


MIN_TEST_AUC = 0.75


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    return parser.parse_args()


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Returns ROC AUC and classification report for a trained model on the test set."""
    conversion_probability = model.predict_proba(X_test)[:, 1]
    predicted_conversion = model.predict(X_test)

    return {
        "test_auc": roc_auc_score(y_test, conversion_probability),
        "classification_report": classification_report(
            y_test,
            predicted_conversion,
            output_dict=True,
        ),
    }


def save_results_csv(results_df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def save_json(data: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def log_experiment(result_row: dict, file_path="outputs/experiments.csv"):
    """Appends a timestamped experiment result row to the experiments CSV log."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    result_row["timestamp"] = datetime.utcnow().isoformat()

    df = pd.DataFrame([result_row])

    if Path(file_path).exists():
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)


def main():
    """Loads a registry model by run ID and exits non-zero if AUC is below the acceptance threshold."""
    args = parse_args()
    config = load_config()

    run_dir = Path("models/registry") / args.run_id
    model_path = run_dir / "model.joblib"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    model = joblib.load(model_path)

    df = load_train_data(config["data"]["train_path"])
    validate_training_dataframe(df, target_col=config["data"]["target_column"])
    X, y = split_features_target(df, target_col=config["data"]["target_column"])

    results = evaluate_classifier(model, X, y)
    test_auc = results["test_auc"]

    print(f"Run: {args.run_id}")
    print(f"Test AUC: {test_auc:.4f}")

    if test_auc < MIN_TEST_AUC:
        print(f"Rejected: AUC below threshold {MIN_TEST_AUC:.2f}")
        sys.exit(1)

    print("Accepted")
    sys.exit(0)


if __name__ == "__main__":
    main()
