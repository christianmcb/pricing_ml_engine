from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report


def evaluate_classifier(model, X_test, y_test) -> dict:
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
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    result_row["timestamp"] = datetime.utcnow().isoformat()

    df = pd.DataFrame([result_row])

    if Path(file_path).exists():
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)