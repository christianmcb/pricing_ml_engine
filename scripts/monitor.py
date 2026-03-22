import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_processing import (
    REQUIRED_FEATURE_COLUMNS,
    load_test_data,
    validate_inference_dataframe,
)
from src.live_ops_utils import extract_batch_order, utc_now_iso

EPSILON = 1e-6
DEFAULT_MONITORING_DIR = "outputs/monitoring"


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Monitor live pricing predictions for drift and performance."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/live_predictions.csv",
        help="Scored live predictions from run_live_inference.py",
    )
    parser.add_argument(
        "--baseline",
        default=config["data"]["test_path"],
        help="Reference dataset used as baseline for drift comparisons",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_MONITORING_DIR,
        help="Directory for monitoring outputs",
    )
    parser.add_argument(
        "--rolling-window-batches",
        type=int,
        default=5,
        help="Number of recent batches to use for rolling summaries",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.2,
        help="PSI threshold for drift alerting",
    )
    parser.add_argument(
        "--pred-mean-shift-threshold",
        type=float,
        default=0.05,
        help="Alert when mean conversion probability shifts by this amount",
    )
    return parser.parse_args()


def load_predictions(predictions_path: str) -> pd.DataFrame:
    """Loads and validates the live predictions CSV, sorted by batch order."""
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path)
    required_prediction_cols = {
        "batch_id",
        "conversion_probability",
        "predicted_conversion",
        "recommended_premium",
    }
    missing = required_prediction_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing columns: {sorted(missing)}")

    feature_subset = validate_inference_dataframe(df[REQUIRED_FEATURE_COLUMNS])
    df = df.copy()
    for col in REQUIRED_FEATURE_COLUMNS:
        df[col] = feature_subset[col]

    if "inference_timestamp_utc" in df.columns:
        df["inference_timestamp_utc"] = pd.to_datetime(
            df["inference_timestamp_utc"], errors="coerce", utc=True
        )
    else:
        df["inference_timestamp_utc"] = pd.NaT

    df["batch_order"] = df["batch_id"].map(extract_batch_order)
    return df.sort_values(["batch_order", "inference_timestamp_utc"], kind="stable")


def load_baseline_features(baseline_path: str) -> pd.DataFrame:
    baseline_df = load_test_data(baseline_path)
    return validate_inference_dataframe(baseline_df)


def make_numeric_bins(reference: pd.Series, max_bins: int = 10) -> np.ndarray:
    clean = reference.dropna().astype(float)
    if clean.empty:
        return np.array([-np.inf, np.inf])

    quantiles = np.linspace(0.0, 1.0, max_bins + 1)
    edges = np.unique(np.quantile(clean, quantiles))

    if len(edges) == 1:
        value = float(edges[0])
        edges = np.array([value - 1.0, value + 1.0])

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi_from_distributions(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected.astype(float), EPSILON, None)
    actual = np.clip(actual.astype(float), EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def calculate_numeric_psi(reference: pd.Series, current: pd.Series) -> float:
    bins = make_numeric_bins(reference)
    ref_counts = pd.cut(reference.astype(float), bins=bins, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    cur_counts = pd.cut(current.astype(float), bins=bins, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    ref_dist = ref_counts.to_numpy(dtype=float)
    cur_dist = cur_counts.to_numpy(dtype=float)
    return psi_from_distributions(ref_dist, cur_dist)


def calculate_categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    ref_dist = reference.astype(str).value_counts(normalize=True)
    cur_dist = current.astype(str).value_counts(normalize=True)

    all_categories = sorted(set(ref_dist.index).union(set(cur_dist.index)))
    ref_values = np.array([ref_dist.get(cat, 0.0) for cat in all_categories], dtype=float)
    cur_values = np.array([cur_dist.get(cat, 0.0) for cat in all_categories], dtype=float)
    return psi_from_distributions(ref_values, cur_values)


def calculate_feature_psi(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Calculates PSI for each required feature column against the baseline distribution."""
    feature_psi = {}
    for col in REQUIRED_FEATURE_COLUMNS:
        if reference_df[col].dtype == object:
            feature_psi[col] = calculate_categorical_psi(reference_df[col], current_df[col])
        else:
            feature_psi[col] = calculate_numeric_psi(reference_df[col], current_df[col])
    return feature_psi


def get_actual_column(df: pd.DataFrame) -> str | None:
    for col in ["actual_response", "Response"]:
        if col in df.columns:
            return col
    return None


def calculate_supervised_metrics(df: pd.DataFrame) -> dict:
    """Returns supervised performance metrics if a ground-truth label column is present."""
    actual_col = get_actual_column(df)
    if actual_col is None:
        return {}

    y_true = df[actual_col].astype(int)
    y_prob = df["conversion_probability"].astype(float)
    y_pred = df["predicted_conversion"].astype(int)

    metrics = {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "brier_score": safe_float(brier_score_loss(y_true, y_prob)),
    }

    try:
        metrics["log_loss"] = safe_float(log_loss(y_true, y_prob))
    except ValueError:
        metrics["log_loss"] = None

    try:
        metrics["roc_auc"] = safe_float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else None
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def summarize_window(
    window_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    batch_id: str,
    psi_threshold: float,
    pred_mean_shift_threshold: float,
    baseline_pred_mean: float,
) -> tuple[dict, list[dict]]:
    """Produces a rolling-window monitoring summary and per-feature PSI drift rows for a batch."""
    feature_psi = calculate_feature_psi(baseline_df, window_df)
    max_psi_feature = max(feature_psi, key=feature_psi.get)
    max_psi_value = feature_psi[max_psi_feature]

    mean_prob = float(window_df["conversion_probability"].mean())
    predicted_positive_rate = float(window_df["predicted_conversion"].mean())
    recommended_premium_mean = float(window_df["recommended_premium"].mean())
    recommended_premium_std = float(window_df["recommended_premium"].std(ddof=0))
    null_rate_mean = float(window_df[REQUIRED_FEATURE_COLUMNS].isna().mean().mean())

    summary = {
        "batch_id": batch_id,
        "rows": int(len(window_df)),
        "window_start_batch": str(window_df["batch_id"].iloc[0]),
        "window_end_batch": str(window_df["batch_id"].iloc[-1]),
        "conversion_probability_mean": mean_prob,
        "conversion_probability_std": safe_float(window_df["conversion_probability"].std(ddof=0)),
        "conversion_probability_p10": safe_float(window_df["conversion_probability"].quantile(0.10)),
        "conversion_probability_p50": safe_float(window_df["conversion_probability"].quantile(0.50)),
        "conversion_probability_p90": safe_float(window_df["conversion_probability"].quantile(0.90)),
        "predicted_conversion_rate": predicted_positive_rate,
        "recommended_premium_mean": recommended_premium_mean,
        "recommended_premium_std": safe_float(recommended_premium_std),
        "null_rate_mean": null_rate_mean,
        "max_feature_psi": float(max_psi_value),
        "max_feature_psi_name": str(max_psi_feature),
        "prediction_mean_shift": float(mean_prob - baseline_pred_mean),
        "drift_alert": bool(max_psi_value >= psi_threshold),
        "prediction_shift_alert": bool(
            abs(mean_prob - baseline_pred_mean) >= pred_mean_shift_threshold
        ),
    }

    summary.update(calculate_supervised_metrics(window_df))

    drift_rows = []
    for feature_name, psi_value in feature_psi.items():
        drift_rows.append(
            {
                "batch_id": batch_id,
                "feature": feature_name,
                "psi": float(psi_value),
                "psi_alert": bool(psi_value >= psi_threshold),
            }
        )

    return summary, drift_rows


def write_outputs(
    output_dir: Path,
    batch_metrics_df: pd.DataFrame,
    feature_drift_df: pd.DataFrame,
    summary_payload: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_metrics_df.to_csv(output_dir / "batch_metrics.csv", index=False)
    feature_drift_df.to_csv(output_dir / "feature_drift.csv", index=False)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)


def main() -> None:
    """Runs the monitoring pipeline and writes batch metrics, feature drift, and summary to disk."""
    args = parse_args()

    predictions_df = load_predictions(args.predictions)
    baseline_df = load_baseline_features(args.baseline)

    if predictions_df.empty:
        raise ValueError("Predictions file is empty. Run live inference first.")

    unique_batches = (
        predictions_df[["batch_id", "batch_order"]]
        .drop_duplicates()
        .sort_values("batch_order", kind="stable")
    )
    batch_ids = unique_batches["batch_id"].tolist()

    first_batch_id = batch_ids[0]
    baseline_pred_mean = float(
        predictions_df.loc[
            predictions_df["batch_id"] == first_batch_id, "conversion_probability"
        ].mean()
    )

    batch_metrics_rows = []
    feature_drift_rows = []

    for idx, batch_id in enumerate(batch_ids):
        start_idx = max(0, idx + 1 - args.rolling_window_batches)
        window_batch_ids = batch_ids[start_idx : idx + 1]
        window_df = predictions_df[predictions_df["batch_id"].isin(window_batch_ids)].copy()

        summary, drift_rows = summarize_window(
            window_df=window_df,
            baseline_df=baseline_df,
            batch_id=batch_id,
            psi_threshold=args.psi_threshold,
            pred_mean_shift_threshold=args.pred_mean_shift_threshold,
            baseline_pred_mean=baseline_pred_mean,
        )
        batch_metrics_rows.append(summary)
        feature_drift_rows.extend(drift_rows)

    batch_metrics_df = pd.DataFrame(batch_metrics_rows)
    feature_drift_df = pd.DataFrame(feature_drift_rows)
    latest = batch_metrics_df.iloc[-1].to_dict()

    summary_payload = {
        "generated_at_utc": utc_now_iso(),
        "predictions_path": args.predictions,
        "baseline_path": args.baseline,
        "n_prediction_rows": int(len(predictions_df)),
        "n_batches": int(len(batch_ids)),
        "rolling_window_batches": int(args.rolling_window_batches),
        "psi_threshold": float(args.psi_threshold),
        "pred_mean_shift_threshold": float(args.pred_mean_shift_threshold),
        "latest_batch_summary": latest,
        "has_supervised_metrics": bool(get_actual_column(predictions_df) is not None),
    }

    write_outputs(
        output_dir=Path(args.output_dir),
        batch_metrics_df=batch_metrics_df,
        feature_drift_df=feature_drift_df,
        summary_payload=summary_payload,
    )

    print(f"Saved monitoring outputs to {args.output_dir}")
    print(f"Latest batch: {latest['batch_id']}")
    print(
        f"Latest max PSI: {latest['max_feature_psi']:.4f} ({latest['max_feature_psi_name']})"
    )
    print(f"Latest drift alert: {latest['drift_alert']}")
    print(f"Latest prediction shift alert: {latest['prediction_shift_alert']}")


if __name__ == "__main__":
    main()