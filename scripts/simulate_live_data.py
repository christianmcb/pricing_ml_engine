import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.data_processing import (
    REQUIRED_FEATURE_COLUMNS,
    load_test_data,
    validate_inference_dataframe,
)
from src.live_ops_utils import save_json, utc_now_iso


INTEGER_COLUMNS = {"Age", "Driving_License", "Previously_Insured", "Vintage"}
LOW_CARDINALITY_NUMERIC = {"Region_Code", "Policy_Sales_Channel"}


def _normalize_probs(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = values.value_counts(dropna=False)
    categories = counts.index.to_numpy()
    probs = (counts / counts.sum()).to_numpy(dtype=float)
    return categories, probs


def _build_profile(reference_df: pd.DataFrame) -> dict:
    profile = {
        "categorical": {},
        "discrete_numeric": {},
        "continuous": {},
        "ranges": {},
    }

    for col in REQUIRED_FEATURE_COLUMNS:
        series = reference_df[col]

        if series.dtype == object:
            categories, probs = _normalize_probs(series.astype(str))
            profile["categorical"][col] = {
                "categories": categories.tolist(),
                "probs": probs.tolist(),
            }
            continue

        if col in LOW_CARDINALITY_NUMERIC:
            categories, probs = _normalize_probs(series)
            profile["discrete_numeric"][col] = {
                "values": categories.tolist(),
                "probs": probs.tolist(),
            }
            continue

        col_min = float(series.min())
        col_max = float(series.max())
        profile["ranges"][col] = {"min": col_min, "max": col_max}

        if col == "Annual_Premium":
            logged = np.log1p(series.astype(float).clip(lower=0.0))
            profile["continuous"][col] = {
                "kind": "lognormal",
                "mu": float(logged.mean()),
                "sigma": max(float(logged.std(ddof=0)), 1e-6),
            }
        else:
            values = series.astype(float).to_numpy()
            profile["continuous"][col] = {
                "kind": "normal",
                "mu": float(values.mean()),
                "sigma": max(float(values.std(ddof=0)), 1e-6),
            }

    return profile


def _sample_from_profile(profile: dict, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    generated = {}

    for col, info in profile["categorical"].items():
        generated[col] = rng.choice(info["categories"], size=n_rows, p=info["probs"])

    for col, info in profile["discrete_numeric"].items():
        generated[col] = rng.choice(info["values"], size=n_rows, p=info["probs"])

    for col, info in profile["continuous"].items():
        kind = info["kind"]
        if kind == "lognormal":
            sampled = np.expm1(rng.normal(info["mu"], info["sigma"], size=n_rows))
        else:
            sampled = rng.normal(info["mu"], info["sigma"], size=n_rows)

        min_val = profile["ranges"][col]["min"]
        max_val = profile["ranges"][col]["max"]
        sampled = np.clip(sampled, min_val, max_val)

        if col in INTEGER_COLUMNS:
            sampled = np.rint(sampled).astype(int)
        elif col == "Annual_Premium":
            sampled = np.round(sampled, 2)

        generated[col] = sampled

    df = pd.DataFrame(generated)
    return df[REQUIRED_FEATURE_COLUMNS]


def _apply_drift(
    df: pd.DataFrame,
    *,
    mode: str,
    feature: str,
    strength: float,
    batch_idx: int,
    n_batches: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if mode == "none":
        return df

    factor = strength
    if mode == "trend":
        factor = strength * (batch_idx / max(1, n_batches - 1))
    elif mode == "sudden" and batch_idx < n_batches // 2:
        return df

    shifted = df.copy()

    if feature == "Annual_Premium":
        shifted[feature] = np.round(shifted[feature].astype(float) * (1.0 + factor), 2)
    elif feature == "Age":
        shifted[feature] = np.rint(shifted[feature].astype(float) + (8.0 * factor)).astype(
            int
        )
    elif feature == "Vintage":
        shifted[feature] = np.rint(
            shifted[feature].astype(float) + (40.0 * factor)
        ).astype(int)
    elif feature in {"Region_Code", "Policy_Sales_Channel"}:
        shifted[feature] = shifted[feature].astype(float) + (5.0 * factor)
    elif feature == "Vehicle_Damage":
        p_yes = min(max(0.15 + (0.7 * factor), 0.01), 0.95)
        mask = rng.random(len(shifted)) < p_yes
        shifted.loc[mask, feature] = "Yes"
        shifted.loc[~mask, feature] = "No"
    elif feature == "Previously_Insured":
        p_one = min(max(0.5 - (0.4 * factor), 0.01), 0.99)
        shifted[feature] = (rng.random(len(shifted)) < p_one).astype(int)

    return shifted


def _write_metadata(path: Path, payload: dict) -> None:
    save_json(path, payload)


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Generate synthetic live inference batches based on reference data distributions."
    )
    parser.add_argument(
        "--reference-path",
        default=config["data"]["test_path"],
        help="Reference CSV used only to estimate feature distributions.",
    )
    parser.add_argument("--out-dir", default="data/live")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=config["data"]["random_state"])
    parser.add_argument(
        "--drift-mode",
        choices=["none", "trend", "sudden"],
        default="none",
    )
    parser.add_argument(
        "--drift-feature",
        choices=REQUIRED_FEATURE_COLUMNS,
        default="Annual_Premium",
    )
    parser.add_argument("--drift-strength", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_df = load_test_data(args.reference_path)
    reference_df = validate_inference_dataframe(reference_df)
    profile = _build_profile(reference_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    generated_files = []

    for batch_idx in range(args.n_batches):
        batch_df = _sample_from_profile(profile, args.batch_size, rng)
        batch_df = _apply_drift(
            batch_df,
            mode=args.drift_mode,
            feature=args.drift_feature,
            strength=args.drift_strength,
            batch_idx=batch_idx,
            n_batches=args.n_batches,
            rng=rng,
        )

        batch_df = validate_inference_dataframe(batch_df)
        batch_file = out_dir / f"batch_{batch_idx:04d}.csv"
        batch_df.to_csv(batch_file, index=False)

        generated_files.append(
            {
                "batch_idx": batch_idx,
                "path": str(batch_file),
                "rows": len(batch_df),
            }
        )

    metadata = {
        "generated_at_utc": utc_now_iso(),
        "reference_path": args.reference_path,
        "out_dir": str(out_dir),
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "seed": args.seed,
        "drift_mode": args.drift_mode,
        "drift_feature": args.drift_feature,
        "drift_strength": args.drift_strength,
        "generated_files": generated_files,
    }
    _write_metadata(out_dir / "generation_metadata.json", metadata)

    print(f"Generated {args.n_batches} synthetic live batches in {out_dir}")
    print(f"Metadata: {out_dir / 'generation_metadata.json'}")


if __name__ == "__main__":
    main()
