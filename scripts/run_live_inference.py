import argparse
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import joblib
import pandas as pd

from src.config import load_config
from src.data_processing import validate_inference_dataframe
from src.live_ops_utils import load_json, load_json_or_default, save_json, utc_now_iso


def load_state(state_path: Path) -> dict:
    return load_json_or_default(state_path, {"processed_batches": []})


def save_state(state_path: Path, state: dict) -> None:
    save_json(state_path, state)


def load_model_run_id(model_path: str) -> str:
    """Reads the run_id from model_metadata.json next to the model. Returns 'unknown' on failure."""
    model_path_obj = Path(model_path)
    candidates = [
        model_path_obj.with_name("model_metadata.json"),
        PROJECT_ROOT / "models" / "current" / "model_metadata.json",
    ]
    for metadata_path in candidates:
        if metadata_path.exists():
            try:
                meta = load_json(metadata_path)
                return str(meta.get("run_id", "unknown"))
            except Exception:
                return "unknown"
    return "unknown"


def score_batch(
    model,
    batch_df: pd.DataFrame,
    batch_file: Path,
    base_premium: float,
    demand_multiplier: float,
    model_run_id: str,
    model_path: str,
) -> pd.DataFrame:
    """Scores a batch CSV and returns a DataFrame with predictions and batch metadata attached."""
    inference_df = validate_inference_dataframe(batch_df)

    conversion_probability = model.predict_proba(inference_df)[:, 1]
    predicted_conversion = model.predict(inference_df)
    demand_adjustment = conversion_probability * demand_multiplier
    recommended_premium = base_premium + demand_adjustment

    output_df = inference_df.copy()
    output_df["conversion_probability"] = conversion_probability
    output_df["predicted_conversion"] = predicted_conversion
    output_df["demand_adjustment"] = demand_adjustment.round(2)
    output_df["recommended_premium"] = recommended_premium.round(2)

    output_df["batch_id"] = batch_file.stem
    output_df["batch_file"] = str(batch_file)
    output_df["inference_timestamp_utc"] = utc_now_iso()
    output_df["model_run_id"] = model_run_id
    output_df["model_path"] = model_path

    return output_df


def append_predictions(output_path: Path, scored_df: pd.DataFrame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    scored_df.to_csv(output_path, mode="a", header=write_header, index=False)


def list_batch_files(live_dir: Path, pattern: str) -> list[Path]:
    return sorted([p for p in live_dir.glob(pattern) if p.is_file()])


def parse_args() -> argparse.Namespace:
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Run live inference on incoming batch CSV files."
    )
    parser.add_argument("--live-dir", default="data/live")
    parser.add_argument("--pattern", default="batch_*.csv")
    parser.add_argument("--model", default=config["artifacts"]["model_path"])
    parser.add_argument("--output", default=config["artifacts"]["predictions_path"])
    parser.add_argument("--state-file", default="outputs/live_inference_state.json")
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    parser.add_argument("--loop", action="store_true", help="Keep polling for new files.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after processing this many new batches.",
    )
    return parser.parse_args()


def main() -> None:
    """Processes new batch CSV files in the live directory and appends scored predictions to output."""
    args = parse_args()

    live_dir = Path(args.live_dir)
    output_path = Path(args.output)
    state_path = Path(args.state_file)

    if not live_dir.exists():
        raise FileNotFoundError(f"Live directory does not exist: {live_dir}")
    
    model = joblib.load(args.model)
    model_run_id = load_model_run_id(args.model)

    config = load_config()
    base_premium = float(config["pricing"]["base_premium"])
    demand_multiplier = float(config["pricing"]["demand_multiplier"])

    state = load_state(state_path)
    processed = set(state.get("processed_batches", []))
    processed_count = 0

    while True:
        all_batches = list_batch_files(live_dir, args.pattern)
        new_batches = [p for p in all_batches if p.name not in processed]

        if not new_batches:
            if not args.loop:
                print("No new batches found. Exiting.")
                break
            if args.poll_seconds > 0:
                time.sleep(args.poll_seconds)
            continue
        
        for batch_file in new_batches:
            batch_df = pd.read_csv(batch_file)
            
            scored_df = score_batch(
                model=model,
                batch_df=batch_df,
                batch_file=batch_file,
                base_premium=base_premium,
                demand_multiplier=demand_multiplier,
                model_run_id=model_run_id,
                model_path=args.model,
            )
            append_predictions(output_path, scored_df)

            processed.add(batch_file.name)
            state["processed_batches"] = sorted(processed)
            state["last_processed_at_utc"] = utc_now_iso()
            save_state(state_path, state)

            processed_count += 1
            print(
                f"Processed {batch_file.name} | rows={len(scored_df)} | "
                f"output={output_path}"
            )

            if args.max_batches is not None and processed_count >= args.max_batches:
                print(f"Reached --max-batches={args.max_batches}. Exiting.")
                return
        
        if not args.loop:
            break

        if args.poll_seconds > 0:
            time.sleep(args.poll_seconds)



if __name__ == "__main__":
    main()