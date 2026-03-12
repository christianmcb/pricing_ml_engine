from pathlib import Path
import argparse
import joblib
import pandas as pd

from src.config import load_config
from src.data_processing import validate_inference_dataframe


def predict_from_csv(
    input_path: str,
    output_path: str,
    model_path: str,
    base_premium: float,
    demand_multiplier: float,
):
    model = joblib.load(model_path)
    df = pd.read_csv(input_path)
    df = validate_inference_dataframe(df)

    conversion_probability = model.predict_proba(df)[:, 1]
    predicted_conversion = model.predict(df)

    demand_adjustment = conversion_probability * demand_multiplier
    recommended_premium = base_premium + demand_adjustment

    output_df = df.copy()
    output_df["conversion_probability"] = conversion_probability
    output_df["predicted_conversion"] = predicted_conversion
    output_df["demand_adjustment"] = demand_adjustment.round(2)
    output_df["recommended_premium"] = recommended_premium.round(2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(output_df.head())


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Run batch pricing predictions.")
    parser.add_argument("--input", default=config["data"]["test_path"])
    parser.add_argument("--output", default=config["artifacts"]["predictions_path"])
    parser.add_argument("--model", default=config["artifacts"]["model_path"])
    args = parser.parse_args()

    predict_from_csv(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        base_premium=config["pricing"]["base_premium"],
        demand_multiplier=config["pricing"]["demand_multiplier"],
    )


if __name__ == "__main__":
    main()
