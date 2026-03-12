import pandas as pd


REQUIRED_FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]


def load_train_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def load_test_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def validate_training_dataframe(df: pd.DataFrame, target_col: str = "Response") -> None:
    missing_features = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Training data missing required feature columns: {missing_features}")

    if target_col not in df.columns:
        raise ValueError(f"Training data missing target column: {target_col}")


def validate_inference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    missing_features = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Inference data missing required feature columns: {missing_features}")

    extra_columns = [col for col in df.columns if col not in REQUIRED_FEATURE_COLUMNS]
    if extra_columns:
        raise ValueError(f"Inference data has unexpected columns: {extra_columns}")

    return df[REQUIRED_FEATURE_COLUMNS]


def split_features_target(df: pd.DataFrame, target_col: str = "Response"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y