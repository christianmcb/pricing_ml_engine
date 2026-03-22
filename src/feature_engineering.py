from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "Gender",
    "Region_Code",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Policy_Sales_Channel",
]

NUMERIC_FEATURES = [
    "Age",
    "Driving_License",
    "Previously_Insured",
    "Annual_Premium",
    "Vintage",
]


@contextmanager
def suppress_lightgbm_warnings():
    """Temporarily suppresses LightGBM warning logs."""
    try:
        import lightgbm.basic as lgb_basic
    except Exception:
        yield
        return

    original_log_warning = getattr(lgb_basic, "_log_warning", None)

    if original_log_warning is None:
        yield
        return

    try:
        lgb_basic._log_warning = lambda *_args, **_kwargs: None
        yield
    finally:
        lgb_basic._log_warning = original_log_warning


def build_preprocessor() -> ColumnTransformer:
    """Builds a ColumnTransformer with scaling for numeric and one-hot encoding for categorical features."""
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_preprocessor_for_dataframe(df: pd.DataFrame) -> ColumnTransformer:
    """Builds a ColumnTransformer for all columns in a DataFrame."""
    categorical_features = []
    numeric_features = []

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical_features.append(col)
        else:
            numeric_features.append(col)

    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("Cannot build preprocessor for empty DataFrame")

    return ColumnTransformer(transformers)


def maybe_apply_openfe(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Applies OpenFE when enabled in config and returns transformed train/test sets."""
    training_cfg = config.get("training", {})
    openfe_enabled = bool(training_cfg.get("openfe", False))

    if not openfe_enabled:
        return X_train, X_test, {"enabled": False, "generated_features": 0}

    requested_n_jobs = int(training_cfg.get("openfe_n_jobs", -1))
    cpu_count = os.cpu_count() or 1
    if requested_n_jobs <= 0:
        # scikit-learn style: -1 means all cores, -2 all but one, etc.
        n_jobs = max(1, cpu_count + 1 + requested_n_jobs)
    else:
        n_jobs = requested_n_jobs

    top_features = training_cfg.get("openfe_top_features", None)
    if top_features is not None:
        top_features = int(top_features)

    try:
        from openfe import OpenFE, transform
    except ImportError as exc:
        raise ImportError(
            "OpenFE is enabled in config, but package is not installed. Run: pip install openfe"
        ) from exc

    ofe = OpenFE()
    with suppress_lightgbm_warnings():
        features = ofe.fit(data=X_train, label=y_train, n_jobs=n_jobs)
    if top_features is not None:
        features = features[:top_features]

    X_train_fe, X_test_fe = transform(X_train, X_test, features, n_jobs=n_jobs)
    return X_train_fe, X_test_fe, {
        "enabled": True,
        "generated_features": len(features),
        "requested_n_jobs": requested_n_jobs,
        "n_jobs": n_jobs,
        "top_features": top_features,
    }
