import warnings
warnings.filterwarnings("ignore")

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ParameterSampler,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from src.config import load_config
from src.data_processing import (
    load_train_data,
    split_features_target,
    validate_training_dataframe,
)
from src.evaluate_model import (
    evaluate_classifier,
    log_experiment,
    save_json,
    save_results_csv,
)
from src.feature_engineering import build_preprocessor
from src.logger import get_logger


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def make_run_id() -> str:
    """Create a UTC timestamp-based run id."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_artifact_root(base_model_path: str) -> Path:
    """
    Resolve the model artifact root directory.

    Examples:
    - models/model.joblib         -> models
    - models/current/model.joblib -> models
    - outputs/models/model.joblib -> outputs/models
    """
    base_path = Path(base_model_path)
    parent = base_path.parent

    if parent.name == "current":
        return parent.parent

    return parent


def build_versioned_paths(base_model_path: str, run_id: str) -> dict:
    """
    Build run-specific artifact paths under:
    <artifact_root>/registry/<run_id>/
    """
    base_path = Path(base_model_path)
    artifact_root = resolve_artifact_root(base_model_path)

    stem = base_path.stem
    suffix = base_path.suffix or ".joblib"

    run_dir = artifact_root / "registry" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "artifact_root": artifact_root,
        "run_dir": run_dir,
        "model_path": run_dir / f"{stem}{suffix}",
        "comparison_path": run_dir / "model_comparison.csv",
        "params_path": run_dir / "best_params.json",
        "metadata_path": run_dir / "model_metadata.json",
        "feature_importance_path": run_dir / "feature_importance.csv",
    }


def build_model_pipelines(preprocessor, random_state: int):
    return {
        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            )),
        ]),
        "LightGBM": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
    }


def get_param_distributions():
    return {
        "RandomForest": {
            "classifier__n_estimators": randint(200, 400),
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": randint(2, 8),
            "classifier__min_samples_leaf": randint(1, 4),
        },
        "XGBoost": {
            "classifier__n_estimators": randint(200, 400),
            "classifier__max_depth": randint(3, 8),
            "classifier__learning_rate": uniform(0.03, 0.07),
            "classifier__subsample": uniform(0.7, 0.3),
            "classifier__colsample_bytree": uniform(0.7, 0.3),
        },
        "LightGBM": {
            "classifier__n_estimators": randint(200, 400),
            "classifier__num_leaves": randint(20, 80),
            "classifier__max_depth": [-1, 5, 10, 15],
            "classifier__learning_rate": uniform(0.03, 0.07),
            "classifier__subsample": uniform(0.7, 0.3),
            "classifier__colsample_bytree": uniform(0.7, 0.3),
        },
    }


def tune_model_with_progress(
    logger,
    model_name,
    pipeline,
    param_dist,
    X_train,
    y_train,
    X_test,
    y_test,
    cv,
    scoring,
    n_iter,
    random_state,
):
    sampled_params = list(ParameterSampler(
        param_dist,
        n_iter=n_iter,
        random_state=random_state,
    ))

    best_score = -np.inf
    best_params = None

    logger.info("Starting tuning for %s with %d trials", model_name, len(sampled_params))
    logger.info("Using models: RandomForest | XGBoost | LightGBM")
    logger.info("Hyperparameter search iterations: %d", n_iter)

    progress_bar = tqdm(
        sampled_params,
        desc=f"Tuning {model_name}",
        leave=True,
        dynamic_ncols=True,
    )

    for trial_idx, params in enumerate(progress_bar, start=1):
        candidate = clone(pipeline)
        candidate.set_params(**params)

        cv_scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )

        mean_score = np.mean(cv_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            logger.info(
                "%s new best %s at trial %d: %.4f",
                model_name,
                scoring,
                trial_idx,
                best_score,
            )

        progress_bar.set_postfix({
            "best": f"{best_score:.4f}",
            "trial": f"{mean_score:.4f}",
        })

    if best_params is None:
        raise ValueError(f"No valid hyperparameter set found for {model_name}")

    best_model = clone(pipeline)
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    eval_results = evaluate_classifier(best_model, X_test, y_test)

    logger.info(
        "Finished %s | best_cv=%.4f | test_auc=%.4f",
        model_name,
        best_score,
        eval_results["test_auc"],
    )

    return {
        "model_name": model_name,
        "best_model": best_model,
        "best_params": best_params,
        "best_cv_auc": best_score,
        "test_auc": eval_results["test_auc"],
    }


def main():
    config = load_config()
    logger = get_logger(__name__, config["artifacts"]["log_path"])

    logger.info("Training pipeline started")

    train_path = config["data"]["train_path"]
    target_column = config["data"]["target_column"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    cv_folds = config["training"]["cv_folds"]
    n_iter = config["training"]["n_iter"]
    scoring = config["training"]["scoring"]

    model_path = config["artifacts"]["model_path"]

    df = load_train_data(train_path)
    validate_training_dataframe(df, target_col=target_column)
    X, y = split_features_target(df, target_col=target_column)

    logger.info("Loaded training data from %s with shape %s", train_path, df.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()
    pipelines = build_model_pipelines(preprocessor, random_state=random_state)
    param_distributions = get_param_distributions()

    best_models = {}
    best_params_by_model = {}
    summary_rows = []

    run_id = make_run_id()
    versioned_paths = build_versioned_paths(model_path, run_id)

    logger.info("Run id: %s", run_id)
    logger.info("Artifact root: %s", versioned_paths["artifact_root"])
    logger.info("Versioned artifact directory: %s", versioned_paths["run_dir"])

    git_hash = get_git_hash()

    for model_name, pipeline in pipelines.items():
        result = tune_model_with_progress(
            logger=logger,
            model_name=model_name,
            pipeline=pipeline,
            param_dist=param_distributions[model_name],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            random_state=random_state,
        )

        best_models[model_name] = result["best_model"]
        best_params_by_model[model_name] = result["best_params"]

        experiment_row = {
            "run_id": run_id,
            "model": model_name,
            "best_cv_auc": float(result["best_cv_auc"]),
            "test_auc": float(result["test_auc"]),
            "best_params": json.dumps(result["best_params"], default=str),
            "git_hash": git_hash,
        }

        summary_rows.append(experiment_row)
        log_experiment(experiment_row)

    results_df = pd.DataFrame(summary_rows).sort_values(by="test_auc", ascending=False)

    best_model_name = results_df.iloc[0]["model"]
    best_model = best_models[best_model_name]
    best_params = best_params_by_model[best_model_name]

    joblib.dump(best_model, versioned_paths["model_path"])

    save_results_csv(results_df, versioned_paths["comparison_path"])
    save_json(
        {
            "run_id": run_id,
            "best_model": best_model_name,
            "best_params": best_params,
        },
        versioned_paths["params_path"],
    )

    logger.info("Best model selected: %s", best_model_name)
    logger.info("Saved versioned model to %s", versioned_paths["model_path"])

    print("\n=== Model Comparison ===")
    print(results_df[["model", "best_cv_auc", "test_auc"]])
    print(f"\nBest overall model: {best_model_name}")
    print(f"Run id: {run_id}")

    if hasattr(best_model.named_steps["classifier"], "feature_importances_"):
        feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
        importances = best_model.named_steps["classifier"].feature_importances_

        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        fi.to_csv(versioned_paths["feature_importance_path"], index=False)

    metadata = {
        "run_id": run_id,
        "model_name": best_model_name,
        "git_hash": git_hash,
        "best_cv_auc": float(results_df.iloc[0]["best_cv_auc"]),
        "test_auc": float(results_df.iloc[0]["test_auc"]),
        "artifact_root": str(versioned_paths["artifact_root"]),
        "artifact_dir": str(versioned_paths["run_dir"]),
        "model_path": str(versioned_paths["model_path"]),
        "comparison_path": str(versioned_paths["comparison_path"]),
        "params_path": str(versioned_paths["params_path"]),
    }

    with open(versioned_paths["metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training pipeline finished")
    logger.info("Candidate model stored in registry run folder: %s", versioned_paths["run_dir"])
    logger.info("Promote explicitly via Makefile when ready")

if __name__ == "__main__":
    main()