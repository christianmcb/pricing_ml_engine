from datetime import datetime, timezone
import json
from pathlib import Path


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
    artifact_root = resolve_artifact_root(base_model_path)
    run_dir = artifact_root / "registry" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "artifact_root": artifact_root,
        "run_dir": run_dir,
        "model_path": run_dir / "model.joblib",
        "comparison_path": run_dir / "model_comparison.csv",
        "params_path": run_dir / "best_params.json",
        "metadata_path": run_dir / "model_metadata.json",
        "feature_importance_path": run_dir / "feature_importance.csv",
    }


def list_run_ids(base_model_path: str) -> list[str]:
    """Return sorted run IDs available under the model registry."""
    registry_dir = resolve_artifact_root(base_model_path) / "registry"
    if not registry_dir.exists():
        return []

    return sorted([entry.name for entry in registry_dir.iterdir() if entry.is_dir()])


def latest_run_id(base_model_path: str) -> str | None:
    """Return the latest run id from the registry, or None if empty."""
    run_ids = list_run_ids(base_model_path)
    if not run_ids:
        return None
    return run_ids[-1]


def get_model_path(
    base_model_path: str,
    run_id: str | None = None,
    prefer_current: bool = True,
) -> Path:
    """
    Resolve a model artifact path.

    Priority:
    1) explicit run id (registry)
    2) current promoted model (if prefer_current)
    3) latest registry run model
    """
    artifact_root = resolve_artifact_root(base_model_path)
    current_model = Path(base_model_path)

    if run_id:
        model_path = artifact_root / "registry" / run_id / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model run not found: {run_id}")
        return model_path

    if prefer_current and current_model.exists():
        return current_model

    latest = latest_run_id(base_model_path)
    if latest is None:
        raise FileNotFoundError(
            "No model found in current or registry. Train and promote a model first."
        )

    latest_model = artifact_root / "registry" / latest / "model.joblib"
    if not latest_model.exists():
        raise FileNotFoundError(f"Latest registry run is missing model file: {latest}")

    return latest_model


def get_model_metadata(base_model_path: str, model_path: Path) -> dict:
    """Load model metadata JSON adjacent to the selected model if it exists."""
    metadata_path = model_path.parent / "model_metadata.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r") as f:
        return json.load(f)