from pathlib import Path

import pytest

from src.model_registry import build_versioned_paths, get_model_path, latest_run_id


def test_build_versioned_paths_creates_run_folder(tmp_path):
    base_model_path = tmp_path / "models" / "current" / "model.joblib"
    paths = build_versioned_paths(str(base_model_path), "20260320T120000Z")

    assert paths["run_dir"].exists()
    assert paths["model_path"].parent.name == "20260320T120000Z"
    assert paths["run_dir"].parent.name == "registry"


def test_get_model_path_prefers_current_model(tmp_path):
    current_model = tmp_path / "models" / "current" / "model.joblib"
    current_model.parent.mkdir(parents=True, exist_ok=True)
    current_model.write_text("stub")

    resolved = get_model_path(str(current_model))
    assert resolved == current_model


def test_get_model_path_falls_back_to_latest_registry(tmp_path):
    base_model_path = tmp_path / "models" / "current" / "model.joblib"

    older = tmp_path / "models" / "registry" / "20260320T110000Z" / "model.joblib"
    newer = tmp_path / "models" / "registry" / "20260320T130000Z" / "model.joblib"
    older.parent.mkdir(parents=True, exist_ok=True)
    newer.parent.mkdir(parents=True, exist_ok=True)
    older.write_text("old")
    newer.write_text("new")

    resolved = get_model_path(str(base_model_path), prefer_current=False)
    assert resolved == newer
    assert latest_run_id(str(base_model_path)) == "20260320T130000Z"


def test_get_model_path_with_explicit_run_id(tmp_path):
    base_model_path = tmp_path / "models" / "current" / "model.joblib"
    target = tmp_path / "models" / "registry" / "20260320T150000Z" / "model.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("run-model")

    resolved = get_model_path(str(base_model_path), run_id="20260320T150000Z")
    assert resolved == target


def test_get_model_path_raises_on_missing_run(tmp_path):
    base_model_path = tmp_path / "models" / "current" / "model.joblib"

    with pytest.raises(FileNotFoundError):
        get_model_path(str(base_model_path), run_id="does-not-exist")