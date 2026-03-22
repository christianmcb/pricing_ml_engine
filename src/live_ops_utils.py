import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return load_json(path)


def extract_batch_order(batch_id: str) -> int:
    import re

    match = re.search(r"(\d+)$", str(batch_id))
    if match:
        return int(match.group(1))
    return 10**9