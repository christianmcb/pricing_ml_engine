from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    with path.open("r") as f:
        return yaml.safe_load(f)