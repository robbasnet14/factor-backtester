"""Load and validate the YAML config."""
from pathlib import Path
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
