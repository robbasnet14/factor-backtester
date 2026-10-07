"""Load the YAML config.

Relative paths in a config (`data.cache_dir`, `output_dir`, `plugin_dirs`) are relative to
the config file, not to wherever the command is run from, so a run reads
and writes the same places from any directory. Absolute paths, and paths
starting with `~`, are used as given.
"""

from pathlib import Path

import yaml

DEFAULT_OUTPUT_DIR = "outputs"


def load_config(path: str | Path = "config.yaml") -> dict:
    config_path = Path(path).expanduser().resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = config_path.parent
    cfg["data"]["cache_dir"] = str(_resolve(cfg["data"]["cache_dir"], base))
    cfg["output_dir"] = str(_resolve(cfg.get("output_dir", DEFAULT_OUTPUT_DIR), base))
    cfg["plugin_dirs"] = [str(_resolve(d, base)) for d in cfg.get("plugin_dirs") or []]
    return cfg


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path
