"""
Configuration loader helper reading configs/config.yaml.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path = Path("configs/config.yaml")) -> Dict[str, Any]:
    """
    Loads YAML configuration file safely.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
