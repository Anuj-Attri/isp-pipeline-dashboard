from typing import Any, Dict, Optional

import os

import yaml


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "config",
    "default_config.yaml",
)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = path or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "default_config.yaml")
    )
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return data

