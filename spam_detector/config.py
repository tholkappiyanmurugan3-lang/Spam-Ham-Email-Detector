"""
spam_detector.config
====================
Centralised configuration loader.

Loads config.yaml (path set via CONFIG_PATH env var) and merges it
with environment variables from .env.  All other modules import the
singleton ``cfg`` object instead of reading files themselves.

Trade-off: a single config object creates a global dependency, but for
a project of this size it avoids the overhead of a DI framework.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Load .env file (safe even if missing) and Streamlit secrets
# ---------------------------------------------------------------------------
load_dotenv()
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)) and str(k) not in os.environ:
                os.environ[str(k)] = str(v)
except Exception:
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


class _Config:
    """
    Singleton wrapper around the YAML config dict.

    Attributes are accessed via dot-notation helpers for readability.
    """

    def __init__(self) -> None:
        config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
        self._data: dict[str, Any] = _load_yaml(config_path)
        logger.info(f"Config loaded from {config_path.resolve()}")

    # ------------------------------------------------------------------
    # Generic accessor
    # ------------------------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        """Retrieve a nested value by dot-path keys."""
        return _deep_get(self._data, *keys, default=default)

    # ------------------------------------------------------------------
    # Convenience properties (avoids magic strings elsewhere)
    # ------------------------------------------------------------------
    @property
    def seed(self) -> int:
        return self.get("project", "seed", default=42)

    @property
    def model_name(self) -> str:
        return self.get("model", "base_model", default="distilbert-base-uncased")

    @property
    def hf_repo(self) -> str:
        return os.getenv("HF_MODEL_REPO", self.get("model", "hf_repo", default=""))

    @property
    def num_labels(self) -> int:
        return self.get("model", "num_labels", default=2)

    @property
    def max_length(self) -> int:
        return self.get("model", "max_length", default=256)

    @property
    def id2label(self) -> dict[int, str]:
        raw = self.get("model", "id2label", default={0: "HAM", 1: "SPAM"})
        return {int(k): v for k, v in raw.items()}

    @property
    def label2id(self) -> dict[str, int]:
        return self.get("model", "label2id", default={"HAM": 0, "SPAM": 1})

    @property
    def model_dir(self) -> Path:
        return Path(self.get("paths", "model_dir", default="models/distilbert_spam"))

    @property
    def raw_data_dir(self) -> Path:
        return Path(self.get("paths", "raw_data", default="data/raw"))

    @property
    def processed_data_dir(self) -> Path:
        return Path(self.get("paths", "processed_data", default="data/processed"))

    @property
    def db_path(self) -> Path:
        return Path(self.get("paths", "db_path", default="data/predictions.db"))

    @property
    def logs_dir(self) -> Path:
        return Path(self.get("paths", "logs_dir", default="logs"))

    @property
    def training(self) -> dict[str, Any]:
        return self.get("training", default={})

    @property
    def preprocessing(self) -> dict[str, Any]:
        return self.get("preprocessing", default={})

    @property
    def shap_cfg(self) -> dict[str, Any]:
        return self.get("shap", default={})

    @property
    def streamlit_cfg(self) -> dict[str, Any]:
        return self.get("streamlit", default={})

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO").upper()


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
cfg = _Config()
