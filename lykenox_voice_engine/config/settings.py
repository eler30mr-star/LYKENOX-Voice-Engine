"""Persistent app settings."""

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "lykenox_voice_engine" / "config" / "app_settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "api_host": "127.0.0.1",
    "api_port": 8765,
    "models_dir": "models",
    "profiles_dir": "profiles",
    "datasets_dir": "datasets",
    "outputs_dir": "outputs",
    "device": "cpu",
}


def load_settings() -> dict[str, Any]:
    """Load settings merged with safe local defaults."""
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        values = {}
    settings = dict(DEFAULT_SETTINGS)
    settings.update({k: v for k, v in values.items() if v is not None})
    return settings
