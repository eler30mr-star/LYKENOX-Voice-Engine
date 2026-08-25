"""Application settings for the local voice engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    """Resolved app settings loaded from JSON config."""

    api_host: str
    api_port: int
    models_dir: Path
    profiles_dir: Path
    datasets_dir: Path
    outputs_dir: Path
    device: str


def load_settings(root: Path) -> AppSettings:
    """Load app settings from config/app_settings.json relative to the project root."""

    data = json.loads((root / "config" / "app_settings.json").read_text(encoding="utf-8"))
    return AppSettings(
        api_host=data["api_host"],
        api_port=int(data["api_port"]),
        models_dir=root / data["models_dir"],
        profiles_dir=root / data["profiles_dir"],
        datasets_dir=root / data["datasets_dir"],
        outputs_dir=root / data["outputs_dir"],
        device=data.get("device", "cpu"),
    )
