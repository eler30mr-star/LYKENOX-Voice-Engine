"""Run the native LYKENOX Voice Engine desktop app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.ui.main_window import run_app


if __name__ == "__main__":
    raise SystemExit(run_app(ROOT))
