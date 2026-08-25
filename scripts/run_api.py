"""Run the local FastAPI service."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.config.settings import load_settings


if __name__ == "__main__":
    settings = load_settings(ROOT)
    uvicorn.run("lykenox_voice_engine.api.server:app", host=settings.api_host, port=settings.api_port)
