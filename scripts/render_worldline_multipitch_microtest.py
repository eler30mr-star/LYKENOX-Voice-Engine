"""Render monopitch vs multipitch WORLDLINE-R microtest for baila conmigo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.multipitch_microtest import render_baila_conmigo_microtest  # noqa: E402


def main() -> None:
    """Render both paths only when real multipitch microtest samples exist."""

    payload = render_baila_conmigo_microtest(ROOT)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload.get("missing_multipitch_aliases"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
