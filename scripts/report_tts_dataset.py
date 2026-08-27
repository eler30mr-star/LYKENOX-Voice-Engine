"""Write the current neural TTS dataset readiness report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.tts_dataset_report import build_tts_dataset_report  # noqa: E402


def main() -> None:
    """Build the report and print it as JSON."""

    print(json.dumps(build_tts_dataset_report(ROOT), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
