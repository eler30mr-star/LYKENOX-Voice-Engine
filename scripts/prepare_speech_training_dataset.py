"""Prepare accepted LYKENOX speech recordings for training review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.identity_dataset_preparer import IdentityDatasetPreparer  # noqa: E402


def main() -> None:
    """Build train/validation manifests and quality reports."""

    report = IdentityDatasetPreparer(ROOT).prepare_speech()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
