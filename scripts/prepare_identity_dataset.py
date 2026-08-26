"""Initialize the LYKENOX identity speech/singing dataset folders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.identity_dataset import IdentityDatasetService  # noqa: E402


def main() -> None:
    """Create folders, starter prompts, and print the dataset summary."""

    service = IdentityDatasetService(ROOT)
    structure = service.ensure_structure()
    summary = service.summary()
    print(json.dumps({"structure": structure, "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
