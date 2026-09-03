"""Robust one-command entrypoint for common-conditioning vocoder forensics.

No audio generation, model inference, training, or checkpoint writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from diagnose_vocoder_common_conditioning_anomalies_v1 import diagnose


def main() -> None:
    print(json.dumps(diagnose(ROOT), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
