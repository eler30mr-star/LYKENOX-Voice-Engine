"""Filter LYKENOX speech data into trainable and blocked manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.speech_dataset_filter import filter_speech_manifest


def main() -> None:
    """Run the speech dataset quality filter."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech" / "manifest.auto.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech",
    )
    args = parser.parse_args()

    result = filter_speech_manifest(args.manifest, args.output_dir)
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
