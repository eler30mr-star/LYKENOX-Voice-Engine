"""Create first-pass speech training files from corrected transcript CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.transcript_review import apply_transcripts


def main() -> None:
    """Apply transcript corrections to the prepared speech dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=ROOT
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech"
        / "transcript_review.transcribed.csv",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech",
    )
    parser.add_argument("--source", default="faster-whisper-base-cpu-int8")
    parser.add_argument("--suffix", default="auto")
    args = parser.parse_args()

    result = apply_transcripts(
        prepared_dir=args.prepared_dir,
        review_csv=args.review_csv,
        source=args.source,
        suffix=args.suffix,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
