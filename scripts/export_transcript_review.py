"""Export transcript review rows for identity speech recordings."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Create a CSV template for correcting speech transcripts."""

    report_path = ROOT / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech" / "quality_report.json"
    output_path = ROOT / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech" / "transcript_review.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "utterance_id",
                "duration_sec",
                "wav_path",
                "current_text",
                "corrected_text",
                "warning",
            ],
        )
        writer.writeheader()
        for row in report.get("warnings", []):
            writer.writerow(
                {
                    "utterance_id": row["utterance_id"],
                    "duration_sec": row["duration_sec"],
                    "wav_path": row["wav_path"],
                    "current_text": row["text"],
                    "corrected_text": "",
                    "warning": row["warning"],
                }
            )
    print(output_path)


if __name__ == "__main__":
    main()
