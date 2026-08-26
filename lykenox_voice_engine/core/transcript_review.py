"""Apply reviewed or automatic transcripts to prepared speech manifests."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptApplyResult:
    """Summary of a transcript application pass."""

    output_manifest: str
    output_train_csv: str
    output_val_csv: str
    output_report: str
    updated_count: int
    missing_count: int
    source: str
    ready_for_trial_training: bool
    ready_for_production_training: bool


def apply_transcripts(
    prepared_dir: Path,
    review_csv: Path,
    source: str,
    suffix: str = "auto",
) -> TranscriptApplyResult:
    """Create training files with corrected transcripts while preserving originals.

    Args:
        prepared_dir: Directory containing manifest.jsonl, train.csv, and val.csv.
        review_csv: CSV with utterance_id and corrected_text columns.
        source: Human-readable transcript source label.
        suffix: Output filename suffix.

    Returns:
        A summary with generated file paths and readiness flags.

    Raises:
        FileNotFoundError: If required manifest or review CSV files are missing.
        ValueError: If the review CSV lacks required columns.
    """

    manifest_path = prepared_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not review_csv.exists():
        raise FileNotFoundError(review_csv)

    corrections = _read_corrections(review_csv)
    rows = _read_manifest(manifest_path)
    updated_rows = []
    updated_count = 0
    missing_count = 0
    for row in rows:
        corrected_text = corrections.get(row["utterance_id"])
        if corrected_text:
            row = dict(row)
            row["original_text"] = row["text"]
            row["text"] = corrected_text
            row["transcript_source"] = source
            row["needs_transcript_review"] = False
            row["warning"] = None
            updated_count += 1
        elif row.get("needs_transcript_review"):
            missing_count += 1
        updated_rows.append(row)

    train_rows, val_rows = _split_by_manifest(updated_rows)
    output_manifest = prepared_dir / f"manifest.{suffix}.jsonl"
    output_train = prepared_dir / f"train.{suffix}.csv"
    output_val = prepared_dir / f"val.{suffix}.csv"
    output_report = prepared_dir / f"quality_report.{suffix}.json"

    _write_manifest(output_manifest, updated_rows)
    _write_csv(output_train, train_rows)
    _write_csv(output_val, val_rows)
    result = TranscriptApplyResult(
        output_manifest=str(output_manifest),
        output_train_csv=str(output_train),
        output_val_csv=str(output_val),
        output_report=str(output_report),
        updated_count=updated_count,
        missing_count=missing_count,
        source=source,
        ready_for_trial_training=bool(updated_rows) and missing_count == 0,
        ready_for_production_training=False,
    )
    output_report.write_text(
        json.dumps(result.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _read_corrections(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if rows and ("utterance_id" not in rows[0] or "corrected_text" not in rows[0]):
        raise ValueError("review CSV must include utterance_id and corrected_text")
    return {
        row["utterance_id"]: " ".join(row["corrected_text"].split())
        for row in rows
        if row.get("corrected_text", "").strip()
    }


def _read_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _split_by_manifest(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    return train, val


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["utterance_id", "wav_path", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "utterance_id": row["utterance_id"],
                    "wav_path": row["wav_path"],
                    "text": row["text"],
                }
            )
