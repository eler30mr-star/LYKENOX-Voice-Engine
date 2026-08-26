"""Filter prepared speech manifests into trainable and blocked subsets."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


GENERIC_TEXT_PATTERNS = (
    re.compile(r"^frases?\s+", re.IGNORECASE),
    re.compile(r"^habla\s+", re.IGNORECASE),
)


@dataclass(frozen=True)
class SpeechFilterResult:
    """Summary of a speech dataset filtering pass."""

    usable_manifest: str
    usable_train_csv: str
    usable_val_csv: str
    blocked_csv: str
    report_path: str
    total_rows: int
    usable_rows: int
    train_rows: int
    val_rows: int
    blocked_rows: int
    usable_minutes: float
    segmentation_minutes: float
    record_more_recommended: bool


def filter_speech_manifest(
    manifest_path: Path,
    output_dir: Path,
    min_text_chars: int = 35,
    max_duration_sec: float = 20.0,
    min_chars_per_sec: float = 3.0,
    max_duplicate_text_count: int = 2,
) -> SpeechFilterResult:
    """Write trainable speech rows and blocked reasons from a prepared manifest.

    Args:
        manifest_path: JSONL manifest created by transcript application.
        output_dir: Directory where filtered files are written.
        min_text_chars: Minimum transcript length for direct TTS training.
        max_duration_sec: Maximum clip length before segmentation is required.
        min_chars_per_sec: Minimum text density to catch title-like metadata.
        max_duplicate_text_count: Maximum repeated identical transcript rows to keep.

    Returns:
        Paths and counts for the filtered dataset.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """

    rows = _read_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_counts = Counter(str(row.get("text", "")).strip().lower() for row in rows)
    kept_by_text: Counter[str] = Counter()
    usable = []
    blocked = []
    for row in rows:
        reason = _block_reason(
            row,
            text_counts,
            kept_by_text,
            min_text_chars,
            max_duration_sec,
            min_chars_per_sec,
            max_duplicate_text_count,
        )
        if reason:
            blocked.append({**row, "block_reason": reason})
        else:
            text_key = str(row["text"]).strip().lower()
            kept_by_text[text_key] += 1
            usable.append(row)

    train, val = _split(usable)
    usable_manifest = output_dir / "manifest.filtered.jsonl"
    usable_train = output_dir / "train.filtered.csv"
    usable_val = output_dir / "val.filtered.csv"
    blocked_csv = output_dir / "blocked.filtered.csv"
    report_path = output_dir / "quality_report.filtered.json"
    _write_manifest(usable_manifest, train + val)
    _write_csv(usable_train, train)
    _write_csv(usable_val, val)
    _write_blocked_csv(blocked_csv, blocked)
    segmentation_minutes = round(
        sum(float(row.get("duration_sec", 0.0)) for row in blocked if row.get("block_reason") == "needs_segmentation")
        / 60.0,
        2,
    )
    result = SpeechFilterResult(
        usable_manifest=str(usable_manifest),
        usable_train_csv=str(usable_train),
        usable_val_csv=str(usable_val),
        blocked_csv=str(blocked_csv),
        report_path=str(report_path),
        total_rows=len(rows),
        usable_rows=len(usable),
        train_rows=len(train),
        val_rows=len(val),
        blocked_rows=len(blocked),
        usable_minutes=round(sum(float(row.get("duration_sec", 0.0)) for row in usable) / 60.0, 2),
        segmentation_minutes=segmentation_minutes,
        record_more_recommended=len(usable) < 50,
    )
    report_path.write_text(
        json.dumps({**result.__dict__, "blocked_reasons": dict(Counter(row["block_reason"] for row in blocked))}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _block_reason(
    row: dict[str, object],
    text_counts: Counter[str],
    kept_by_text: Counter[str],
    min_text_chars: int,
    max_duration_sec: float,
    min_chars_per_sec: float,
    max_duplicate_text_count: int,
) -> str | None:
    text = str(row.get("text", "")).strip()
    duration = float(row.get("duration_sec", 0.0))
    text_key = text.lower()
    if not Path(str(row.get("wav_path", ""))).exists():
        return "missing_wav"
    if _is_generic_text(text):
        return "generic_metadata"
    if len(text) < min_text_chars:
        return "too_little_text"
    if duration > max_duration_sec:
        return "needs_segmentation"
    if duration > 0 and (len(text) / duration) < min_chars_per_sec:
        return "text_audio_mismatch"
    if text_counts[text_key] > max_duplicate_text_count and kept_by_text[text_key] >= max_duplicate_text_count:
        return "duplicate_over_limit"
    return None


def _is_generic_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in GENERIC_TEXT_PATTERNS)


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _split(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows:
        return [], []
    val_count = max(1, round(len(rows) * 0.1))
    train = []
    val = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["split"] = "val" if index % max(1, len(rows) // val_count) == 0 else "train"
        if updated["split"] == "val":
            val.append(updated)
        else:
            train.append(updated)
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


def _write_blocked_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["utterance_id", "duration_sec", "text", "wav_path", "block_reason"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
