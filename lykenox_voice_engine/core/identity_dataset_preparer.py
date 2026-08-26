"""Prepare identity recordings for the first speech-model training pass."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from lykenox_voice_engine.core.identity_dataset import IdentityDatasetService, IdentityTakeMetadata


@dataclass(frozen=True)
class PreparedUtterance:
    """One utterance row for a speech training manifest."""

    utterance_id: str
    wav_path: str
    text: str
    duration_sec: float
    rms: int
    peak: int
    split: str
    source_take_id: str
    needs_transcript_review: bool
    warning: str | None


class IdentityDatasetPreparer:
    """Create deterministic manifests while preserving original accepted recordings."""

    def __init__(self, root: Path, profile: str = "lykenox") -> None:
        self.root = root
        self.profile = profile
        self.service = IdentityDatasetService(root, profile)
        self.prepared_dir = self.service.base_dir / "prepared" / "speech"
        self.wav_dir = self.prepared_dir / "wav"
        self.manifest_path = self.prepared_dir / "manifest.jsonl"
        self.train_path = self.prepared_dir / "train.csv"
        self.val_path = self.prepared_dir / "val.csv"
        self.report_path = self.prepared_dir / "quality_report.json"

    def prepare_speech(self) -> dict[str, object]:
        """Prepare accepted speech takes for training review."""

        self.service.ensure_structure()
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        takes = self.service.takes("speech", "accepted")
        rows = [self._prepare_take(index, take, takes) for index, take in enumerate(takes, start=1)]
        train, val = _split(rows)
        self._write_jsonl(rows)
        self._write_csv(self.train_path, train)
        self._write_csv(self.val_path, val)
        report = self._report(rows, train, val)
        self.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _prepare_take(
        self,
        index: int,
        take: IdentityTakeMetadata,
        all_takes: list[IdentityTakeMetadata],
    ) -> PreparedUtterance:
        source = Path(take.wav_path)
        utterance_id = f"speech_{index:04d}_{take.id}"
        target = self.wav_dir / f"{utterance_id}.wav"
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
        warning = _warning_for_take(take, all_takes)
        return PreparedUtterance(
            utterance_id=utterance_id,
            wav_path=str(target),
            text=take.text,
            duration_sec=take.duration_sec,
            rms=take.rms,
            peak=take.peak,
            split="pending",
            source_take_id=take.id,
            needs_transcript_review=warning is not None,
            warning=warning,
        )

    def _write_jsonl(self, rows: list[PreparedUtterance]) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def _write_csv(self, path: Path, rows: list[PreparedUtterance]) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["utterance_id", "wav_path", "text"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "utterance_id": row.utterance_id,
                        "wav_path": row.wav_path,
                        "text": row.text,
                    }
                )

    def _report(
        self,
        rows: list[PreparedUtterance],
        train: list[PreparedUtterance],
        val: list[PreparedUtterance],
    ) -> dict[str, object]:
        durations = [row.duration_sec for row in rows]
        peaks = [row.peak for row in rows]
        warnings = [row for row in rows if row.needs_transcript_review]
        total_sec = sum(durations)
        return {
            "prepared_dir": str(self.prepared_dir),
            "manifest": str(self.manifest_path),
            "train_csv": str(self.train_path),
            "val_csv": str(self.val_path),
            "accepted_speech_count": len(rows),
            "train_count": len(train),
            "val_count": len(val),
            "total_minutes": round(total_sec / 60.0, 2),
            "median_duration_sec": round(statistics.median(durations), 2) if durations else 0.0,
            "long_take_count": sum(1 for value in durations if value > 30.0),
            "near_clip_count": sum(1 for value in peaks if value > 30000),
            "transcript_review_count": len(warnings),
            "ready_for_training": bool(rows) and not warnings and total_sec >= 20 * 60,
            "blocking_reason": _blocking_reason(warnings, total_sec),
            "warnings": [asdict(row) for row in warnings[:20]],
        }


def _warning_for_take(
    take: IdentityTakeMetadata,
    all_takes: list[IdentityTakeMetadata],
) -> str | None:
    same_text = [item for item in all_takes if item.text == take.text and item.status == "accepted"]
    if take.duration_sec > 30.0 and len(take.text) < 120:
        return "audio largo con texto corto; transcripcion probablemente incompleta"
    if len(same_text) > 3 and take.duration_sec > 15.0:
        return "muchas tomas comparten el mismo texto; revisar transcripcion real"
    return None


def _split(rows: list[PreparedUtterance]) -> tuple[list[PreparedUtterance], list[PreparedUtterance]]:
    if not rows:
        return [], []
    val_count = max(1, round(len(rows) * 0.1))
    val_ids = {row.utterance_id for index, row in enumerate(rows) if index % max(1, len(rows) // val_count) == 0}
    train = []
    val = []
    for row in rows:
        updated = PreparedUtterance(
            utterance_id=row.utterance_id,
            wav_path=row.wav_path,
            text=row.text,
            duration_sec=row.duration_sec,
            rms=row.rms,
            peak=row.peak,
            split="val" if row.utterance_id in val_ids else "train",
            source_take_id=row.source_take_id,
            needs_transcript_review=row.needs_transcript_review,
            warning=row.warning,
        )
        if updated.split == "val":
            val.append(updated)
        else:
            train.append(updated)
    return train, val


def _blocking_reason(warnings: list[PreparedUtterance], total_sec: float) -> str | None:
    if warnings:
        return "hay audios cuyo texto en metadata no parece coincidir con el contenido real"
    if total_sec < 20 * 60:
        return "faltan al menos 20 minutos de speech aceptado"
    return None
