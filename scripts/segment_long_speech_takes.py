"""Segment long LYKENOX speech recordings into short TTS training clips."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Create sentence-level WAV clips from blocked long speech rows."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blocked-csv",
        type=Path,
        default=ROOT
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech"
        / "blocked.filtered.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech_segmented",
    )
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--min-sec", type=float, default=3.0)
    parser.add_argument("--max-sec", type=float, default=15.0)
    parser.add_argument("--limit-files", type=int, default=0)
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    rows = _long_rows(args.blocked_csv)
    if args.limit_files:
        rows = rows[: args.limit_files]
    wav_dir = args.output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments = []
    for file_index, row in enumerate(rows, start=1):
        print(f"[{file_index}/{len(rows)}] {Path(row['wav_path']).name}", flush=True)
        generated = _segment_file(model, row, wav_dir, args.min_sec, args.max_sec)
        segments.extend(generated)

    train, val = _split(segments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.output_dir / "manifest.segmented.jsonl", train + val)
    _write_csv(args.output_dir / "train.segmented.csv", train)
    _write_csv(args.output_dir / "val.segmented.csv", val)
    report = {
        "source_blocked_csv": str(args.blocked_csv),
        "output_dir": str(args.output_dir),
        "source_files": len(rows),
        "segments": len(segments),
        "train_rows": len(train),
        "val_rows": len(val),
        "minutes": round(sum(float(row["duration_sec"]) for row in segments) / 60.0, 2),
        "ready_for_trial_training": len(segments) >= 50,
    }
    (args.output_dir / "quality_report.segmented.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _long_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [row for row in csv.DictReader(input_file) if row.get("block_reason") == "needs_segmentation"]


def _segment_file(model: object, row: dict[str, str], wav_dir: Path, min_sec: float, max_sec: float) -> list[dict[str, object]]:
    source = Path(row["wav_path"])
    source_id = row["utterance_id"]
    whisper_segments, _info = model.transcribe(
        str(source),
        language="es",
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
        condition_on_previous_text=False,
    )
    rows = []
    buffer_text: list[str] = []
    start_sec: float | None = None
    end_sec = 0.0
    segment_index = 1
    for segment in whisper_segments:
        text = " ".join(segment.text.strip().split())
        if not text:
            continue
        if start_sec is None:
            start_sec = float(segment.start)
        buffer_text.append(text)
        end_sec = float(segment.end)
        duration = end_sec - start_sec
        sentence_done = text.endswith((".", "?", "!", "¿"))
        if duration >= max_sec or (duration >= min_sec and sentence_done):
            output = _cut_segment(source, wav_dir, source_id, segment_index, start_sec, end_sec)
            rows.append(_row(source_id, segment_index, output, buffer_text, start_sec, end_sec))
            segment_index += 1
            buffer_text = []
            start_sec = None
    if buffer_text and start_sec is not None and end_sec - start_sec >= min_sec:
        output = _cut_segment(source, wav_dir, source_id, segment_index, start_sec, end_sec)
        rows.append(_row(source_id, segment_index, output, buffer_text, start_sec, end_sec))
    return rows


def _cut_segment(source: Path, wav_dir: Path, source_id: str, index: int, start: float, end: float) -> Path:
    output = wav_dir / f"{source_id}_seg_{index:03d}.wav"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "48000",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


def _row(source_id: str, index: int, wav_path: Path, texts: list[str], start: float, end: float) -> dict[str, object]:
    return {
        "utterance_id": f"{source_id}_seg_{index:03d}",
        "wav_path": str(wav_path),
        "text": " ".join(texts),
        "duration_sec": round(end - start, 3),
        "source_utterance_id": source_id,
        "source_start_sec": round(start, 3),
        "source_end_sec": round(end, 3),
        "split": "pending",
    }


def _split(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = []
    val = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["split"] = "val" if index % 10 == 0 else "train"
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
            writer.writerow({"utterance_id": row["utterance_id"], "wav_path": row["wav_path"], "text": row["text"]})


if __name__ == "__main__":
    main()
