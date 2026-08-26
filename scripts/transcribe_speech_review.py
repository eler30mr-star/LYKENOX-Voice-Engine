"""Transcribe speech review WAVs with faster-whisper."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Fill transcript review CSV rows with automatic Spanish transcription."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="faster-whisper model size")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    review_path = (
        ROOT
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech"
        / "transcript_review.csv"
    )
    output_path = review_path.with_name("transcript_review.transcribed.csv")
    with review_path.open("r", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    for index, row in enumerate(rows, start=1):
        wav_path = row["wav_path"]
        print(f"[{index}/{len(rows)}] {Path(wav_path).name}", flush=True)
        segments, info = model.transcribe(
            wav_path,
            language="es",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        row["corrected_text"] = _clean_text(text)
        row["detected_language"] = info.language
        row["language_probability"] = f"{info.language_probability:.4f}"
    fieldnames = list(rows[0].keys()) if rows else []
    for extra in ["detected_language", "language_probability"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


if __name__ == "__main__":
    main()
