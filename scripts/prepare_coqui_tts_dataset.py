"""Prepare segmented LYKENOX speech data in Coqui TTS metadata format."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Create Coqui metadata files from segmented train/validation CSVs."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ROOT
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech_segmented"
        / "train.segmented.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=ROOT
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech_segmented"
        / "val.segmented.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "lykenox" / "identity_voice" / "prepared" / "coqui_segmented",
    )
    parser.add_argument("--ascii-text", action="store_true", help="Normalize metadata text to Coqui default characters")
    args = parser.parse_args()

    report = prepare_dataset(args.train_csv, args.val_csv, args.output_dir, ascii_text=args.ascii_text)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def prepare_dataset(train_csv: Path, val_csv: Path, output_dir: Path, ascii_text: bool = False) -> dict[str, object]:
    """Copy WAV files and write Coqui metadata CSVs."""

    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _copy_rows(train_csv, wav_dir, ascii_text)
    val_rows = _copy_rows(val_csv, wav_dir, ascii_text)
    _write_metadata(output_dir / "metadata_train.csv", train_rows)
    _write_metadata(output_dir / "metadata_val.csv", val_rows)
    report = {
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "metadata_train": str(output_dir / "metadata_train.csv"),
        "metadata_val": str(output_dir / "metadata_val.csv"),
        "ascii_text": ascii_text,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _copy_rows(csv_path: Path, wav_dir: Path, ascii_text: bool) -> list[dict[str, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            source = Path(row["wav_path"])
            target_name = f"{row['utterance_id']}.wav"
            target = wav_dir / target_name
            if not target.exists():
                shutil.copy2(source, target)
            rows.append(
                {
                    "audio_file": f"wavs/{target_name}",
                    "text": _ascii_text(row["text"]) if ascii_text else row["text"],
                    "speaker_name": "lykenox",
                }
            )
    return rows


def _ascii_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("ñ", "n").replace("Ñ", "N"))
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(stripped.split())


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
