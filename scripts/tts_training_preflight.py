"""Validate the local CPU TTS training runtime against the prepared LYKENOX dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Run a CPU/audio preflight without starting model training."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ROOT
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech"
        / "train.auto.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=ROOT
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech"
        / "val.auto.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "tts_training_preflight.json",
    )
    parser.add_argument("--sample-count", type=int, default=5)
    args = parser.parse_args()

    report = run_preflight(args.train_csv, args.val_csv, args.sample_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def run_preflight(train_csv: Path, val_csv: Path, sample_count: int) -> dict[str, object]:
    """Validate package imports, dataset rows, WAV readability, and mel extraction."""

    start = time.perf_counter()
    import librosa
    import numpy as np
    import soundfile
    import torch

    train_rows = _read_rows(train_csv)
    val_rows = _read_rows(val_csv)
    selected_rows = train_rows[:sample_count]
    sample_reports = []
    for row in selected_rows:
        wav_path = Path(row["wav_path"])
        audio, sample_rate = soundfile.read(wav_path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=40,
            fmax=min(12_000, sample_rate // 2),
            power=1.0,
        )
        mel_tensor = torch.from_numpy(np.log(np.maximum(mel, 1e-5))).float()
        sample_reports.append(
            {
                "utterance_id": row["utterance_id"],
                "wav_exists": wav_path.exists(),
                "sample_rate": int(sample_rate),
                "duration_sec": round(float(len(audio) / sample_rate), 3),
                "text_chars": len(row["text"]),
                "mel_shape": list(mel_tensor.shape),
                "mel_finite": bool(torch.isfinite(mel_tensor).all().item()),
            }
        )

    total_seconds = time.perf_counter() - start
    return {
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "librosa": librosa.__version__,
            "soundfile": soundfile.__version__,
        },
        "dataset": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "missing_wavs": _missing_wavs(train_rows + val_rows),
        },
        "samples_checked": sample_reports,
        "ready_for_cpu_microtraining": bool(train_rows) and bool(val_rows) and not _missing_wavs(train_rows + val_rows),
        "elapsed_sec": round(total_seconds, 3),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def _missing_wavs(rows: list[dict[str, str]]) -> list[str]:
    return [row["wav_path"] for row in rows if not Path(row["wav_path"]).exists()]


if __name__ == "__main__":
    main()
