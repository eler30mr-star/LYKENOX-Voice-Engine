"""Tests for TTS runtime preflight helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.tts_training_preflight import _missing_wavs, _read_rows


class TestTtsTrainingPreflight(unittest.TestCase):
    """Validate dataset CSV helper behavior."""

    def test_reads_rows_and_reports_missing_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "take.wav"
            existing.write_bytes(b"RIFF")
            csv_path = root / "train.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["utterance_id", "wav_path", "text"])
                writer.writeheader()
                writer.writerow({"utterance_id": "one", "wav_path": str(existing), "text": "hola"})
                writer.writerow(
                    {
                        "utterance_id": "two",
                        "wav_path": str(root / "missing.wav"),
                        "text": "adios",
                    }
                )

            rows = _read_rows(csv_path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(_missing_wavs(rows), [str(root / "missing.wav")])


if __name__ == "__main__":
    unittest.main()
