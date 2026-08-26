"""Tests for filtering speech manifests before TTS training."""

from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

from lykenox_voice_engine.core.speech_dataset_filter import filter_speech_manifest


class TestSpeechDatasetFilter(unittest.TestCase):
    """Validate direct-train and blocked speech row classification."""

    def test_filters_generic_and_long_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav = root / "take.wav"
            _write_wav(wav)
            manifest = root / "manifest.auto.jsonl"
            rows = [
                _row("good", wav, "Esta es una frase clara y util para entrenar.", 8.0),
                _row("generic", wav, "Frases rapidas", 8.0),
                _row("long", wav, "Esta transcripcion larga necesita segmentarse antes.", 45.0),
            ]
            manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            result = filter_speech_manifest(manifest, root / "out")

            self.assertEqual(result.usable_rows, 1)
            self.assertEqual(result.blocked_rows, 2)
            blocked = (root / "out" / "blocked.filtered.csv").read_text(encoding="utf-8")
            self.assertIn("generic_metadata", blocked)
            self.assertIn("needs_segmentation", blocked)

    def test_limits_duplicate_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav = root / "take.wav"
            _write_wav(wav)
            manifest = root / "manifest.auto.jsonl"
            text = "Esta frase repetida solo debe quedarse dos veces."
            rows = [_row(f"dup-{index}", wav, text, 8.0) for index in range(3)]
            manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            result = filter_speech_manifest(manifest, root / "out")

            self.assertEqual(result.usable_rows, 2)
            self.assertEqual(result.blocked_rows, 1)


def _row(utterance_id: str, wav: Path, text: str, duration: float) -> dict[str, object]:
    return {
        "utterance_id": utterance_id,
        "wav_path": str(wav),
        "text": text,
        "duration_sec": duration,
        "split": "train",
    }


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(48_000)
        writer.writeframes(b"\0" * 48_000)


if __name__ == "__main__":
    unittest.main()
