"""Tests for professional speech dataset preparation."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

from lykenox_voice_engine.core.identity_dataset import IdentityDatasetService, IdentityPrompt
from lykenox_voice_engine.core.identity_dataset_preparer import IdentityDatasetPreparer


class TestIdentityDatasetPreparer(unittest.TestCase):
    """Validate training manifests and transcript safety checks."""

    def test_prepares_manifest_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = IdentityDatasetService(root)
            for index in range(12):
                prompt = IdentityPrompt(f"speech-{index:03d}", "speech", f"Texto de prueba numero {index}.")
                path = root / f"take_{index}.wav"
                _write_sine(path, duration=2.0)
                service.register_take("speech", prompt, path)

            report = IdentityDatasetPreparer(root).prepare_speech()

            self.assertEqual(report["accepted_speech_count"], 12)
            self.assertGreater(report["train_count"], report["val_count"])
            self.assertTrue(Path(report["manifest"]).exists())
            self.assertFalse(report["ready_for_training"])

    def test_blocks_long_audio_with_short_repeated_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = IdentityDatasetService(root)
            prompt = IdentityPrompt("speech-001", "speech", "Hola.")
            for index in range(4):
                path = root / f"long_{index}.wav"
                _write_sine(path, duration=31.0)
                service.register_take("speech", prompt, path)

            report = IdentityDatasetPreparer(root).prepare_speech()

            self.assertGreater(report["transcript_review_count"], 0)
            self.assertFalse(report["ready_for_training"])
            self.assertIn("metadata", report["blocking_reason"])


def _write_sine(path: Path, duration: float) -> None:
    sample_rate = 48_000
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        frames = bytearray()
        for index in range(int(sample_rate * duration)):
            value = int(4000 * math.sin(2.0 * math.pi * 130.0 * index / sample_rate))
            frames.extend(value.to_bytes(2, "little", signed=True))
        writer.writeframes(bytes(frames))


if __name__ == "__main__":
    unittest.main()
