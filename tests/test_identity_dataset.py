"""Tests for the LYKENOX identity speech/singing dataset."""

from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from lykenox_voice_engine.core.identity_dataset import (
    IdentityDatasetService,
    analyze_identity_wav,
)


class TestIdentityDataset(unittest.TestCase):
    """Validate dataset structure, metadata, and quality-only acceptance."""

    def test_creates_speech_and_singing_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = IdentityDatasetService(Path(temp_dir))
            structure = service.ensure_structure()

            self.assertTrue(Path(structure["speech_raw"]).exists())
            self.assertTrue(Path(structure["singing_raw"]).exists())
            self.assertEqual(len(service.prompts("speech")), 5)
            self.assertEqual(len(service.prompts("singing")), 5)

    def test_registers_take_with_measured_pitch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = IdentityDatasetService(root)
            prompt = service.next_prompt("speech")
            wav_path = root / "take.wav"
            _write_sine(wav_path, 130.0)

            metadata = service.register_take("speech", prompt, wav_path)

            self.assertEqual(metadata.status, "accepted")
            self.assertGreater(metadata.measured_f0_hz, 0)
            self.assertEqual(metadata.prompt_id, prompt.id)
            self.assertTrue(Path(metadata.wav_path).exists())

    def test_rejects_quality_not_pitch_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "quiet.wav"
            _write_sine(wav_path, 180.0, amplitude=100)

            report = analyze_identity_wav(wav_path)

            self.assertFalse(report["valid"])
            self.assertEqual(report["reason"], "nivel demasiado bajo")


def _write_sine(path: Path, hz: float, amplitude: int = 4000) -> None:
    sample_rate = 48_000
    duration = 1.5
    total = int(sample_rate * duration)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        frames = bytearray()
        for index in range(total):
            value = int(amplitude * math.sin(2.0 * math.pi * hz * index / sample_rate))
            frames.extend(value.to_bytes(2, "little", signed=True))
        writer.writeframes(bytes(frames))


if __name__ == "__main__":
    unittest.main()
