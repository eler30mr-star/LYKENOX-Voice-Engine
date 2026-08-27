"""Tests for UTAU-style oto.ini handling."""

from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from lykenox_voice_engine.core.oto import OtoEntry, estimate_oto_entry, parse_oto, write_oto


class TestOto(unittest.TestCase):
    """Validate OTO parsing and initial timing estimation."""

    def test_parse_and_write_oto(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "oto.ini"
            entry = OtoEntry("ba.wav", "ba", 1.0, 80.0, -50.0, 60.0, 25.0)
            write_oto(path, [entry])
            parsed = parse_oto(path)
        self.assertEqual(parsed["ba"].wav, "ba.wav")
        self.assertEqual(parsed["ba"].preutterance, 60.0)

    def test_estimate_oto_entry_from_energy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ba.wav"
            _write_wav(path)
            entry = estimate_oto_entry(path, "ba")
        self.assertEqual(entry.alias, "ba")
        self.assertGreaterEqual(entry.preutterance, entry.offset)


def _write_wav(path: Path) -> None:
    sample_rate = 48_000
    frames = bytearray(b"\x00\x00" * int(sample_rate * 0.03))
    for index in range(int(sample_rate * 0.2)):
        value = int(math.sin(index / 10.0) * 9000)
        frames.extend(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(bytes(frames))


if __name__ == "__main__":
    unittest.main()
