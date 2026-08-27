"""Tests for minimal MIDI parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.core.midi import parse_midi


class TestMidi(unittest.TestCase):
    """Validate a tiny Standard MIDI File."""

    def test_parse_single_note_with_tempo_and_lyric(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "one.mid"
            path.write_bytes(_single_note_midi())
            parsed = parse_midi(path)
        self.assertEqual(parsed.tempo, 120)
        self.assertEqual(parsed.notes[0].lyric, "la")
        self.assertEqual(parsed.notes[0].midi, 60)
        self.assertEqual(parsed.notes[0].duration, 1.0)


def _single_note_midi() -> bytes:
    header = b"MThd" + (6).to_bytes(4, "big") + b"\x00\x00\x00\x01\x01\xE0"
    events = bytearray()
    events.extend(b"\x00\xFF\x51\x03\x07\xA1\x20")
    events.extend(b"\x00\xFF\x05\x02la")
    events.extend(b"\x00\x90\x3C\x40")
    events.extend(b"\x83\x60\x80\x3C\x00")
    events.extend(b"\x00\xFF\x2F\x00")
    return header + b"MTrk" + len(events).to_bytes(4, "big") + bytes(events)


if __name__ == "__main__":
    unittest.main()
