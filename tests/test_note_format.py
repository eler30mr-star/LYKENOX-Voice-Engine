"""Tests for the internal singing note format."""

from __future__ import annotations

import unittest

from lykenox_voice_engine.models.note import NoteSequence


class TestNoteFormat(unittest.TestCase):
    """Validate accepted and rejected note payloads."""

    def test_accepts_valid_notes(self) -> None:
        sequence = NoteSequence.from_dict(
            {"tempo": 120, "notes": [{"lyric": "la", "midi": 60, "start": 0.0, "duration": 1.0}]}
        )
        self.assertEqual(sequence.tempo, 120)
        self.assertEqual(sequence.notes[0].midi, 60)

    def test_rejects_invalid_duration(self) -> None:
        with self.assertRaises(ValueError):
            NoteSequence.from_dict(
                {"tempo": 120, "notes": [{"lyric": "la", "midi": 60, "start": 0.0, "duration": 0.0}]}
            )


if __name__ == "__main__":
    unittest.main()
