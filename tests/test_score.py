"""Tests for saved LYKENOX score files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.core.score import load_score, score_to_json


class TestScore(unittest.TestCase):
    """Verify scores are real files instead of hidden report data."""

    def test_load_microtest_score(self) -> None:
        root = Path(__file__).resolve().parents[1]

        score = load_score(root / "scores" / "baila_conmigo_microtest.json")

        self.assertEqual(score.lyrics, "baila conmigo")
        self.assertEqual(score.tempo, 120)
        self.assertEqual([note.lyric for note in score.notes], ["bai", "la", "con", "mi", "go"])
        self.assertEqual([note.midi for note in score.notes], [60, 62, 64, 62, 60])

    def test_score_to_json_keeps_notes_for_editor(self) -> None:
        root = Path(__file__).resolve().parents[1]
        score = load_score(root / "scores" / "baila_conmigo_microtest.json")

        payload = score_to_json(score)

        self.assertIn('"midi": 64', payload)
        self.assertIn('"lyric": "con"', payload)


if __name__ == "__main__":
    unittest.main()
