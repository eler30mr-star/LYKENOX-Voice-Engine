"""Tests for OpenUtau WORLDLINE-R phrase request semantics."""

from __future__ import annotations

import unittest
from pathlib import Path

from lykenox_voice_engine.core.oto import parse_oto
from lykenox_voice_engine.engines.openutau_phrase_adapter import OpenUtauPhraseAdapter
from lykenox_voice_engine.models.notes import NoteEvent


class TestOpenUtauPhraseAdapter(unittest.TestCase):
    """Verify OpenUtau-style timing and pitch request construction."""

    def setUp(self) -> None:
        """Create an adapter over the checked-in LYKENOX Spanish Lite voicebank."""

        self.root = Path(__file__).resolve().parents[1]
        self.wav_dir = self.root / "profiles" / "lykenox" / "voicebank" / "wav"
        self.oto = parse_oto(self.root / "profiles" / "lykenox" / "voicebank" / "oto.ini")
        self.adapter = OpenUtauPhraseAdapter(
            self.wav_dir,
            self.oto,
            lambda lyric: [lyric],
            120,
        )

    def test_pitch_curve_matches_midi_sequence(self) -> None:
        """Sampled F0 curve follows the requested MIDI note sequence."""

        notes = [
            NoteEvent("a", 60, 0.0, 1.5),
            NoteEvent("a", 62, 1.5, 1.5),
            NoteEvent("a", 64, 3.0, 1.5),
            NoteEvent("a", 62, 4.5, 1.5),
            NoteEvent("a", 60, 6.0, 1.5),
        ]
        phrase = self.adapter.build_phrase(notes)
        f0 = self.adapter.sample_f0_curve(phrase)
        for request, expected in zip(phrase.requests, [261.63, 293.66, 329.63, 293.66, 261.63]):
            sample_sec = (
                request.pos_ms
                + request.phone.preutter_ms
                + request.consonant
                + 250.0
            ) / 1000.0
            self.assertAlmostEqual(f0[int(sample_sec * 100)], expected, delta=0.5)

    def test_skipover_uses_openutau_formula(self) -> None:
        """skipOver is oto.Preutter * stretchRatio - phone.leadingMs."""

        phrase = self.adapter.build_phrase([NoteEvent("a", 60, 0.0, 1.5)])
        request = phrase.requests[0]
        self.assertEqual(request.velocity, 100)
        self.assertAlmostEqual(request.skip_over, 0.0, places=3)

    def test_generated_cutoff_preserves_sustain_region(self) -> None:
        """Generated positive cutoff leaves a long analysis region for WORLDLINE-R."""

        phrase = self.adapter.build_phrase([NoteEvent("a", 60, 0.0, 1.5)])
        request = phrase.requests[0]
        self.assertGreater(request.cutoff, 0)
        useful_ms = 5860.0 - request.offset - request.cutoff
        self.assertGreater(useful_ms, 3000.0)


if __name__ == "__main__":
    unittest.main()
