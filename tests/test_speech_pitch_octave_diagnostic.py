from __future__ import annotations

import inspect
import unittest

import torch

from tools.diagnostics import speech_pitch_octave_diagnostic as diagnostic


class SpeechPitchOctaveDiagnosticTests(unittest.TestCase):
    def test_detects_up_and_down_octave_like_adjacent_voiced_jumps(self) -> None:
        f0 = torch.tensor([100.0, 102.0, 204.0, 202.0, 101.0, 100.0])
        jumps = diagnostic.detect_octave_jumps(f0, tolerance_fraction=0.05)
        self.assertEqual(len(jumps), 2)
        self.assertEqual(jumps[0]["frame"], 2)
        self.assertEqual(jumps[0]["direction"], "up_2x")
        self.assertEqual(jumps[1]["frame"], 4)
        self.assertEqual(jumps[1]["direction"], "down_0.5x")

    def test_unvoiced_gap_breaks_adjacent_transition(self) -> None:
        f0 = torch.tensor([100.0, 0.0, 200.0])
        self.assertEqual(diagnostic.detect_octave_jumps(f0), [])
        self.assertEqual(diagnostic._voiced_transition_count(f0), 0)

    def test_tolerance_rejects_non_octave_large_jump(self) -> None:
        f0 = torch.tensor([100.0, 180.0, 100.0])
        self.assertEqual(diagnostic.detect_octave_jumps(f0, tolerance_fraction=0.05), [])

    def test_even_selection_spans_manifest(self) -> None:
        self.assertEqual(diagnostic._evenly_spaced_indices(118, 5), [0, 29, 58, 88, 117])
        self.assertEqual(diagnostic._evenly_spaced_indices(14, 3), [0, 6, 13])

    def test_diagnostic_is_cpu_read_only_and_uses_pitch_v1(self) -> None:
        source = inspect.getsource(diagnostic)
        lowered = source.lower()
        self.assertIn("extract_pitch_frames", source)
        self.assertIn('"device": "cpu"', source)
        self.assertIn('"pitch_cache_modified": False', source)
        self.assertIn('"optimizer_created": False', source)
        self.assertIn('"parameter_update_executed": False', source)
        self.assertIn('"checkpoint_written": False', source)
        for forbidden in (
            "torch.optim",
            "optimizer.step",
            ".backward(",
            "from_pretrained",
            "vocos",
            "hifigan",
            "bigvgan",
            "cuda",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
