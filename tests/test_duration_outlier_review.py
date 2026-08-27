from __future__ import annotations

import unittest

from lykenox_voice_engine.training.speech_duration_outlier_review import (
    _diagnosis_for_cache,
    classify_boundary_pattern,
)


class DurationOutlierReviewTests(unittest.TestCase):
    def test_boundary_heavy_pattern_is_detected(self) -> None:
        self.assertEqual(
            classify_boundary_pattern(20, 5),
            "boundary_silence_absorption_likely",
        )

    def test_v1_boundary_pattern_requests_algorithm_fix(self) -> None:
        diagnosis, next_gate = _diagnosis_for_cache(
            "boundary_silence_absorption_likely",
            "alignment-v1",
            25,
        )
        self.assertEqual(diagnosis, "boundary_silence_absorption_likely")
        self.assertEqual(next_gate, "fix_boundary_blank_assignment")

    def test_v2_boundary_pattern_is_residual_not_same_bug(self) -> None:
        diagnosis, next_gate = _diagnosis_for_cache(
            "boundary_silence_absorption_likely",
            "alignment-v2",
            4,
        )
        self.assertEqual(diagnosis, "residual_boundary_alignment_outliers")
        self.assertEqual(next_gate, "inspect_residual_boundary_outliers")

    def test_clean_v2_cache_advances_to_acoustic_smoke(self) -> None:
        diagnosis, next_gate = _diagnosis_for_cache(
            "no_long_nonpause_outliers",
            "alignment-v2",
            0,
        )
        self.assertEqual(diagnosis, "duration_distribution_clean")
        self.assertEqual(next_gate, "aligned_acoustic_smoke")


if __name__ == "__main__":
    unittest.main()
