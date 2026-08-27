from __future__ import annotations

import unittest

from lykenox_voice_engine.training.speech_duration_outlier_review import (
    classify_boundary_pattern,
)


class SpeechDurationOutlierReviewTests(unittest.TestCase):
    def test_boundary_heavy_pattern_is_flagged(self) -> None:
        self.assertEqual(
            classify_boundary_pattern(boundary_count=8, interior_count=2),
            "boundary_silence_absorption_likely",
        )

    def test_mixed_pattern_is_distinguished(self) -> None:
        self.assertEqual(
            classify_boundary_pattern(boundary_count=2, interior_count=4),
            "mixed_boundary_and_interior_outliers",
        )

    def test_no_outliers_passes_cleanly(self) -> None:
        self.assertEqual(
            classify_boundary_pattern(boundary_count=0, interior_count=0),
            "no_long_nonpause_outliers",
        )


if __name__ == "__main__":
    unittest.main()
