from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.core.ctc_alignment import CTCForcedAlignment
from lykenox_voice_engine.core.ctc_alignment_diagnostics import (
    ctc_frame_ownership_breakdown,
)
from lykenox_voice_engine.training.speech_interior_alignment_review import (
    classify_interior_mechanism,
    classify_interior_set,
)


class CTCAlignmentDiagnosticsTests(unittest.TestCase):
    def test_breakdown_separates_direct_and_allocated_blank_frames(self) -> None:
        alignment = CTCForcedAlignment(
            state_path=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
            target_durations=torch.tensor([2, 1], dtype=torch.long),
            leading_boundary_frames=1,
            trailing_boundary_frames=1,
            score=-5.0,
            score_per_step=-1.0,
            downsampled_steps=5,
            mel_frames=5,
        )
        breakdown = ctc_frame_ownership_breakdown(
            alignment,
            target_steps=2,
            frame_stride=1,
        )
        self.assertEqual(breakdown.direct_target_frames.tolist(), [1, 1])
        self.assertEqual(breakdown.allocated_blank_frames.tolist(), [1, 0])
        self.assertEqual(breakdown.content_frames.tolist(), [2, 1])

    def test_classifies_blank_dominant_duration(self) -> None:
        self.assertEqual(
            classify_interior_mechanism(20, 80),
            "interior_blank_allocation_dominant",
        )

    def test_classifies_direct_dominant_duration(self) -> None:
        self.assertEqual(
            classify_interior_mechanism(80, 20),
            "direct_ctc_occupancy_dominant",
        )

    def test_aggregate_diagnosis_requires_majority(self) -> None:
        diagnosis, next_gate = classify_interior_set(
            [
                "interior_blank_allocation_dominant",
                "interior_blank_allocation_dominant",
                "interior_blank_allocation_dominant",
                "direct_ctc_occupancy_dominant",
                "mixed_ctc_occupancy",
            ]
        )
        self.assertEqual(diagnosis, "interior_blank_allocation_dominant")
        self.assertEqual(next_gate, "fix_interior_blank_assignment")


if __name__ == "__main__":
    unittest.main()
