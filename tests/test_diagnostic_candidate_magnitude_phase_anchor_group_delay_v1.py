from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostic_candidate_magnitude_phase_anchor_group_delay_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_candidate_magnitude_phase_anchor_group_delay_v1",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class CandidateMagnitudePhaseAnchorGroupDelayDiagnosticTests(unittest.TestCase):
    def test_scope_and_fixed_controls(self) -> None:
        self.assertEqual(diagnostic.CROSSOVER_LOW_HZ, 3000.0)
        self.assertEqual(diagnostic.CROSSOVER_HIGH_HZ, 4000.0)
        self.assertEqual(diagnostic.SMOOTH_PHASE_OFFSET_BINS, 63)
        self.assertEqual(
            diagnostic.DEFAULT_UTTERANCE_IDS,
            (
                "speech_0021_6cd35984e877_seg_001",
                "speech_0022_ba721f6129b9_seg_005",
            ),
        )

    def test_diagnostic_is_no_training_no_renderer_modification(self) -> None:
        source = inspect.getsource(diagnostic).lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "torch.save(",
            "posthoc_gain_normalization_used\": true",
            "posthoc_eq_used\": true",
            "posthoc_denoising_used\": true",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"training_executed": false', source)
        self.assertIn('"optimizer_created": false', source)
        self.assertIn('"renderer_modified": false', source)

    def test_only_anchor_changes_while_candidate_magnitude_and_target_dphase_are_fixed(self) -> None:
        source = inspect.getsource(diagnostic.run_candidate_magnitude_phase_anchor_group_delay)
        self.assertIn("target_inc = _temporal_phase_increments(target_spec)", source)
        self.assertIn("candidate_mag = candidate_spec.abs()", source)
        self.assertIn("target_low_anchor", source)
        self.assertIn("target_high_anchor", source)
        self.assertIn("smooth_corrected_anchor", source)
        self.assertIn("candidate_mag_target_dphase_candidate_anchor_render", source)
        self.assertIn("candidate_mag_target_dphase_target_anchor_ceiling", source)
        self.assertIn("candidate_mag_target_dphase_target_low_anchor_render", source)
        self.assertIn("candidate_mag_target_dphase_target_high_anchor_render", source)
        self.assertIn("candidate_mag_target_dphase_smooth_group_delay_anchor_render", source)


if __name__ == "__main__":
    unittest.main()
