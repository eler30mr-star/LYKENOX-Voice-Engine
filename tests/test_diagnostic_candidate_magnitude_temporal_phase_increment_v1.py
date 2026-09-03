from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path
import unittest

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostic_candidate_magnitude_temporal_phase_increment_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_candidate_magnitude_temporal_phase_increment_v1", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class CandidateMagnitudeTemporalPhaseIncrementTests(unittest.TestCase):
    def test_gold_oracle_scope_is_fixed(self) -> None:
        self.assertEqual(
            diagnostic.DEFAULT_UTTERANCE_IDS,
            (
                "speech_0021_6cd35984e877_seg_001",
                "speech_0022_ba721f6129b9_seg_005",
            ),
        )

    def test_phase_reconstruction_preserves_given_temporal_increments(self) -> None:
        torch.manual_seed(7)
        bins = 13
        frames = 17
        increments = (torch.rand(bins, frames - 1) * 2.0 - 1.0) * math.pi
        initial = (torch.rand(bins) * 2.0 - 1.0) * math.pi
        phase = diagnostic._phase_from_temporal_increments(increments, initial)
        spectrum = torch.polar(torch.ones_like(phase), phase)
        rebuilt = diagnostic._temporal_phase_increments(spectrum)
        circular_delta = torch.atan2(
            torch.sin(rebuilt - increments),
            torch.cos(rebuilt - increments),
        )
        self.assertLess(float(circular_delta.abs().max()), 2.0e-5)

    def test_different_absolute_anchor_keeps_same_temporal_increments(self) -> None:
        torch.manual_seed(9)
        increments = (torch.rand(9, 11) * 2.0 - 1.0) * math.pi
        phase_a = diagnostic._phase_from_temporal_increments(
            increments,
            torch.zeros(9),
        )
        phase_b = diagnostic._phase_from_temporal_increments(
            increments,
            torch.linspace(-math.pi, math.pi, 9),
        )
        inc_a = diagnostic._temporal_phase_increments(torch.polar(torch.ones_like(phase_a), phase_a))
        inc_b = diagnostic._temporal_phase_increments(torch.polar(torch.ones_like(phase_b), phase_b))
        delta = torch.atan2(torch.sin(inc_a - inc_b), torch.cos(inc_a - inc_b))
        self.assertLess(float(delta.abs().max()), 2.0e-5)

    def test_diagnostic_is_no_training_and_renderer_is_not_modified(self) -> None:
        source = inspect.getsource(diagnostic)
        lowered = source.lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "torch.save(",
            "posthoc_gain_normalization_used\": true",
            "posthoc_eq_used\": true",
            "posthoc_denoising_used\": true",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertIn('"training_executed": False', source)
        self.assertIn('"renderer_modified": False', source)
        self.assertIn("render_time_varying_minimum_phase", source)

    def test_gate_keeps_candidate_magnitude_and_swaps_only_phase_evolution(self) -> None:
        source = inspect.getsource(diagnostic.run_candidate_magnitude_temporal_phase_increment)
        self.assertIn("candidate_mag = candidate_spec.abs()", source)
        self.assertIn("target_inc = _temporal_phase_increments(target_spec)", source)
        self.assertIn("candidate_inc = _temporal_phase_increments(candidate_spec)", source)
        self.assertIn("candidate_mag_target_dphase_candidate_anchor", source)
        self.assertIn("candidate_mag_target_dphase_zero_anchor", source)
        self.assertIn("candidate_mag_candidate_dphase_target_anchor", source)


if __name__ == "__main__":
    unittest.main()
