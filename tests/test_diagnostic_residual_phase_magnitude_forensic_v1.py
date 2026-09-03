from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostic_residual_phase_magnitude_forensic_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_residual_phase_magnitude_forensic_v1",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class ResidualPhaseMagnitudeForensicTests(unittest.TestCase):
    def test_locked_gold_oracle_utterances(self) -> None:
        self.assertEqual(
            diagnostic.DEFAULT_UTTERANCE_IDS,
            (
                "speech_0021_6cd35984e877_seg_001",
                "speech_0022_ba721f6129b9_seg_005",
            ),
        )

    def test_hybrid_residuals_preserve_exact_length_and_finiteness(self) -> None:
        torch.manual_seed(7)
        samples = diagnostic.HOP_LENGTH * 24
        target = torch.randn(samples, dtype=torch.float32)
        candidate = torch.randn(samples, dtype=torch.float32)
        target_mag_candidate_phase, candidate_mag_target_phase = diagnostic._hybrid_residuals(
            target,
            candidate,
        )
        self.assertEqual(target_mag_candidate_phase.shape, target.shape)
        self.assertEqual(candidate_mag_target_phase.shape, target.shape)
        self.assertTrue(bool(torch.isfinite(target_mag_candidate_phase).all()))
        self.assertTrue(bool(torch.isfinite(candidate_mag_target_phase).all()))

    def test_diagnostic_has_no_training_or_postprocess_path(self) -> None:
        lowered = inspect.getsource(diagnostic).lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "torch.save(",
            "posthoc_gain_normalization_used\": true",
            "posthoc_eq_used\": true",
            "posthoc_denoising_used\": true",
            "predicted_duration_modified\": true",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_expected_listening_outputs_are_locked(self) -> None:
        source = inspect.getsource(diagnostic.run_residual_phase_magnitude_forensic)
        self.assertIn("__identity_roundtrip_ceiling.wav", source)
        self.assertIn("__candidate_statistics_render.wav", source)
        self.assertIn("__target_mag_candidate_phase_render.wav", source)
        self.assertIn("__candidate_mag_target_phase_render.wav", source)
        self.assertIn("\"metrics_can_accept_product_quality\": False", source)


if __name__ == "__main__":
    unittest.main()
