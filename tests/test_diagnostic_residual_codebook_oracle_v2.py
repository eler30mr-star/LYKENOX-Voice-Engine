from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostic_residual_codebook_oracle_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_residual_codebook_oracle_v2", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class ResidualCodebookOracleV2Tests(unittest.TestCase):
    def test_exact_scaled_codeword_recovers_target_level_and_shape(self) -> None:
        torch.manual_seed(7)
        codewords = torch.randn(4, diagnostic.CODEVECTOR_SAMPLES, dtype=torch.float32)
        target = codewords[2] * 11.5
        indices = torch.arange(4, dtype=torch.long)
        selected, index, gain, similarity, mse = (
            diagnostic._oracle_select_level_valid_codevector(target, codewords, indices)
        )
        self.assertEqual(index, 2)
        self.assertAlmostEqual(gain, 11.5, places=4)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        self.assertLess(mse, 1.0e-10)
        self.assertLess(float((selected - target).abs().max()), 1.0e-5)

    def test_sign_is_part_of_oracle_excitation_gain(self) -> None:
        torch.manual_seed(13)
        codewords = torch.randn(3, diagnostic.CODEVECTOR_SAMPLES, dtype=torch.float32)
        target = codewords[1] * -3.25
        indices = torch.arange(3, dtype=torch.long)
        selected, index, gain, similarity, mse = (
            diagnostic._oracle_select_level_valid_codevector(target, codewords, indices)
        )
        self.assertEqual(index, 1)
        self.assertLess(gain, 0.0)
        self.assertAlmostEqual(abs(gain), 3.25, places=4)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        self.assertLess(mse, 1.0e-10)
        self.assertLess(float((selected - target).abs().max()), 1.0e-5)

    def test_no_arbitrary_gain_ceiling_or_post_filter_level_fix(self) -> None:
        source = inspect.getsource(diagnostic)
        self.assertNotIn("MAX_ORACLE_GAIN", source)
        self.assertNotIn("clamp(0.0, 4.0)", source)
        self.assertIn("gain_applied_at_excitation_before_filter", source)
        self.assertIn('"posthoc_output_gain_normalization_used": False', source)
        run_source = inspect.getsource(diagnostic.run_residual_codebook_oracle_v2)
        self.assertIn("render_time_varying_minimum_phase", run_source)
        self.assertNotIn("prediction = prediction *", run_source)
        self.assertNotIn("reference_rms / prediction_rms", run_source)

    def test_report_exposes_audibility_ratios(self) -> None:
        source = inspect.getsource(diagnostic.run_residual_codebook_oracle_v2)
        self.assertIn("selected_to_target_residual_rms_ratio", source)
        self.assertIn("prediction_to_reference_rms_ratio", source)
        self.assertIn("mean_absolute_cosine_similarity", source)
        self.assertIn("vocoder_minimum_phase_residual_codebook_oracle_v2", source)
        self.assertIn("__residual_codebook_oracle_v2.wav", source)

    def test_policy_scope_stays_diagnostic_only(self) -> None:
        source = inspect.getsource(diagnostic).lower()
        self.assertIn('"oracle_indices_signs_or_gains_valid_for_product_inference": false', source)
        self.assertIn('"heldout_residual_added_to_codebook": false', source)
        self.assertIn('"training_executed": false', source)
        self.assertIn('"optimizer_created": false', source)
        self.assertIn('"checkpoint_written": false', source)
        self.assertIn('"production_renderer_modified": false', source)
        for forbidden in (
            "from_pretrained",
            "vocos",
            "hifigan",
            "bigvgan",
            "torch.optim",
            "optimizer.step",
            ".backward(",
            "cuda",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
