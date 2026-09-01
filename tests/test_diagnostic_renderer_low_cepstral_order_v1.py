from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as production


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostic_renderer_low_cepstral_order_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_renderer_low_cepstral_order_v1", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class LowCepstralOrderOracleDiagnosticTests(unittest.TestCase):
    def test_diagnostic_changes_only_order_64_to_32(self) -> None:
        self.assertEqual(production.CEPSTRAL_ORDER, 64)
        self.assertEqual(diagnostic.PRODUCTION_CEPSTRAL_ORDER, 64)
        self.assertEqual(diagnostic.CEPSTRAL_ORDER_TEST, 32)
        source = inspect.getsource(diagnostic.run_low_cepstral_order_oracle)
        self.assertIn("cepstral_order=CEPSTRAL_ORDER_TEST", source)
        self.assertIn("render_owned_minimum_phase_vocoder_path", source)
        self.assertNotIn("build_neutral_excitation_band_split", source)
        self.assertNotIn("render_owned_minimum_phase_vocoder_path_band_split", source)
        self.assertNotIn("_gaussian_aperiodic_noise", source)
        self.assertNotIn("render_time_varying_minimum_phase_no_crossfade", source)

    def test_production_renderer_accepts_order_32_without_code_change(self) -> None:
        frames = 5
        cepstrum = torch.zeros(1, frames, diagnostic.CEPSTRAL_ORDER_TEST, dtype=torch.float32)
        f0 = torch.full((1, frames), 140.0, dtype=torch.float32)
        voiced = torch.ones_like(f0)
        periodicity = torch.full_like(f0, 0.8)
        waveform, excitation = production.render_owned_minimum_phase_vocoder_path(
            cepstrum,
            f0,
            voiced,
            periodicity,
            noise_seed=0,
        )
        expected_samples = frames * production.HOP_LENGTH
        self.assertEqual(waveform.shape, (1, expected_samples))
        self.assertEqual(excitation.shape, waveform.shape)
        self.assertTrue(bool(torch.isfinite(waveform).all()))

    def test_scope_is_three_heldout_items_and_separate_output(self) -> None:
        source = inspect.getsource(diagnostic.run_low_cepstral_order_oracle)
        self.assertEqual(diagnostic.DEFAULT_SPLIT, "val")
        self.assertEqual(diagnostic.DEFAULT_ITEMS, 3)
        self.assertEqual(diagnostic.NOISE_SEED, 0)
        self.assertEqual(diagnostic.SAMPLE_RATE, 24000)
        self.assertEqual(diagnostic.HOP_LENGTH, 256)
        self.assertEqual(diagnostic.N_FFT, 1024)
        self.assertIn("vocoder_minimum_phase_oracle_cepstral_order_32_v1", source)
        self.assertIn("__cepstral_order_32.wav", source)
        self.assertIn("original_oracle_prediction", source)
        self.assertIn("minimum_pitch_period_samples", source)

    def test_report_makes_isolation_and_policy_state_explicit(self) -> None:
        source = inspect.getsource(diagnostic)
        self.assertIn('"only_cepstral_order_changed": True', source)
        self.assertIn('"broadband_excitation_unchanged": True', source)
        self.assertIn('"band_split_excitation_used": False', source)
        self.assertIn('"gaussian_noise_used": False', source)
        self.assertIn('"production_hash_noise_used": True', source)
        self.assertIn('"production_crossfade_used": True', source)
        self.assertIn('"production_renderer_modified": False', source)
        self.assertIn('"model_used": False', source)
        self.assertIn('"training_executed": False', source)
        self.assertIn('"checkpoint_loaded": False', source)
        self.assertIn('"checkpoint_written": False', source)
        self.assertIn('"posthoc_gain_normalization_used": False', source)
        self.assertIn('"posthoc_eq_used": False', source)
        self.assertIn('"posthoc_denoising_used": False', source)
        self.assertIn('"predicted_duration_modified": False', source)

    def test_diagnostic_has_no_training_external_model_or_postprocess_path(self) -> None:
        lowered = inspect.getsource(diagnostic).lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "torch.save(",
            "torch.load(",
            "from_pretrained",
            "vocos",
            "hifigan",
            "bigvgan",
            "cuda",
            "normalize(",
            "equalizer",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
