from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as production


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnostic_renderer_gaussian_noise_v1.py"
SPEC = importlib.util.spec_from_file_location("diagnostic_renderer_gaussian_noise_v1", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class GaussianNoiseRendererDiagnosticTests(unittest.TestCase):
    def test_gaussian_noise_is_deterministic_for_same_seed(self) -> None:
        first = diagnostic._gaussian_aperiodic_noise(
            8192,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=97,
        )
        repeat = diagnostic._gaussian_aperiodic_noise(
            8192,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=97,
        )
        other = diagnostic._gaussian_aperiodic_noise(
            8192,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=98,
        )
        self.assertTrue(torch.equal(first, repeat))
        self.assertFalse(torch.equal(first, other))
        self.assertLess(abs(float(first.mean())), 0.05)
        self.assertLess(abs(float(first.std()) - 1.0), 0.05)

    def test_unvoiced_excitation_is_exact_seeded_gaussian_noise(self) -> None:
        frames = 8
        f0 = torch.zeros(1, frames, dtype=torch.float32)
        voiced = torch.zeros_like(f0)
        periodicity = torch.zeros_like(f0)
        excitation = diagnostic.build_neutral_excitation_gaussian_noise(
            f0,
            voiced,
            periodicity,
            noise_seed=31,
        )
        expected = diagnostic._gaussian_aperiodic_noise(
            frames * diagnostic.HOP_LENGTH,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=31,
        ).unsqueeze(0)
        self.assertTrue(torch.equal(excitation, expected))

    def test_path_preserves_production_filter_renderer_and_crossfade(self) -> None:
        source = inspect.getsource(diagnostic.render_owned_minimum_phase_vocoder_path_gaussian_noise)
        self.assertIn("build_neutral_excitation_gaussian_noise", source)
        self.assertIn("render_time_varying_minimum_phase", source)
        self.assertNotIn("render_time_varying_minimum_phase_no_crossfade", source)
        production_source = inspect.getsource(production.render_time_varying_minimum_phase)
        self.assertIn("previous_output", production_source)
        self.assertIn("alpha", production_source)

    def test_only_noise_generator_changes_in_excitation_copy(self) -> None:
        source = inspect.getsource(diagnostic.build_neutral_excitation_gaussian_noise)
        self.assertIn("fixed_linear_frame_to_sample", source)
        self.assertIn("_fixed_lowpass_kernel", source)
        self.assertIn("F.conv1d", source)
        self.assertIn("periodic_strength * bandlimited_pulse", source)
        self.assertIn("aperiodic_strength * base_noise", source)
        self.assertIn("_gaussian_aperiodic_noise", source)
        self.assertNotIn("_deterministic_aperiodic_noise", source)

    def test_oracle_scope_is_three_val_items_and_separate_output(self) -> None:
        source = inspect.getsource(diagnostic.run_gaussian_noise_oracle)
        self.assertEqual(diagnostic.DEFAULT_SPLIT, "val")
        self.assertEqual(diagnostic.DEFAULT_ITEMS, 3)
        self.assertEqual(diagnostic.NOISE_SEED, 0)
        self.assertEqual(diagnostic.SAMPLE_RATE, 24000)
        self.assertEqual(diagnostic.HOP_LENGTH, 256)
        self.assertEqual(diagnostic.N_FFT, 1024)
        self.assertEqual(diagnostic.CEPSTRAL_ORDER, 64)
        self.assertIn("vocoder_minimum_phase_oracle_gaussian_noise_v1", source)
        self.assertIn("original_crossfade_prediction", source)
        self.assertIn("__gaussian_noise.wav", source)

    def test_diagnostic_is_cpu_read_only_and_has_no_postprocessing(self) -> None:
        source = inspect.getsource(diagnostic)
        lowered = source.lower()
        self.assertIn('torch.Generator(device="cpu")', source)
        self.assertIn('subtype="FLOAT"', source)
        self.assertIn('"production_renderer_modified": False', source)
        self.assertIn('"only_noise_generator_changed": True', source)
        self.assertIn('"model_used": False', source)
        self.assertIn('"training_executed": False', source)
        self.assertIn('"checkpoint_loaded": False', source)
        self.assertIn('"checkpoint_written": False', source)
        self.assertIn('"posthoc_gain_normalization_used": False', source)
        self.assertIn('"posthoc_eq_used": False', source)
        self.assertIn('"posthoc_denoising_used": False', source)
        self.assertIn('"predicted_duration_modified": False', source)
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
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
