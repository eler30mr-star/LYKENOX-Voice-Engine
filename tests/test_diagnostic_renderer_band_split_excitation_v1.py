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
    / "diagnostic_renderer_band_split_excitation_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_renderer_band_split_excitation_v1", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class BandSplitExcitationDiagnosticTests(unittest.TestCase):
    def test_highpass_is_exact_complement_of_lowpass_kernel(self) -> None:
        low = production._fixed_lowpass_kernel(
            device=torch.device("cpu"),
            dtype=torch.float64,
            taps=diagnostic.BAND_SPLIT_TAPS,
            cutoff_hz=diagnostic.BAND_SPLIT_HZ,
            sample_rate=diagnostic.SAMPLE_RATE,
        )
        high = diagnostic._fixed_highpass_kernel(
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        combined = low + high
        expected = torch.zeros_like(combined)
        expected[(diagnostic.BAND_SPLIT_TAPS - 1) // 2] = 1.0
        self.assertLess(float((combined - expected).abs().max()), 1.0e-12)

    def test_fully_aperiodic_input_recombines_original_noise(self) -> None:
        frames = 8
        f0 = torch.zeros(1, frames, dtype=torch.float32)
        voiced = torch.zeros_like(f0)
        periodicity = torch.zeros_like(f0)
        actual = diagnostic.build_neutral_excitation_band_split(
            f0,
            voiced,
            periodicity,
            noise_seed=19,
        )
        expected = production._deterministic_aperiodic_noise(
            frames * diagnostic.HOP_LENGTH,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=19,
        ).unsqueeze(0)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(float((actual - expected).abs().max()), 2.0e-5)

    def test_only_final_band_mixing_changes_in_excitation_copy(self) -> None:
        source = inspect.getsource(diagnostic.build_neutral_excitation_band_split)
        self.assertIn("fixed_linear_frame_to_sample", source)
        self.assertIn("_deterministic_aperiodic_noise", source)
        self.assertIn("bandlimited_pulse", source)
        self.assertIn("raw_mix = periodic_strength * bandlimited_pulse", source)
        self.assertIn("low_band", source)
        self.assertIn("high_band_reboosted", source)
        self.assertIn("HIGH_PERIODIC_STRENGTH_SCALE", source)
        self.assertNotIn("_gaussian_aperiodic_noise", source)

    def test_filter_renderer_and_crossfade_remain_production_path(self) -> None:
        path_source = inspect.getsource(
            diagnostic.render_owned_minimum_phase_vocoder_path_band_split
        )
        self.assertIn("render_time_varying_minimum_phase", path_source)
        production_source = inspect.getsource(production.render_time_varying_minimum_phase)
        self.assertIn("previous_filter", production_source)
        self.assertIn("previous_output", production_source)
        self.assertIn("alpha", production_source)

    def test_oracle_scope_and_output_are_separate(self) -> None:
        source = inspect.getsource(diagnostic.run_band_split_oracle)
        self.assertEqual(diagnostic.DEFAULT_SPLIT, "val")
        self.assertEqual(diagnostic.DEFAULT_ITEMS, 3)
        self.assertEqual(diagnostic.NOISE_SEED, 0)
        self.assertEqual(diagnostic.BAND_SPLIT_HZ, 2000.0)
        self.assertEqual(diagnostic.HIGH_PERIODIC_STRENGTH_SCALE, 0.5)
        self.assertIn("vocoder_minimum_phase_oracle_band_split_v1", source)
        self.assertIn("__band_split.wav", source)
        self.assertIn("original_oracle_prediction", source)

    def test_diagnostic_is_cpu_read_only_and_has_no_postprocessing(self) -> None:
        source = inspect.getsource(diagnostic)
        lowered = source.lower()
        self.assertIn('subtype="FLOAT"', source)
        self.assertIn('"production_renderer_modified": False', source)
        self.assertIn('"only_final_excitation_band_mixing_changed": True', source)
        self.assertIn('"crossfade_used": True', source)
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
