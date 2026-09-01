from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as production


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnostic_renderer_no_crossfade_v1.py"
SPEC = importlib.util.spec_from_file_location("diagnostic_renderer_no_crossfade_v1", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class NoCrossfadeRendererDiagnosticTests(unittest.TestCase):
    def test_identical_frame_filters_match_production_renderer(self) -> None:
        torch.manual_seed(7)
        frames = 5
        excitation = torch.randn(1, frames * diagnostic.HOP_LENGTH, dtype=torch.float32)
        one_frame = torch.zeros(1, 1, diagnostic.CEPSTRAL_ORDER, dtype=torch.float32)
        one_frame[..., 0] = -0.2
        one_frame[..., 1] = 0.03
        cepstrum = one_frame.expand(1, frames, -1).contiguous()

        expected = production.render_time_varying_minimum_phase(excitation, cepstrum)
        actual = diagnostic.render_time_varying_minimum_phase_no_crossfade(excitation, cepstrum)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(float((actual - expected).abs().max()), 1.0e-6)

    def test_no_crossfade_function_uses_current_filter_only(self) -> None:
        source = inspect.getsource(diagnostic.render_time_varying_minimum_phase_no_crossfade)
        self.assertIn("block = current", source)
        self.assertNotIn("previous_filter", source)
        self.assertNotIn("previous_output", source)
        self.assertNotIn("torch.linspace", source)

    def test_diagnostic_keeps_step3_sources_and_separate_output(self) -> None:
        source = inspect.getsource(diagnostic.run_no_crossfade_oracle)
        self.assertIn("collect_owned_vocoder_utterances", inspect.getsource(diagnostic))
        self.assertIn("_reference_log_magnitude", inspect.getsource(diagnostic))
        self.assertIn("reference_log_magnitude_to_one_sided_cepstrum", inspect.getsource(diagnostic))
        self.assertIn("vocoder_minimum_phase_oracle_no_crossfade_v1", source)
        self.assertIn("original_crossfade_prediction", source)
        self.assertEqual(diagnostic.DEFAULT_SPLIT, "val")
        self.assertEqual(diagnostic.DEFAULT_ITEMS, 3)
        self.assertEqual(diagnostic.NOISE_SEED, 0)

    def test_diagnostic_is_cpu_read_only_and_has_no_postprocessing(self) -> None:
        source = inspect.getsource(diagnostic)
        lowered = source.lower()
        self.assertIn('subtype="FLOAT"', source)
        self.assertIn('"production_renderer_modified": False', source)
        self.assertIn('"excitation_path_changed": False', source)
        self.assertIn('"crossfade_used": False', source)
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
