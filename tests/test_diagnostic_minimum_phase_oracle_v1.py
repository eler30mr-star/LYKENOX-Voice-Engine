from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnostic_minimum_phase_oracle_v1.py"
SPEC = importlib.util.spec_from_file_location("diagnostic_minimum_phase_oracle_v1", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


class MinimumPhaseOracleV1Tests(unittest.TestCase):
    def test_reference_log_magnitude_uses_exact_renderer_geometry(self) -> None:
        frames = 32
        samples = frames * oracle.HOP_LENGTH
        waveform = torch.sin(torch.arange(samples, dtype=torch.float32) * 0.031)
        log_magnitude, analysis_frames = oracle._reference_log_magnitude(
            waveform,
            frame_count=frames,
        )
        self.assertEqual(log_magnitude.shape, (frames, 513))
        self.assertGreaterEqual(analysis_frames, frames)
        self.assertTrue(bool(torch.isfinite(log_magnitude).all()))

    def test_script_is_pure_dsp_read_only_oracle(self) -> None:
        source = inspect.getsource(oracle)
        lowered = source.lower()
        self.assertIn("collect_owned_vocoder_utterances", source)
        self.assertIn("_centered_stft_magnitude", source)
        self.assertIn("reference_log_magnitude_to_one_sided_cepstrum", source)
        self.assertIn("render_owned_minimum_phase_vocoder_path", source)
        self.assertIn('subtype="FLOAT"', source)
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

    def test_oracle_defaults_match_requested_scope(self) -> None:
        self.assertEqual(oracle.DEFAULT_SPLIT, "val")
        self.assertEqual(oracle.DEFAULT_ITEMS, 3)
        self.assertEqual(oracle.SAMPLE_RATE, 24000)
        self.assertEqual(oracle.HOP_LENGTH, 256)
        self.assertEqual(oracle.N_FFT, 1024)
        self.assertEqual(oracle.CEPSTRAL_ORDER, 64)
        self.assertEqual(oracle.NOISE_SEED, 0)
        self.assertEqual(oracle.POLICY_ID, "LYX-POL-001")


if __name__ == "__main__":
    unittest.main()
