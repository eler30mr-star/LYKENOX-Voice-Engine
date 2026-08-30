from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_audit import (
    AUDIT_VERSION,
    BANDS_HZ,
    OUTPUT_DIR_NAME,
    _mel_bin_centers_hz,
    _mel_metrics,
    run_acoustic_mel_fidelity_audit,
)


class AcousticMelFidelityAuditTests(unittest.TestCase):
    def test_identity_metrics_are_exact(self) -> None:
        centers = _mel_bin_centers_hz(sample_rate=24000, n_fft=1024, mel_bins=80)
        target = torch.linspace(-4.0, 1.0, 80).repeat(5, 1)
        metrics = _mel_metrics(target, target.clone(), centers_hz=centers)
        self.assertAlmostEqual(metrics["mel_l1"], 0.0, places=8)
        self.assertAlmostEqual(metrics["centered_shape_l1"], 0.0, places=8)
        self.assertAlmostEqual(metrics["spectral_delta_ratio"], 1.0, places=8)
        self.assertAlmostEqual(metrics["temporal_delta_ratio"], 0.0, places=8)
        for name, _low, _high in BANDS_HZ:
            self.assertAlmostEqual(metrics[f"band_{name}_relative_db"], 0.0, places=8)

    def test_high_band_attenuation_is_detected(self) -> None:
        centers = _mel_bin_centers_hz(sample_rate=24000, n_fft=1024, mel_bins=80)
        target = torch.zeros((8, 80), dtype=torch.float32)
        prediction = target.clone()
        high = (centers >= 3000.0) & (centers < 8000.0)
        prediction[:, high] -= 1.0
        metrics = _mel_metrics(prediction, target, centers_hz=centers)
        self.assertLess(metrics["band_3k_8k_relative_db"], -8.0)
        self.assertGreater(metrics["centered_shape_l1"], 0.0)

    def test_audit_contract_is_read_only(self) -> None:
        self.assertEqual(AUDIT_VERSION, "acoustic-mel-fidelity-heldout-audit-v1")
        self.assertIn("mel_fidelity", OUTPUT_DIR_NAME)
        source = inspect.getsource(run_acoustic_mel_fidelity_audit)
        for forbidden in (
            "optimizer.step",
            ".backward(",
            "run_acoustic_frame_context_training",
            '"training_authorized": True',
            '"predicted_duration_modified": True',
            '"posthoc_gain_normalization_used": True',
            '"posthoc_eq_used": True',
            '"posthoc_denoising_used": True',
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"checkpoints_unchanged": checkpoints_unchanged', source)
        self.assertIn('"training_started": False', source)
        self.assertIn('"training_authorized": False', source)
        self.assertIn('"teacher_duration_grid_used": True', source)


if __name__ == "__main__":
    unittest.main()
