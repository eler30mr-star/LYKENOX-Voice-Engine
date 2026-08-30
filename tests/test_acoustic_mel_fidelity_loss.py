from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_loss import (
    ACOUSTIC_MEL_FIDELITY_LOSS_VERSION,
    CLARITY_BAND_WEIGHTS,
    acoustic_mel_fidelity_loss,
    mel_bin_centers_hz,
)


class AcousticMelFidelityLossTests(unittest.TestCase):
    def test_identity_is_zero(self) -> None:
        torch.manual_seed(7)
        target = torch.randn(2, 9, 80)
        mask = torch.ones(2, 9, dtype=torch.bool)
        result = acoustic_mel_fidelity_loss(
            target.clone(),
            target,
            mask,
            sample_rate=24000,
            n_fft=1024,
        )
        self.assertEqual(ACOUSTIC_MEL_FIDELITY_LOSS_VERSION, "acoustic-mel-fidelity-loss-v1")
        self.assertAlmostEqual(float(result.total), 0.0, places=7)
        self.assertAlmostEqual(float(result.clarity_underpresence), 0.0, places=7)

    def test_high_band_underpresence_adds_penalty_and_gradient(self) -> None:
        centers = mel_bin_centers_hz(sample_rate=24000, n_fft=1024, mel_bins=80)
        target = torch.zeros(1, 8, 80)
        prediction = target.clone()
        high = (centers >= 3000.0) & (centers < 8000.0)
        prediction[..., high] -= 1.0
        prediction.requires_grad_(True)
        mask = torch.ones(1, 8, dtype=torch.bool)
        result = acoustic_mel_fidelity_loss(
            prediction,
            target,
            mask,
            sample_rate=24000,
            n_fft=1024,
        )
        self.assertGreater(float(result.clarity_underpresence.detach()), 0.0)
        result.total.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad[..., high].abs().sum()), 0.0)

    def test_low_band_has_no_asymmetric_clarity_authority(self) -> None:
        self.assertEqual(CLARITY_BAND_WEIGHTS[0], 0.0)
        centers = mel_bin_centers_hz(sample_rate=24000, n_fft=1024, mel_bins=80)
        target = torch.zeros(1, 6, 80)
        prediction = target.clone()
        low = (centers >= 80.0) & (centers < 300.0)
        prediction[..., low] -= 1.0
        result = acoustic_mel_fidelity_loss(
            prediction,
            target,
            torch.ones(1, 6, dtype=torch.bool),
            sample_rate=24000,
            n_fft=1024,
        )
        self.assertAlmostEqual(float(result.clarity_underpresence), 0.0, places=7)
        self.assertGreater(float(result.mel_l1), 0.0)

    def test_padding_does_not_contribute(self) -> None:
        target = torch.zeros(1, 5, 80)
        prediction = target.clone()
        prediction[:, 3:] = 100.0
        mask = torch.tensor([[True, True, True, False, False]])
        result = acoustic_mel_fidelity_loss(
            prediction,
            target,
            mask,
            sample_rate=24000,
            n_fft=1024,
        )
        self.assertAlmostEqual(float(result.total), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
