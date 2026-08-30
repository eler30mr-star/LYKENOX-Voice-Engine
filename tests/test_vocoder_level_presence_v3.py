from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
    _band_fractions,
    target_relative_presence_loss,
)


class VocoderLevelPresenceV3Tests(unittest.TestCase):
    def _wave(self, amplitudes: tuple[float, float, float, float]) -> torch.Tensor:
        sample_rate = 24_000
        samples = 4096
        t = torch.arange(samples, dtype=torch.float32) / sample_rate
        frequencies = (180.0, 600.0, 1800.0, 5000.0)
        wave = sum(
            amplitude * torch.sin(2.0 * math.pi * frequency * t)
            for amplitude, frequency in zip(amplitudes, frequencies, strict=True)
        )
        return wave.unsqueeze(0)

    def test_identity_has_zero_presence_loss(self) -> None:
        target = self._wave((1.0, 0.7, 0.35, 0.20))
        result = target_relative_presence_loss(target, target)
        self.assertEqual(VOCODER_LEVEL_PRESENCE_VERSION, "vocoder-level-presence-v3")
        self.assertLess(float(result.loss), 1e-7)
        self.assertLess(float(result.presence_1k_8k_error_db), 1e-6)

    def test_high_band_underpresence_adds_one_sided_guard(self) -> None:
        target = self._wave((1.0, 0.7, 0.35, 0.20))
        prediction = self._wave((1.0, 0.7, 0.12, 0.035)).requires_grad_(True)

        result = target_relative_presence_loss(prediction, target)
        pred = _band_fractions(
            prediction,
            sample_rate=24_000,
            n_fft=1024,
            hop_length=256,
            eps=1e-8,
        )
        ref = _band_fractions(
            target,
            sample_rate=24_000,
            n_fft=1024,
            hop_length=256,
            eps=1e-8,
        )
        log_delta = torch.log(pred.clamp_min(1e-8)) - torch.log(ref.clamp_min(1e-8))
        weights = torch.tensor([0.75, 1.0, 1.5, 1.5])
        symmetric_only = (
            F.smooth_l1_loss(
                log_delta,
                torch.zeros_like(log_delta),
                reduction="none",
            )
            * weights
        ).mean()

        self.assertLess(float(pred[0, 2]), float(ref[0, 2]))
        self.assertLess(float(pred[0, 3]), float(ref[0, 3]))
        self.assertGreater(float(result.loss), float(symmetric_only) + 1e-5)
        result.loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
