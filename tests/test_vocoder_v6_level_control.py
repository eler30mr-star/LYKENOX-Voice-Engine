from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV6
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
    target_relative_level_loss,
)


class VocoderV6LevelControlTests(unittest.TestCase):
    def _model(self) -> LykenoxVocoderGeneratorV6:
        return LykenoxVocoderGeneratorV6(
            frame_channels=64,
            upsample_channels=(48, 40, 32, 24),
            sample_channels=32,
            sample_dilations=(1, 2, 4, 8),
        ).eval()

    def test_level_control_raises_rms_without_rewriting_shape(self) -> None:
        torch.manual_seed(6600)
        model = self._model()
        frames = 12
        mel = torch.randn(1, frames, model.config.mel_bins)
        f0_hz = torch.full((1, frames), 120.0)
        voiced = torch.ones(1, frames)

        with torch.inference_mode():
            baseline = model(mel, f0_hz, voiced)
            original = model.level_logit_bias_parameter.detach().clone()
            model.level_logit_bias_parameter.add_(0.50)
            raised = model(mel, f0_hz, voiced)
            model.level_logit_bias_parameter.copy_(original)

        baseline_rms = baseline.square().mean().sqrt()
        raised_rms = raised.square().mean().sqrt()
        self.assertGreater(float(raised_rms), float(baseline_rms) * 1.15)

        baseline_unit = baseline / baseline_rms.clamp_min(1e-8)
        raised_unit = raised / raised_rms.clamp_min(1e-8)
        shape_delta = (baseline_unit - raised_unit).abs().mean()
        self.assertLess(float(shape_delta), 0.05)

        self.assertTrue(model.waveform_shape_level_decoupled)
        self.assertEqual(
            model.level_control_family,
            "mel_conditioned_frame_rms_envelope",
        )

    def test_level_loss_tracks_absolute_rms_error(self) -> None:
        torch.manual_seed(7)
        target = torch.randn(1, 4096)
        target = target / target.square().mean().sqrt() * 0.04
        weak = target * 0.25
        closer = target * 0.80

        weak_result = target_relative_level_loss(weak, target)
        closer_result = target_relative_level_loss(closer, target)

        self.assertEqual(VOCODER_LEVEL_PRESENCE_VERSION, "vocoder-level-presence-v2")
        self.assertLess(float(closer_result.loss), float(weak_result.loss))
        self.assertLess(
            float(closer_result.rms_error_db),
            float(weak_result.rms_error_db),
        )
        self.assertGreater(float(weak_result.rms_error_db), 10.0)


if __name__ == "__main__":
    unittest.main()
