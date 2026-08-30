from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV7
from lykenox_voice_engine.training.speech_vocoder_v7_content_loss import (
    VOCODER_V7_CONTENT_LOSS_VERSION,
    V7MelContentConsistencyLoss,
)


class VocoderV7ContractTests(unittest.TestCase):
    def _small_model(self) -> LykenoxVocoderGeneratorV7:
        return LykenoxVocoderGeneratorV7(
            frame_channels=64,
            upsample_channels=(64, 48, 32),
            residual_kernels=(3,),
            residual_dilations=(1,),
        )

    def test_v7_has_no_sample_rate_source_shortcut_contract(self) -> None:
        model = self._small_model()
        self.assertTrue(model.source_free)
        self.assertFalse(model.explicit_source)
        self.assertFalse(model.explicit_sinusoidal_carrier)
        self.assertEqual(model.deterministic_harmonics, 0)
        self.assertFalse(model.voiced_noise_source)
        self.assertFalse(model.deterministic_noise_conditioning)
        self.assertFalse(model.raw_source_bypass)
        self.assertFalse(model.sample_phase_conditioning)
        self.assertFalse(model.sample_rate_pitch_features)
        self.assertEqual(model.pitch_conditioning_scope, "frame_latent_only")
        self.assertFalse(model.local_unit_rms_shape_normalization)
        self.assertFalse(model.global_unit_rms_shape_normalization)
        self.assertFalse(model.level_rescue_branch)
        self.assertFalse(model.posthoc_gain_normalization)
        self.assertFalse(model.perceptually_accepted)

        implementation = (
            inspect.getsource(LykenoxVocoderGeneratorV7._frame_latent)
            + inspect.getsource(LykenoxVocoderGeneratorV7.forward)
        )
        for forbidden in (
            "cumsum(",
            "torch.sin(",
            "torch.cos(",
            "randn(",
            "torch.rand(",
            "remainder(",
        ):
            self.assertNotIn(forbidden, implementation)

    def test_v7_forward_shape_and_gradients(self) -> None:
        torch.manual_seed(7)
        model = self._small_model()
        frames = 4
        mel = torch.randn(1, frames, model.config.mel_bins)
        f0 = torch.full((1, frames), 110.0)
        voiced = torch.ones(1, frames)
        waveform = model(mel, f0, voiced)
        self.assertEqual(tuple(waveform.shape), (1, frames * model.config.hop_length))
        self.assertTrue(bool(torch.isfinite(waveform).all()))
        self.assertLessEqual(float(waveform.abs().max()), 1.0)

        waveform.square().mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))

    def test_v7_content_loss_prefers_matching_conditioning_mel(self) -> None:
        self.assertEqual(VOCODER_V7_CONTENT_LOSS_VERSION, "vocoder-v7-mel-content-v1")
        torch.manual_seed(17)
        loss_fn = V7MelContentConsistencyLoss(boundary_margin_frames=0)
        frames = 10
        samples = frames * loss_fn.config.hop_length
        waveform = torch.randn(1, samples) * 0.03
        with torch.no_grad():
            encoded = torch.log(loss_fn.mel(waveform).clamp_min(1e-5))
            conditioning = encoded[..., :frames].transpose(1, 2).contiguous()
        matching = loss_fn(waveform, conditioning).total
        mismatched = loss_fn(waveform, conditioning.flip(dims=(1,))).total
        self.assertLess(float(matching), float(mismatched))


if __name__ == "__main__":
    unittest.main()
