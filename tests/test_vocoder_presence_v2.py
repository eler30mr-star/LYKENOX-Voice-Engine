from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_presence_v2 as presence


class OwnedVocoderPresenceV2Tests(unittest.TestCase):
    def test_target_self_is_zero_and_uses_only_valid_context(self) -> None:
        torch.manual_seed(31)
        target = torch.randn(1, 64 * 256, dtype=torch.float32).tanh()
        result = presence.target_relative_presence_loss_v2(target, target)
        self.assertEqual(
            presence.OWNED_VOCODER_PRESENCE_V2_VERSION,
            "owned-vocoder-presence-v2-valid-context-target-relative",
        )
        self.assertEqual(result.analysis_frame_count, 65)
        self.assertEqual(result.valid_frame_count, 61)
        self.assertLess(float(result.loss), 1e-8)
        self.assertLess(float(result.presence_1k_8k_error_db), 1e-7)
        self.assertTrue(
            torch.allclose(
                result.prediction_band_fractions,
                result.target_band_fractions,
                atol=1e-7,
                rtol=0.0,
            )
        )

    def test_high_band_underpresence_has_finite_nonzero_gradient(self) -> None:
        torch.manual_seed(37)
        target = torch.randn(1, 64 * 256, dtype=torch.float32).tanh()
        spectrum = torch.fft.rfft(target, dim=1)
        frequencies = torch.fft.rfftfreq(target.shape[1], d=1.0 / 24000.0)
        gain = torch.ones_like(frequencies)
        gain[(frequencies >= 1000.0) & (frequencies < 8000.0)] = 0.72
        candidate = torch.fft.irfft(
            spectrum * gain.unsqueeze(0),
            n=target.shape[1],
            dim=1,
        ).detach().requires_grad_(True)
        result = presence.target_relative_presence_loss_v2(candidate, target)
        gradient = torch.autograd.grad(result.loss, candidate)[0]
        self.assertTrue(bool(torch.isfinite(result.loss)))
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)
        self.assertGreater(float(result.presence_1k_8k_error_db), 0.0)

    def test_presence_v2_masks_centered_crop_edges_and_has_no_posthoc_path(self) -> None:
        source = inspect.getsource(presence).lower()
        self.assertIn("valid_centered_frame_mask", source)
        self.assertIn("spectrum = spectrum[..., valid_mask]", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("normalize", source)
        self.assertNotIn("equalizer", source)
        self.assertNotIn("denoise", source)


if __name__ == "__main__":
    unittest.main()
