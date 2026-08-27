from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.models.vocoder import LykenoxVocoderConfig, LykenoxVocoderGenerator


class LykenoxVocoderTests(unittest.TestCase):
    def test_upsample_product_matches_hop_length(self) -> None:
        config = LykenoxVocoderConfig()
        product = 1
        for factor in config.upsample_factors:
            product *= factor
        self.assertEqual(product, config.hop_length)

    def test_generator_emits_exact_samples_per_mel_frame(self) -> None:
        config = LykenoxVocoderConfig(channels=32)
        model = LykenoxVocoderGenerator(config)
        mel = torch.randn(2, 7, config.mel_bins)
        waveform = model(mel)
        self.assertEqual(tuple(waveform.shape), (2, 7 * config.hop_length))
        self.assertTrue(bool(torch.isfinite(waveform).all().item()))

    def test_generator_backpropagates(self) -> None:
        config = LykenoxVocoderConfig(channels=32)
        model = LykenoxVocoderGenerator(config)
        mel = torch.randn(1, 4, config.mel_bins)
        waveform = model(mel)
        loss = waveform.square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
