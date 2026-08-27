from __future__ import annotations

import unittest

import torch
from torch import nn

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderConfig,
    LykenoxVocoderGeneratorV1,
    VOCODER_GENERATOR_V1_ARCHITECTURE,
)


class VocoderResizeConvV1Tests(unittest.TestCase):
    def test_exact_hop_length_without_transposed_convolution(self) -> None:
        config = LykenoxVocoderConfig(channels=32)
        model = LykenoxVocoderGeneratorV1(config)
        mel = torch.randn(2, 11, config.mel_bins)
        waveform = model(mel)
        self.assertEqual(tuple(waveform.shape), (2, 11 * config.hop_length))
        self.assertEqual(model.architecture, VOCODER_GENERATOR_V1_ARCHITECTURE)
        self.assertFalse(any(isinstance(module, nn.ConvTranspose1d) for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
