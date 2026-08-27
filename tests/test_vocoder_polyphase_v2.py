from __future__ import annotations

import unittest

import torch
from torch import nn

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV2


class VocoderPolyphaseV2Tests(unittest.TestCase):
    def test_exact_waveform_length(self) -> None:
        model = LykenoxVocoderGeneratorV2()
        mel = torch.randn(2, 11, model.config.mel_bins)
        waveform = model(mel)
        self.assertEqual(tuple(waveform.shape), (2, 11 * model.config.hop_length))

    def test_no_transposed_convolution_or_interpolation_module(self) -> None:
        model = LykenoxVocoderGeneratorV2()
        self.assertFalse(any(isinstance(module, nn.ConvTranspose1d) for module in model.modules()))

    def test_polyphase_initialization_is_phase_equal(self) -> None:
        model = LykenoxVocoderGeneratorV2()
        first_stage = model.stages[0]
        weight = first_stage.expand.weight.detach()
        out_channels = first_stage.out_channels
        factor = first_stage.factor
        grouped = weight.reshape(out_channels, factor, *weight.shape[1:])
        reference = grouped[:, :1]
        self.assertTrue(torch.equal(grouped, reference.expand_as(grouped)))

    def test_output_is_finite(self) -> None:
        model = LykenoxVocoderGeneratorV2()
        mel = torch.randn(1, 8, model.config.mel_bins)
        waveform = model(mel)
        self.assertTrue(bool(torch.isfinite(waveform).all()))


if __name__ == "__main__":
    unittest.main()
