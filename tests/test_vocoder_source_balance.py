from __future__ import annotations

import math

import torch
from torch import nn

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV41
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


def test_v4_1_exact_length_and_no_transposed_convolution() -> None:
    generator = LykenoxVocoderGeneratorV41()
    mel = torch.randn(2, 12, generator.config.mel_bins)
    f0 = torch.full((2, 12), 100.0)
    voiced = torch.ones(2, 12)
    waveform = generator(mel, f0, voiced)
    assert waveform.shape == (2, 12 * generator.config.hop_length)
    assert not any(isinstance(module, nn.ConvTranspose1d) for module in generator.modules())


def test_v4_1_harmonic_envelope_starts_at_v4_baseline() -> None:
    generator = LykenoxVocoderGeneratorV41()
    mel = torch.randn(1, 7, generator.config.mel_bins)
    weights = generator._harmonic_weight_frames(mel)
    expected = torch.tensor(
        [1.0 / float(index) for index in range(1, generator.harmonics + 1)],
        dtype=weights.dtype,
    ).view(1, generator.harmonics, 1)
    assert torch.allclose(weights, expected.expand_as(weights), atol=1e-6, rtol=1e-6)


def test_v4_1_highpass_has_zero_dc_gain() -> None:
    generator = LykenoxVocoderGeneratorV41()
    dc_gain = float(generator.output_highpass_fir.sum())
    assert abs(dc_gain) < 1e-5


def test_source_balance_loss_is_zero_for_identical_waveforms() -> None:
    sample_rate = 24000
    samples = 4096
    t = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = (
        0.5 * torch.sin(2.0 * math.pi * 100.0 * t)
        + 0.2 * torch.sin(2.0 * math.pi * 700.0 * t)
        + 0.1 * torch.sin(2.0 * math.pi * 3500.0 * t)
    ).unsqueeze(0)
    result = target_relative_spectral_balance_loss(waveform, waveform, sample_rate=sample_rate)
    assert float(result.loss) < 1e-7


def test_source_balance_loss_penalizes_missing_upper_speech_energy() -> None:
    sample_rate = 24000
    samples = 4096
    t = torch.arange(samples, dtype=torch.float32) / sample_rate
    target = (
        0.35 * torch.sin(2.0 * math.pi * 100.0 * t)
        + 0.35 * torch.sin(2.0 * math.pi * 900.0 * t)
        + 0.15 * torch.sin(2.0 * math.pi * 3600.0 * t)
    ).unsqueeze(0)
    prediction = (0.75 * torch.sin(2.0 * math.pi * 100.0 * t)).unsqueeze(0)
    result = target_relative_spectral_balance_loss(prediction, target, sample_rate=sample_rate)
    assert torch.isfinite(result.loss)
    assert float(result.loss) > 0.05
