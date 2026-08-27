from __future__ import annotations

import torch
from torch import nn

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV3
from lykenox_voice_engine.training.speech_vocoder_periodicity import (
    hop_periodicity_signature,
    target_referenced_periodicity_loss,
)


def test_v3_exact_length_and_no_transposed_convolution() -> None:
    model = LykenoxVocoderGeneratorV3()
    mel = torch.randn(2, 12, model.config.mel_bins)
    waveform = model(mel)
    assert waveform.shape == (2, 12 * model.config.hop_length)
    assert not any(isinstance(module, nn.ConvTranspose1d) for module in model.modules())


def test_v3_phase_scales_are_bounded() -> None:
    model = LykenoxVocoderGeneratorV3()
    scales = model.phase_scales()
    assert len(scales) == len(model.config.upsample_factors)
    assert all(0.0 < value < 1.0 for value in scales)
    assert all(abs(value - 0.10) < 1e-5 for value in scales)


def test_target_referenced_periodicity_loss_is_zero_for_identical_waveforms() -> None:
    waveform = torch.randn(2, 4096)
    result = target_referenced_periodicity_loss(waveform, waveform, hop_length=256)
    assert torch.allclose(result.loss, torch.zeros_like(result.loss), atol=1e-8)


def test_hop_periodicity_signature_detects_exact_grid_carrier() -> None:
    sample_rate = 24000
    hop_length = 256
    samples = torch.arange(8192, dtype=torch.float32)
    carrier = torch.sin(2.0 * torch.pi * samples / hop_length).unsqueeze(0)
    signature = hop_periodicity_signature(carrier, hop_length=hop_length)
    assert signature.shape == (1, 2)
    assert torch.isfinite(signature).all()
    assert signature[0, 0] > 0.0
