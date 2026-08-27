from __future__ import annotations

import torch
from torch import nn

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV4
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames


def test_v4_exact_waveform_length_and_no_learned_temporal_upsampler() -> None:
    generator = LykenoxVocoderGeneratorV4()
    mel = torch.randn(2, 12, generator.config.mel_bins)
    f0 = torch.full((2, 12), 120.0)
    voiced = torch.ones(2, 12)
    waveform = generator(mel, f0, voiced)
    assert waveform.shape == (2, 12 * generator.config.hop_length)
    assert torch.isfinite(waveform).all()
    assert not any(isinstance(module, nn.ConvTranspose1d) for module in generator.modules())


def test_v4_pitch_source_changes_when_f0_changes() -> None:
    generator = LykenoxVocoderGeneratorV4()
    mel = torch.zeros(1, 8, generator.config.mel_bins)
    voiced = torch.ones(1, 8)
    low = generator(mel, torch.full((1, 8), 90.0), voiced)
    high = generator(mel, torch.full((1, 8), 180.0), voiced)
    assert not torch.equal(low, high)


def test_pitch_extractor_returns_exact_frame_contract() -> None:
    sample_rate = 24000
    hop = 256
    frames = 16
    samples = frames * hop
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = 0.5 * torch.sin(2.0 * torch.pi * 120.0 * time)
    pitch = extract_pitch_frames(
        waveform,
        frame_count=frames,
        sample_rate=sample_rate,
        hop_length=hop,
    )
    assert pitch.f0_hz.shape == (frames,)
    assert pitch.voiced.shape == (frames,)
    assert pitch.periodicity.shape == (frames,)
    voiced_f0 = pitch.f0_hz[pitch.voiced > 0.5]
    assert voiced_f0.numel() > 0
    assert abs(float(voiced_f0.median()) - 120.0) < 5.0
