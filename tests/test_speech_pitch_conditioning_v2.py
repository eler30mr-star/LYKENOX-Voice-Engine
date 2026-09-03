from __future__ import annotations

import torch

from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import (
    _fill_continuous_f0,
    extract_pitch_conditioning_v2,
)


def test_continuous_f0_preserves_anchor_values_and_removes_zero_gap():
    anchor_f0 = torch.tensor([0.0, 100.0, 0.0, 0.0, 125.0, 0.0])
    anchors = anchor_f0 > 0.0
    track = _fill_continuous_f0(anchor_f0, anchors)
    assert track[1].item() == 100.0
    assert track[4].item() == 125.0
    assert torch.all(track > 0.0)
    assert 100.0 < float(track[2]) < float(track[3]) < 125.0


def test_low_energy_periodicity_loses_authority_smoothly():
    sample_rate = 24000
    duration = 8192
    t = torch.arange(duration, dtype=torch.float32) / float(sample_rate)
    strong = 0.8 * torch.sin(2.0 * torch.pi * 120.0 * t[:4096])
    weak = 0.01 * torch.sin(2.0 * torch.pi * 120.0 * t[4096:])
    waveform = torch.cat((strong, weak))
    frames = duration // 256
    result = extract_pitch_conditioning_v2(waveform, frame_count=frames)
    first = result.periodic_strength[: frames // 2].mean()
    second = result.periodic_strength[frames // 2 :].mean()
    assert float(first) > float(second)
    assert torch.all(result.periodic_strength <= result.raw_periodicity.clamp(0.0, 1.0) + 1.0e-6)


def test_public_contract_is_continuous_strength_not_binary_voiced_authority():
    sample_rate = 24000
    duration = 4096
    t = torch.arange(duration, dtype=torch.float32) / float(sample_rate)
    waveform = 0.3 * torch.sin(2.0 * torch.pi * 140.0 * t)
    result = extract_pitch_conditioning_v2(waveform, frame_count=duration // 256)
    assert result.f0_track_hz.shape == result.periodic_strength.shape
    assert result.energy_confidence.shape == result.periodic_strength.shape
    assert torch.all((result.periodic_strength >= 0.0) & (result.periodic_strength <= 1.0))
    assert torch.all((result.energy_confidence >= 0.0) & (result.energy_confidence <= 1.0))
