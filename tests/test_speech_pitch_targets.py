from __future__ import annotations

import math

import torch

from lykenox_voice_engine.training.speech_pitch import (
    PITCH_TARGET_VERSION,
    extract_pitch_frames,
)
from lykenox_voice_engine.training.speech_pitch_cache import (
    PITCH_CACHE_VERSION,
    PITCH_CONFIG,
)


def _sine(samples: int, *, sample_rate: int = 24000, f0_hz: float = 100.0) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float32) / float(sample_rate)
    return 0.4 * torch.sin(2.0 * math.pi * f0_hz * time)


def test_pitch_target_accepts_exact_hop_vocoder_crop() -> None:
    hop = 256
    frame_count = 64
    waveform = _sine(frame_count * hop)
    result = extract_pitch_frames(waveform, frame_count=frame_count, hop_length=hop)
    assert result.f0_hz.shape == (frame_count,)
    assert result.voiced.shape == (frame_count,)
    assert result.periodicity.shape == (frame_count,)


def test_pitch_target_matches_centered_full_utterance_frame_count() -> None:
    hop = 256
    samples = 63 * hop + 137
    waveform = _sine(samples)
    centered_frames = samples // hop + 1
    result = extract_pitch_frames(
        waveform,
        frame_count=centered_frames,
        hop_length=hop,
    )
    assert result.f0_hz.shape == (centered_frames,)
    assert result.voiced.shape == (centered_frames,)
    assert result.periodicity.shape == (centered_frames,)


def test_pitch_target_unvoiced_f0_is_zero_and_voiced_range_is_bounded() -> None:
    hop = 256
    waveform = _sine(80 * hop, f0_hz=100.0)
    result = extract_pitch_frames(waveform, frame_count=80, hop_length=hop)
    assert torch.all(result.f0_hz[result.voiced < 0.5] == 0.0)
    voiced = result.f0_hz[result.voiced > 0.5]
    assert voiced.numel() > 0
    assert float(voiced.min()) >= float(PITCH_CONFIG["min_f0_hz"])
    assert float(voiced.max()) <= float(PITCH_CONFIG["max_f0_hz"])


def test_pitch_cache_versions_are_explicit() -> None:
    assert PITCH_TARGET_VERSION == "lykenox-pitch-v1"
    assert PITCH_CACHE_VERSION == "speech-pitch-cache-v1"
