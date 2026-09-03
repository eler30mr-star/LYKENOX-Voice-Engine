from __future__ import annotations

import torch

from lykenox_voice_engine.training.speech_vocoder_residual_statistics_source_train_v1 import (
    _high_band_flatness,
)


def test_high_band_flatness_indexes_stft_frequency_axis_for_mono_signal():
    torch.manual_seed(20260903)
    signal = torch.randn(8192, dtype=torch.float32)
    value = _high_band_flatness(signal)
    assert value.ndim == 0
    assert torch.isfinite(value)
    assert 0.0 <= float(value) <= 1.0


def test_high_band_flatness_supports_batched_stft_geometry():
    torch.manual_seed(20260904)
    signal = torch.randn(2, 8192, dtype=torch.float32)
    value = _high_band_flatness(signal)
    assert value.ndim == 0
    assert torch.isfinite(value)
    assert 0.0 <= float(value) <= 1.0
