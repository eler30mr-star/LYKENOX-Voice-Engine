import torch

from lykenox_voice_engine.training.speech_vocoder_residual_statistics_source_train_v1 import (
    _high_band_flatness,
)


def test_high_band_flatness_indexes_frequency_axis_for_1d_waveform() -> None:
    torch.manual_seed(20260903)
    waveform = torch.randn(4096, dtype=torch.float32)
    value = _high_band_flatness(waveform)
    assert value.ndim == 0
    assert torch.isfinite(value)
    assert 0.0 <= float(value) <= 1.0


def test_high_band_flatness_indexes_frequency_axis_for_batched_waveform() -> None:
    torch.manual_seed(20260904)
    waveform = torch.randn(2, 4096, dtype=torch.float32)
    value = _high_band_flatness(waveform)
    assert value.ndim == 0
    assert torch.isfinite(value)
    assert 0.0 <= float(value) <= 1.0
