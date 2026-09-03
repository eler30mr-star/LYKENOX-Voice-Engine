from __future__ import annotations

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_residual_statistics_source_v1 import (
    RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE,
    SOURCE_CEPSTRAL_ORDER,
    LykenoxResidualStatisticsSourceV1,
)
from lykenox_voice_engine.training.speech_vocoder_residual_statistics_source_train_v1 import (
    _continuous_noise,
    synthesize_residual_from_statistics,
)


def test_statistics_model_never_emits_waveform_blocks():
    model = LykenoxResidualStatisticsSourceV1()
    assert model.architecture == RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE
    assert model.source_cepstrum_projection.out_features == SOURCE_CEPSTRAL_ORDER - 1
    forbidden = {256, 512}
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.out_features not in forbidden


def test_forward_returns_only_frame_statistics():
    model = LykenoxResidualStatisticsSourceV1()
    frames = 5
    mel = torch.zeros(1, frames, 80)
    f0 = torch.full((1, frames), 120.0)
    energy = torch.full((1, frames), 0.8)
    periodic = torch.full((1, frames), 0.2)
    cepstrum, log_rms, source_periodicity = model(mel, f0, energy, periodic)
    assert cepstrum.shape == (1, frames, SOURCE_CEPSTRAL_ORDER)
    assert log_rms.shape == (1, frames)
    assert source_periodicity.shape == (1, frames)
    assert torch.equal(cepstrum[..., 0], torch.zeros_like(cepstrum[..., 0]))


def test_continuous_noise_does_not_repeat_at_one_acoustic_hop():
    noise = _continuous_noise(4096, seed=17, dtype=torch.float32, device=torch.device("cpu"))
    left = noise[:-256].to(torch.float64)
    right = noise[256:].to(torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    corr = (left * right).sum() / torch.sqrt(left.square().sum() * right.square().sum())
    assert abs(float(corr)) < 0.1


def test_statistics_synthesizer_has_exact_frame_times_hop_length():
    frames = 4
    cepstrum = torch.zeros(1, frames, SOURCE_CEPSTRAL_ORDER)
    log_rms = torch.full((1, frames), -2.0)
    source_periodicity = torch.zeros(1, frames)
    f0 = torch.full((1, frames), 120.0)
    residual = synthesize_residual_from_statistics(
        cepstrum,
        log_rms,
        source_periodicity,
        f0,
        seed=19,
    )
    assert residual.shape == (frames * 256,)
    assert torch.isfinite(residual).all()
