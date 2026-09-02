from __future__ import annotations

import torch

from lykenox_voice_engine.models.vocoder.network_pitch_synchronous_residual_cycle_source_v1 import (
    CYCLE_PHASE_BINS,
    LykenoxPitchSynchronousResidualCycleSourceV1,
)


def _conditioning(frames: int = 12):
    mel = torch.zeros(1, frames, 80)
    f0 = torch.full((1, frames), 150.0)
    voiced = torch.ones(1, frames)
    periodicity = torch.full((1, frames), 0.9)
    return mel, f0, voiced, periodicity


def test_cycle_source_has_explicit_cycle_rms():
    model = LykenoxPitchSynchronousResidualCycleSourceV1().cpu()
    indices = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    cycles, log_rms = model(*_conditioning(), indices)
    assert cycles.shape == (5, CYCLE_PHASE_BINS)
    rms = torch.sqrt(cycles.square().mean(dim=-1).clamp_min(1.0e-12))
    assert torch.allclose(rms, torch.exp(log_rms), atol=2.0e-4, rtol=2.0e-4)


def test_cycle_source_cycle_head_receives_gradient():
    model = LykenoxPitchSynchronousResidualCycleSourceV1().cpu()
    indices = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    cycles, log_rms = model(*_conditioning(), indices)
    loss = cycles.square().mean() + log_rms.square().mean()
    loss.backward()
    shape_grad = model.cycle_shape_projection.weight.grad
    level_grad = model.cycle_level_projection.weight.grad
    assert shape_grad is not None and torch.isfinite(shape_grad).all()
    assert level_grad is not None and torch.isfinite(level_grad).all()


def test_cycle_generation_is_not_sample_rate_autoregressive():
    model = LykenoxPitchSynchronousResidualCycleSourceV1().cpu()
    assert model.previous_cycle_projection.in_features == CYCLE_PHASE_BINS
    assert model.cycle_shape_projection.out_features == CYCLE_PHASE_BINS
    assert CYCLE_PHASE_BINS < 512
