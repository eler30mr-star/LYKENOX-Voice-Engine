from __future__ import annotations

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v2 import (
    LykenoxContinuousResidualSourceV2,
)


def _conditioning(frames: int = 6) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(17)
    mel = torch.randn(1, frames, 80, dtype=torch.float32) * 0.2
    f0 = torch.full((1, frames), 140.0, dtype=torch.float32)
    voiced = torch.ones(1, frames, dtype=torch.float32)
    periodicity = torch.full((1, frames), 0.8, dtype=torch.float32)
    return mel, f0, voiced, periodicity


def test_vector_rms_matches_explicit_predicted_log_rms() -> None:
    torch.manual_seed(23)
    model = LykenoxContinuousResidualSourceV2(
        context_channels=64,
        state_channels=64,
        previous_embed=32,
        dilations=(1, 2),
    )
    vectors, log_rms = model.forward_with_log_rms(*_conditioning())
    measured = torch.sqrt(vectors.square().mean(dim=-1).clamp_min(1.0e-12))
    expected = torch.exp(log_rms)
    assert vectors.shape == (1, 7, 512)
    assert log_rms.shape == (1, 7)
    assert torch.allclose(measured, expected, rtol=2.0e-4, atol=2.0e-6)


def test_level_head_has_direct_nonzero_gradient() -> None:
    torch.manual_seed(29)
    model = LykenoxContinuousResidualSourceV2(
        context_channels=64,
        state_channels=64,
        previous_embed=32,
        dilations=(1,),
    )
    _, log_rms = model.forward_with_log_rms(*_conditioning(frames=4))
    loss = (log_rms + 2.0).square().mean()
    loss.backward()
    weight_grad = model.level_projection.weight.grad
    bias_grad = model.level_projection.bias.grad
    assert weight_grad is not None and torch.isfinite(weight_grad).all()
    assert bias_grad is not None and torch.isfinite(bias_grad).all()
    assert float(weight_grad.abs().sum() + bias_grad.abs().sum()) > 0.0


def test_previous_amplitude_cannot_feed_back_into_recurrent_shape_path() -> None:
    torch.manual_seed(31)
    model = LykenoxContinuousResidualSourceV2(
        context_channels=64,
        state_channels=64,
        previous_embed=32,
        dilations=(1,),
    )
    conditioning = _conditioning(frames=5)
    teacher = torch.randn(1, 6, 512, dtype=torch.float32)
    scaled_teacher = teacher * 9.0
    first_vectors, first_levels = model.forward_with_log_rms(
        *conditioning,
        teacher_vectors=teacher,
        teacher_forcing_ratio=1.0,
        teacher_seed=7,
    )
    second_vectors, second_levels = model.forward_with_log_rms(
        *conditioning,
        teacher_vectors=scaled_teacher,
        teacher_forcing_ratio=1.0,
        teacher_seed=7,
    )
    assert torch.allclose(first_vectors, second_vectors, rtol=1.0e-5, atol=1.0e-6)
    assert torch.allclose(first_levels, second_levels, rtol=1.0e-5, atol=1.0e-6)
