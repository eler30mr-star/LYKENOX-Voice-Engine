from __future__ import annotations

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_coherent_innovation_source_v1 import (
    RESIDUAL_VECTOR_SAMPLES,
    LykenoxCoherentInnovationResidualSourceV1,
)


def _inputs(frames: int = 6):
    mel = torch.zeros(1, frames, 80)
    f0 = torch.full((1, frames), 140.0)
    voiced = torch.ones(1, frames)
    periodicity = torch.full((1, frames), 0.85)
    return mel, f0, voiced, periodicity


def test_generated_vector_rms_matches_explicit_log_rms():
    model = LykenoxCoherentInnovationResidualSourceV1().cpu()
    vectors, _, _, log_rms, _ = model.forward_with_components(*_inputs(), innovation_seed=17)
    rms = torch.sqrt(vectors.square().mean(dim=-1).clamp_min(1.0e-12))
    assert vectors.shape[-1] == RESIDUAL_VECTOR_SAMPLES
    assert torch.allclose(rms, torch.exp(log_rms), atol=2.0e-4, rtol=2.0e-4)


def test_noise_seed_changes_innovation_not_coherent_or_level_path():
    model = LykenoxCoherentInnovationResidualSourceV1().cpu()
    a = model.forward_with_components(*_inputs(), innovation_seed=11)
    b = model.forward_with_components(*_inputs(), innovation_seed=29)
    _, coherent_a, innovation_a, level_a, mix_a = a
    _, coherent_b, innovation_b, level_b, mix_b = b
    assert torch.allclose(coherent_a, coherent_b)
    assert torch.allclose(level_a, level_b)
    assert torch.allclose(mix_a, mix_b)
    assert not torch.allclose(innovation_a, innovation_b)


def test_innovation_heads_receive_direct_gradient():
    model = LykenoxCoherentInnovationResidualSourceV1().cpu()
    vectors, _, _, _, mix = model.forward_with_components(*_inputs(), innovation_seed=7)
    loss = vectors.square().mean() + mix.mean()
    loss.backward()
    mix_grad = model.innovation_mix_projection.weight.grad
    color_grad = model.innovation_color_projection.weight.grad
    assert mix_grad is not None and torch.isfinite(mix_grad).all()
    assert color_grad is not None and torch.isfinite(color_grad).all()


def test_innovation_is_not_recurrent_feedback():
    model = LykenoxCoherentInnovationResidualSourceV1().cpu()
    _, coherent_a, _, level_a, _ = model.forward_with_components(*_inputs(), innovation_seed=1)
    _, coherent_b, _, level_b, _ = model.forward_with_components(*_inputs(), innovation_seed=999)
    assert torch.allclose(coherent_a, coherent_b)
    assert torch.allclose(level_a, level_b)
