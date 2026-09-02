from __future__ import annotations

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v1 import (
    RESIDUAL_VECTOR_SAMPLES,
    LykenoxContinuousResidualSourceV1,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import _ola_vectors


def _conditioning(batch: int = 1, frames: int = 8):
    mel = torch.zeros(batch, frames, 80, dtype=torch.float32)
    f0 = torch.full((batch, frames), 120.0, dtype=torch.float32)
    voiced = torch.ones(batch, frames, dtype=torch.float32)
    periodicity = torch.full((batch, frames), 0.8, dtype=torch.float32)
    return mel, f0, voiced, periodicity


def test_continuous_source_outputs_frame_plus_one_vectors() -> None:
    model = LykenoxContinuousResidualSourceV1().cpu()
    mel, f0, voiced, periodicity = _conditioning(frames=8)
    vectors = model.generate(mel, f0, voiced, periodicity)
    assert vectors.shape == (1, 9, RESIDUAL_VECTOR_SAMPLES)
    assert torch.isfinite(vectors).all()


def test_continuous_source_teacher_forcing_preserves_geometry() -> None:
    model = LykenoxContinuousResidualSourceV1().cpu()
    mel, f0, voiced, periodicity = _conditioning(frames=6)
    teacher = torch.randn(1, 7, RESIDUAL_VECTOR_SAMPLES) * 0.01
    vectors = model(
        mel,
        f0,
        voiced,
        periodicity,
        teacher_vectors=teacher,
        teacher_forcing_ratio=1.0,
        teacher_seed=7,
    )
    assert vectors.shape == teacher.shape


def test_differentiable_ola_backpropagates_to_vectors() -> None:
    vectors = torch.randn(1, 5, RESIDUAL_VECTOR_SAMPLES, requires_grad=True)
    residual = _ola_vectors(vectors, output_samples=4 * 256)
    loss = residual.square().mean()
    loss.backward()
    assert vectors.grad is not None
    assert torch.isfinite(vectors.grad).all()
    assert float(vectors.grad.abs().sum()) > 0.0


def test_free_running_previous_vector_affects_future_output() -> None:
    model = LykenoxContinuousResidualSourceV1().cpu()
    mel, f0, voiced, periodicity = _conditioning(frames=5)
    teacher_a = torch.zeros(1, 6, RESIDUAL_VECTOR_SAMPLES)
    teacher_b = teacher_a.clone()
    teacher_b[:, 0, 0] = 0.5
    # Force a nonzero projection so the architecture-level dependency is testable before training.
    with torch.no_grad():
        model.previous_projection.weight.fill_(1.0e-3)
        model.recurrent.weight_ih.fill_(1.0e-3)
        model.vector_projection.weight.fill_(1.0e-3)
    a = model(
        mel,
        f0,
        voiced,
        periodicity,
        teacher_vectors=teacher_a,
        teacher_forcing_ratio=1.0,
        teacher_seed=1,
    )
    b = model(
        mel,
        f0,
        voiced,
        periodicity,
        teacher_vectors=teacher_b,
        teacher_forcing_ratio=1.0,
        teacher_seed=1,
    )
    assert not torch.equal(a[:, 1:], b[:, 1:])
