from __future__ import annotations

import torch

from scripts.diagnostic_residual_codebook_oracle_v3_sequence_coherent import (
    OVERLAP_WINDOW_FLOOR,
    _paired_overlap_unwindowed,
    _transition_cost,
)
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    HOP_LENGTH,
    _sqrt_hann,
    residual_analysis_vectors,
)


def test_periodic_sqrt_hann_independent_floor_masks_are_asymmetric() -> None:
    window = _sqrt_hann(dtype=torch.float32)
    right_mask = window[HOP_LENGTH:] >= OVERLAP_WINDOW_FLOOR
    left_mask = window[:HOP_LENGTH] >= OVERLAP_WINDOW_FLOOR

    # Regression for the locally observed v3 failure: applying the floor independently can retain
    # a different number of samples on the two halves of a periodic Hann window. The production
    # transition code must therefore use a shared physical-position mask, not independent masks.
    assert int(right_mask.numel()) == HOP_LENGTH
    assert int(left_mask.numel()) == HOP_LENGTH
    assert int(right_mask.sum()) != int(left_mask.sum())


def test_paired_overlap_uses_identical_geometry() -> None:
    previous = torch.randn(12, 2 * HOP_LENGTH, generator=torch.Generator().manual_seed(1))
    current = torch.randn(12, 2 * HOP_LENGTH, generator=torch.Generator().manual_seed(2))

    previous_overlap, current_overlap = _paired_overlap_unwindowed(previous, current)

    assert previous_overlap.ndim == 2
    assert current_overlap.ndim == 2
    assert previous_overlap.shape[0] == 12
    assert current_overlap.shape[0] == 12
    assert previous_overlap.shape[1] == current_overlap.shape[1]
    assert previous_overlap.shape[1] >= HOP_LENGTH // 2

    cost = _transition_cost(previous, current)
    assert cost.shape == (12, 12)
    assert bool(torch.isfinite(cost).all())


def test_true_neighboring_residual_windows_have_near_zero_transition_cost() -> None:
    generator = torch.Generator().manual_seed(3)
    residual = torch.randn(8 * HOP_LENGTH, generator=generator, dtype=torch.float32)
    vectors = residual_analysis_vectors(residual)

    # Each pair below is made from consecutive windows of the exact same residual trajectory.
    for index in range(int(vectors.shape[0]) - 1):
        cost = _transition_cost(vectors[index : index + 1], vectors[index + 1 : index + 2])
        assert cost.shape == (1, 1)
        assert float(cost[0, 0]) < 1.0e-10
