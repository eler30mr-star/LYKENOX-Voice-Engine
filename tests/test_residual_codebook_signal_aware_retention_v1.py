from __future__ import annotations

import torch

from scripts.diagnostic_residual_codebook_signal_aware_retention_v1 import (
    DESCRIPTOR_FEATURES,
    DESCRIPTOR_QUANTILE_BINS,
    _descriptor_features,
    _rank_quantile_cells,
    _select_signal_aware_indices,
)


def _vectors(count: int, samples: int = 512) -> torch.Tensor:
    time = torch.linspace(0.0, 1.0, samples, dtype=torch.float32)
    rows = []
    for index in range(count):
        frequency = 3.0 + float(index % 17)
        phase = 0.07 * float(index)
        carrier = torch.sin(2.0 * torch.pi * frequency * time + phase)
        modulation = 0.25 * torch.sin(2.0 * torch.pi * (frequency * 0.37 + 1.0) * time)
        tilt = (float((index % 9) - 4) / 9.0) * (time - 0.5)
        rows.append((carrier + modulation + tilt).to(torch.float32))
    return torch.stack(rows, dim=0)


def test_signal_descriptor_is_finite_and_has_fixed_geometry() -> None:
    vectors = _vectors(12)
    features = _descriptor_features(vectors)
    assert features.shape == (12, len(DESCRIPTOR_FEATURES))
    assert torch.isfinite(features).all()


def test_rank_quantile_cells_are_deterministic_and_bounded() -> None:
    features = _descriptor_features(_vectors(31))
    first = _rank_quantile_cells(features)
    second = _rank_quantile_cells(features.clone())
    assert torch.equal(first, second)
    assert int(first.min()) >= 0
    assert int(first.max()) < DESCRIPTOR_QUANTILE_BINS ** len(DESCRIPTOR_FEATURES)


def test_signal_aware_selection_is_unique_deterministic_and_exact_budget() -> None:
    vectors = _vectors(40)
    first = _select_signal_aware_indices(vectors, source_start_index=1000, cap=17)
    second = _select_signal_aware_indices(vectors.clone(), source_start_index=1000, cap=17)
    assert torch.equal(first, second)
    assert first.shape == (17,)
    assert int(torch.unique(first).numel()) == 17
    assert int(first.min()) >= 0
    assert int(first.max()) < 40


def test_signal_aware_selection_keeps_all_when_bucket_is_under_budget() -> None:
    vectors = _vectors(9)
    selected = _select_signal_aware_indices(vectors, source_start_index=0, cap=16)
    assert torch.equal(selected, torch.arange(9, dtype=torch.long))
