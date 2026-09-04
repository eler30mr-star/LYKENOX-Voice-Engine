from __future__ import annotations

import torch

from scripts.diagnostic_target_phase_magnitude_level_shape_v1 import (
    _compose_magnitude,
    _decompose_log_magnitude,
)


def test_log_magnitude_level_shape_roundtrip_is_exact() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260904)
    magnitude = torch.rand((513, 41), generator=generator, dtype=torch.float32) * 2.0 + 1.0e-3
    level, shape = _decompose_log_magnitude(magnitude)
    rebuilt = _compose_magnitude(level, shape)
    assert level.shape == (1, 41)
    assert shape.shape == magnitude.shape
    assert torch.allclose(shape.mean(dim=0), torch.zeros(41), atol=2.0e-6, rtol=0.0)
    assert torch.allclose(rebuilt, magnitude, atol=3.0e-6, rtol=3.0e-6)


def test_crossed_level_shape_hybrids_preserve_selected_coordinate() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260905)
    target = torch.rand((513, 31), generator=generator, dtype=torch.float32) + 0.02
    candidate = torch.rand((513, 31), generator=generator, dtype=torch.float32) + 0.02
    target_level, target_shape = _decompose_log_magnitude(target)
    candidate_level, candidate_shape = _decompose_log_magnitude(candidate)

    candidate_shape_target_level = _compose_magnitude(target_level, candidate_shape)
    target_shape_candidate_level = _compose_magnitude(candidate_level, target_shape)

    hybrid1_level, hybrid1_shape = _decompose_log_magnitude(candidate_shape_target_level)
    hybrid2_level, hybrid2_shape = _decompose_log_magnitude(target_shape_candidate_level)

    assert torch.allclose(hybrid1_level, target_level, atol=2.0e-6, rtol=0.0)
    assert torch.allclose(hybrid1_shape, candidate_shape, atol=2.0e-6, rtol=0.0)
    assert torch.allclose(hybrid2_level, candidate_level, atol=2.0e-6, rtol=0.0)
    assert torch.allclose(hybrid2_shape, target_shape, atol=2.0e-6, rtol=0.0)
