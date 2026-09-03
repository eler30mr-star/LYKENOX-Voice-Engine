from __future__ import annotations

import torch

from scripts.render_pitch_synchronous_cycle_source_v3_phase_exclusive_handoff import (
    _apply_c1_bridge_inplace,
    _coverage_runs,
    synthesize_phase_exclusive_handoff,
)


def test_coverage_runs_find_pitch_sync_authority_islands():
    coverage = torch.zeros(40)
    coverage[5:12] = 1.0
    coverage[20:31] = 1.0
    assert _coverage_runs(coverage) == [(5, 12), (20, 31)]


def test_phase_exclusive_handoff_never_raw_mix_away_from_bridge():
    samples = 64
    pitch = torch.ones(samples)
    v2 = torch.full((samples,), -1.0)
    coverage = torch.zeros(samples)
    coverage[16:48] = 1.0
    boundaries = [
        (16, 32, 0, 0.9),
        (32, 48, 1, 0.9),
    ]
    result, handoffs = synthesize_phase_exclusive_handoff(pitch, coverage, v2, boundaries)
    assert len(handoffs) == 2
    assert torch.all(result[:10] == -1.0)
    assert torch.all(result[22:42] == 1.0)
    assert torch.all(result[54:] == -1.0)


def test_c1_bridge_preserves_anchor_value_and_local_slope():
    residual = torch.zeros(80)
    residual[:40] = torch.linspace(-0.4, 0.2, 40)
    residual[40:] = torch.linspace(0.7, -0.1, 40)
    before = residual.clone()
    bridge = _apply_c1_bridge_inplace(residual, edge=40, half_width=8)
    assert bridge is not None
    start, stop = bridge
    end = stop - 1
    assert torch.allclose(residual[start], before[start])
    assert torch.allclose(residual[end], before[end])
    left_slope = residual[start] - residual[start - 1]
    right_slope = residual[end + 1] - residual[end]
    assert torch.allclose(left_slope, before[start] - before[start - 1], atol=1.0e-6)
    assert torch.allclose(right_slope, before[end + 1] - before[end], atol=1.0e-6)


def test_handoff_does_not_change_samples_when_no_pitch_sync_coverage():
    v2 = torch.randn(96)
    pitch = torch.randn(96)
    coverage = torch.zeros(96)
    result, handoffs = synthesize_phase_exclusive_handoff(pitch, coverage, v2, [])
    assert handoffs == []
    assert torch.equal(result, v2)
