from __future__ import annotations

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_stream_source_v1 import (
    BLOCK_SAMPLES,
    CONTINUOUS_STREAM_SOURCE_ARCHITECTURE,
    LykenoxContinuousResidualStreamSourceV1,
    blocks_to_residual,
)


def test_blocks_to_residual_is_exact_unique_concatenation():
    blocks = torch.arange(3 * BLOCK_SAMPLES, dtype=torch.float32).view(1, 3, BLOCK_SAMPLES)
    residual = blocks_to_residual(blocks)
    assert residual.shape == (1, 3 * BLOCK_SAMPLES)
    assert torch.equal(residual, blocks.reshape(1, -1))


def test_changing_one_block_changes_only_its_owned_samples():
    base = torch.zeros(1, 4, BLOCK_SAMPLES)
    changed = base.clone()
    changed[:, 2] = 1.0
    delta = blocks_to_residual(changed) - blocks_to_residual(base)
    assert torch.count_nonzero(delta[:, : 2 * BLOCK_SAMPLES]) == 0
    assert torch.count_nonzero(delta[:, 2 * BLOCK_SAMPLES : 3 * BLOCK_SAMPLES]) == BLOCK_SAMPLES
    assert torch.count_nonzero(delta[:, 3 * BLOCK_SAMPLES :]) == 0


def test_predicted_block_rms_matches_explicit_level():
    torch.manual_seed(7)
    model = LykenoxContinuousResidualStreamSourceV1().eval()
    frames = 5
    mel = torch.randn(1, frames, 80)
    f0 = torch.full((1, frames), 120.0)
    energy = torch.full((1, frames), 0.8)
    periodic = torch.full((1, frames), 0.4)
    with torch.no_grad():
        blocks, log_rms = model.forward_with_log_rms(mel, f0, energy, periodic)
    measured = torch.sqrt(blocks.square().mean(dim=-1))
    assert torch.allclose(measured, torch.exp(log_rms), rtol=2e-4, atol=2e-6)


def test_architecture_has_no_overlap_or_previous_waveform_input():
    model = LykenoxContinuousResidualStreamSourceV1()
    assert CONTINUOUS_STREAM_SOURCE_ARCHITECTURE == "lykenox_owned_continuous_residual_stream_source_v1"
    assert BLOCK_SAMPLES == 256
    assert model.recurrent.input_size == model.context_channels
    assert not hasattr(model, "previous_projection")
    assert not hasattr(model, "previous_block_projection")
