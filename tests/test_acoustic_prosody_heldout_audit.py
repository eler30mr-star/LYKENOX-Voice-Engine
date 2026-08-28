from __future__ import annotations

import torch

from lykenox_voice_engine.training.speech_acoustic_prosody_audit import (
    _classification_metrics,
    _token_internal_pair_mask,
)


def test_token_internal_pair_mask_respects_zero_duration_structural_tokens() -> None:
    # Frames: token0 owns [0,1], token1 owns none, token2 owns [2,3,4].
    durations = torch.tensor([2, 0, 3], dtype=torch.long)
    mask = _token_internal_pair_mask(durations, frame_count=5)
    assert torch.equal(
        mask,
        torch.tensor([True, False, True, True], dtype=torch.bool),
    )


def test_token_internal_pair_mask_allows_single_token_motion() -> None:
    mask = _token_internal_pair_mask(torch.tensor([4]), frame_count=4)
    assert torch.equal(mask, torch.tensor([True, True, True]))


def test_voicing_metrics_are_well_defined() -> None:
    metrics = _classification_metrics(tp=8, tn=7, fp=2, fn=3)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
