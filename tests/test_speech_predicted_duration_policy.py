from __future__ import annotations

import torch

from lykenox_voice_engine.core.spanish_g2p import TOKEN_TO_ID
from lykenox_voice_engine.models.speech.duration_policy import (
    PAUSE_MAX_DURATION_FRAMES,
    regulate_predicted_durations,
)


def test_predicted_duration_policy_is_token_aware() -> None:
    token_ids = torch.tensor(
        [[
            TOKEN_TO_ID["<pad>"],
            TOKEN_TO_ID["<bos>"],
            TOKEN_TO_ID["<eos>"],
            TOKEN_TO_ID["<wb>"],
            TOKEN_TO_ID["<pau_short>"],
            TOKEN_TO_ID["<pau_long>"],
            TOKEN_TO_ID["a"],
            TOKEN_TO_ID["e"],
        ]],
        dtype=torch.long,
    )
    mask = torch.tensor([[False, True, True, True, True, True, True, True]])
    raw = torch.tensor([[200.0, 0.1, 1.6, 0.49, 0.1, 500.0, 0.1, 120.0]])

    actual = regulate_predicted_durations(token_ids, mask, raw)
    expected = torch.tensor(
        [[0, 0, 2, 0, 1, PAUSE_MAX_DURATION_FRAMES, 1, 120]],
        dtype=torch.long,
    )
    assert torch.equal(actual, expected)


def test_predicted_duration_policy_rejects_negative_values() -> None:
    token_ids = torch.tensor([[TOKEN_TO_ID["a"]]], dtype=torch.long)
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    raw = torch.tensor([[-0.1]], dtype=torch.float32)

    try:
        regulate_predicted_durations(token_ids, mask, raw)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative predicted duration must be rejected")
