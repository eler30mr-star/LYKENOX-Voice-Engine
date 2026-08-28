"""Versioned predicted-duration semantics for LYKENOX Speech inference.

Training uses exact alignment-v3 teacher durations and must never pass through this
policy. Product inference has a different requirement: predicted durations need token-
aware lower bounds and a safety ceiling that does not reproduce the historical fixed
1..80 clamp.
"""

from __future__ import annotations

import torch

from lykenox_voice_engine.core.spanish_g2p import TOKEN_TO_ID


PREDICTED_DURATION_POLICY_VERSION = "predicted-duration-policy-v1"

# alignment-v3 may assign acoustic timing to BOS/EOS/WB, but those structural tokens are
# also legitimately zero-duration. Therefore inference must allow, not force, zero.
STRUCTURAL_ZERO_ALLOWED_TOKEN_IDS = (
    TOKEN_TO_ID["<bos>"],
    TOKEN_TO_ID["<eos>"],
    TOKEN_TO_ID["<wb>"],
)
PAUSE_TOKEN_IDS = (
    TOKEN_TO_ID["<pau_short>"],
    TOKEN_TO_ID["<pau_long>"],
)
PAD_TOKEN_ID = TOKEN_TO_ID["<pad>"]

# alignment-v3 observed valid non-pause durations above the old 80-frame ceiling. Keep a
# conservative content/structural safety bound comfortably above that observed range;
# punctuation pauses receive a larger ceiling because they can legitimately be longer.
CONTENT_MAX_DURATION_FRAMES = 160
STRUCTURAL_MAX_DURATION_FRAMES = 160
PAUSE_MAX_DURATION_FRAMES = 320


def regulate_predicted_durations(
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
    duration_prediction: torch.Tensor,
) -> torch.Tensor:
    """Convert raw non-negative duration predictions into inference frame counts.

    Semantics:
    - padding is always exactly zero;
    - BOS/EOS/WB may round to zero or retain predicted positive timing;
    - phonemes, unknown tokens, and explicit pause tokens get at least one frame;
    - valid non-pause durations are not clipped by the legacy 80-frame ceiling;
    - explicit pauses have a larger safety ceiling;
    - the operation is tensor-only and deterministic.
    """

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, text_steps]")
    if token_mask.shape != token_ids.shape:
        raise ValueError("token_mask must match token_ids")
    if duration_prediction.shape != token_ids.shape:
        raise ValueError("duration_prediction must match token_ids")
    if not torch.isfinite(duration_prediction).all():
        raise ValueError("duration_prediction contains non-finite values")
    if not bool((duration_prediction >= 0.0).all()):
        raise ValueError("duration_prediction must be non-negative")

    valid = token_mask.bool() & (token_ids != PAD_TOKEN_ID)
    rounded = torch.round(duration_prediction).to(torch.long)

    structural = torch.zeros_like(valid)
    for token_id in STRUCTURAL_ZERO_ALLOWED_TOKEN_IDS:
        structural = structural | (token_ids == token_id)

    pause = torch.zeros_like(valid)
    for token_id in PAUSE_TOKEN_IDS:
        pause = pause | (token_ids == token_id)

    minimum = torch.ones_like(rounded)
    minimum = torch.where(structural, torch.zeros_like(minimum), minimum)

    maximum = torch.full_like(rounded, CONTENT_MAX_DURATION_FRAMES)
    maximum = torch.where(
        structural,
        torch.full_like(maximum, STRUCTURAL_MAX_DURATION_FRAMES),
        maximum,
    )
    maximum = torch.where(
        pause,
        torch.full_like(maximum, PAUSE_MAX_DURATION_FRAMES),
        maximum,
    )

    regulated = torch.maximum(rounded, minimum)
    regulated = torch.minimum(regulated, maximum)
    return torch.where(valid, regulated, torch.zeros_like(regulated))
