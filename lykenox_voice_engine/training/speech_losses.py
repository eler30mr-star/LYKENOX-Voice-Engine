"""Mask-aware losses for padded LYKENOX Speech batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpeechLosses:
    total: torch.Tensor
    acoustic: torch.Tensor
    duration: torch.Tensor


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 over real mel frames only; padded frames contribute exactly zero."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target mel tensors must have identical shapes")
    if prediction.ndim != 3:
        raise ValueError("mel tensors must have shape [batch, frames, mel_bins]")
    if frame_mask.shape != prediction.shape[:2]:
        raise ValueError("frame_mask must match [batch, frames]")

    mask = frame_mask.unsqueeze(-1).to(prediction.dtype)
    absolute = torch.abs(prediction - target) * mask
    denominator = torch.clamp(
        frame_mask.sum().to(prediction.dtype) * prediction.shape[-1],
        min=1.0,
    )
    return absolute.sum() / denominator


def masked_log_duration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 in log-duration space over real text tokens only."""

    if prediction.shape != target.shape:
        raise ValueError("duration prediction and target must have identical shapes")
    if token_mask.shape != prediction.shape:
        raise ValueError("token_mask must match duration tensors")

    elementwise = F.smooth_l1_loss(
        torch.log1p(prediction),
        torch.log1p(target.to(prediction.dtype)),
        reduction="none",
    )
    mask = token_mask.to(prediction.dtype)
    denominator = torch.clamp(mask.sum(), min=1.0)
    return (elementwise * mask).sum() / denominator


def speech_training_losses(
    *,
    mel_prediction: torch.Tensor,
    mel_target: torch.Tensor,
    mel_mask: torch.Tensor,
    duration_prediction: torch.Tensor,
    duration_target: torch.Tensor,
    token_mask: torch.Tensor,
    duration_weight: float = 0.10,
) -> SpeechLosses:
    """Canonical current acoustic-training loss contract."""

    if duration_weight < 0:
        raise ValueError("duration_weight must be non-negative")
    acoustic = masked_l1_loss(mel_prediction, mel_target, mel_mask)
    duration = masked_log_duration_loss(
        duration_prediction,
        duration_target,
        token_mask,
    )
    total = acoustic + duration_weight * duration
    return SpeechLosses(total=total, acoustic=acoustic, duration=duration)
