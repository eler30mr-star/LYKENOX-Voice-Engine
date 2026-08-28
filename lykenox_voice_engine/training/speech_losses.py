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
    f0: torch.Tensor | None = None
    voicing: torch.Tensor | None = None


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


def masked_log_f0_loss(
    prediction_hz: torch.Tensor,
    target_hz: torch.Tensor,
    voiced_target: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 in log-Hz space on target-voiced real frames only."""

    if prediction_hz.shape != target_hz.shape:
        raise ValueError("F0 prediction and target must have identical shapes")
    if voiced_target.shape != prediction_hz.shape:
        raise ValueError("voiced_target must match F0 tensors")
    if frame_mask.shape != prediction_hz.shape:
        raise ValueError("frame_mask must match F0 tensors")
    if not bool((prediction_hz >= 0.0).all()):
        raise ValueError("F0 prediction must be non-negative")
    if not bool((target_hz >= 0.0).all()):
        raise ValueError("F0 target must be non-negative")

    valid = frame_mask.bool() & (voiced_target > 0.5)
    if not bool(valid.any()):
        return prediction_hz.sum() * 0.0
    pred_log = torch.log(torch.clamp(prediction_hz[valid], min=1e-6))
    target_log = torch.log(torch.clamp(target_hz[valid], min=1e-6))
    return F.smooth_l1_loss(pred_log, target_log)


def masked_voicing_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary voiced/unvoiced BCE over real mel frames only."""

    if logits.shape != target.shape:
        raise ValueError("voicing logits and target must have identical shapes")
    if frame_mask.shape != logits.shape:
        raise ValueError("frame_mask must match voicing tensors")
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        target.to(logits.dtype),
        reduction="none",
    )
    mask = frame_mask.to(logits.dtype)
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
    f0_prediction_hz: torch.Tensor | None = None,
    f0_target_hz: torch.Tensor | None = None,
    voicing_logits: torch.Tensor | None = None,
    voicing_target: torch.Tensor | None = None,
    f0_weight: float = 0.0,
    voicing_weight: float = 0.0,
) -> SpeechLosses:
    """Canonical acoustic-training loss contract, optionally including prosody heads."""

    if duration_weight < 0 or f0_weight < 0 or voicing_weight < 0:
        raise ValueError("speech loss weights must be non-negative")
    acoustic = masked_l1_loss(mel_prediction, mel_target, mel_mask)
    duration = masked_log_duration_loss(
        duration_prediction,
        duration_target,
        token_mask,
    )
    total = acoustic + duration_weight * duration

    f0_loss: torch.Tensor | None = None
    voicing_loss: torch.Tensor | None = None
    if f0_weight > 0.0:
        if f0_prediction_hz is None or f0_target_hz is None or voicing_target is None:
            raise ValueError("F0 supervision tensors are required when f0_weight > 0")
        f0_loss = masked_log_f0_loss(
            f0_prediction_hz,
            f0_target_hz,
            voicing_target,
            mel_mask,
        )
        total = total + f0_weight * f0_loss
    if voicing_weight > 0.0:
        if voicing_logits is None or voicing_target is None:
            raise ValueError("voicing supervision tensors are required when voicing_weight > 0")
        voicing_loss = masked_voicing_loss(
            voicing_logits,
            voicing_target,
            mel_mask,
        )
        total = total + voicing_weight * voicing_loss

    return SpeechLosses(
        total=total,
        acoustic=acoustic,
        duration=duration,
        f0=f0_loss,
        voicing=voicing_loss,
    )
