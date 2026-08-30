"""Target-relative mel fidelity objective for isolated acoustic mel refinement.

This loss is intentionally upstream-only.  It operates on cached log-mel targets and never
changes duration, F0, voicing, waveform level, EQ, or vocoder behavior.  The accepted
acoustic-v2 audit showed strong smoothing (spectral/temporal delta ratios well below one)
and increasing underpresence toward 1-8 kHz, so this objective adds shape/motion fidelity
and an asymmetric target-relative clarity guard while keeping ordinary mel L1 as authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import torchaudio


ACOUSTIC_MEL_FIDELITY_LOSS_VERSION = "acoustic-mel-fidelity-loss-v1"
BANDS_HZ = (
    ("80_300", 80.0, 300.0),
    ("300_1000", 300.0, 1000.0),
    ("1k_3k", 1000.0, 3000.0),
    ("3k_8k", 3000.0, 8000.0),
)
CLARITY_BAND_WEIGHTS = (0.0, 0.25, 1.0, 1.5)
DEFAULT_CENTERED_SHAPE_WEIGHT = 0.50
DEFAULT_SPECTRAL_DELTA_WEIGHT = 0.25
DEFAULT_TEMPORAL_DELTA_WEIGHT = 0.25
DEFAULT_CLARITY_GUARD_WEIGHT = 0.25


@dataclass(frozen=True)
class AcousticMelFidelityLossResult:
    total: torch.Tensor
    mel_l1: torch.Tensor
    centered_shape: torch.Tensor
    spectral_delta: torch.Tensor
    temporal_delta: torch.Tensor
    clarity_underpresence: torch.Tensor


def mel_bin_centers_hz(*, sample_rate: int, n_fft: int, mel_bins: int) -> torch.Tensor:
    fbanks = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=float(sample_rate) / 2.0,
        n_mels=mel_bins,
        sample_rate=sample_rate,
        norm=None,
        mel_scale="htk",
    )
    fft_hz = torch.linspace(0.0, float(sample_rate) / 2.0, n_fft // 2 + 1)
    return fft_hz[torch.argmax(fbanks, dim=0)]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    denominator = weight.sum().clamp_min(1.0)
    return (values * weight).sum() / denominator


def _band_log_mean_linear(
    mel: torch.Tensor,
    frame_mask: torch.Tensor,
    bin_mask: torch.Tensor,
) -> torch.Tensor:
    selected = mel[..., bin_mask]
    expanded_mask = frame_mask.unsqueeze(-1).expand_as(selected)
    counts = expanded_mask.sum(dim=(1, 2)).clamp_min(1)
    masked = selected.masked_fill(~expanded_mask, float("-inf"))
    log_sum = torch.logsumexp(masked.flatten(1), dim=1)
    return log_sum - torch.log(counts.to(mel.dtype))


def acoustic_mel_fidelity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    centered_shape_weight: float = DEFAULT_CENTERED_SHAPE_WEIGHT,
    spectral_delta_weight: float = DEFAULT_SPECTRAL_DELTA_WEIGHT,
    temporal_delta_weight: float = DEFAULT_TEMPORAL_DELTA_WEIGHT,
    clarity_guard_weight: float = DEFAULT_CLARITY_GUARD_WEIGHT,
) -> AcousticMelFidelityLossResult:
    """Measure target-relative mel fidelity on real frames only.

    The clarity guard only penalizes band underpresence.  It does not reward exceeding the
    target; ordinary L1 and centered-shape terms remain symmetric and therefore constrain
    overshoot.  The lowest band has zero clarity weight so the objective cannot improve by
    adding more low-frequency energy.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [batch, frames, mel_bins] shape")
    if frame_mask.shape != prediction.shape[:2]:
        raise ValueError("frame_mask must match [batch, frames]")
    if min(
        centered_shape_weight,
        spectral_delta_weight,
        temporal_delta_weight,
        clarity_guard_weight,
    ) < 0.0:
        raise ValueError("mel fidelity weights must be non-negative")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("mel tensors must be finite")

    frame_mask = frame_mask.bool()
    value_mask = frame_mask.unsqueeze(-1).expand_as(prediction)
    mel_l1 = _masked_mean(torch.abs(prediction - target), value_mask)

    pred_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    centered_shape = _masked_mean(
        torch.abs(pred_centered - target_centered), value_mask
    )

    if prediction.shape[-1] > 1:
        spectral_mask = frame_mask.unsqueeze(-1).expand(
            -1, -1, prediction.shape[-1] - 1
        )
        pred_spectral = prediction[..., 1:] - prediction[..., :-1]
        target_spectral = target[..., 1:] - target[..., :-1]
        spectral_delta = _masked_mean(
            torch.abs(pred_spectral - target_spectral), spectral_mask
        )
    else:
        spectral_delta = prediction.sum() * 0.0

    if prediction.shape[1] > 1:
        pair_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
        temporal_mask = pair_mask.unsqueeze(-1).expand(
            -1, -1, prediction.shape[-1]
        )
        pred_temporal = prediction[:, 1:] - prediction[:, :-1]
        target_temporal = target[:, 1:] - target[:, :-1]
        temporal_delta = _masked_mean(
            torch.abs(pred_temporal - target_temporal), temporal_mask
        )
    else:
        temporal_delta = prediction.sum() * 0.0

    centers = mel_bin_centers_hz(
        sample_rate=sample_rate,
        n_fft=n_fft,
        mel_bins=prediction.shape[-1],
    ).to(prediction.device)
    guard_terms: list[torch.Tensor] = []
    guard_weights: list[float] = []
    for (name, low, high), weight in zip(BANDS_HZ, CLARITY_BAND_WEIGHTS, strict=True):
        del name
        if weight <= 0.0:
            continue
        bins = (centers >= low) & (centers < high)
        if not bool(bins.any()):
            raise RuntimeError(f"clarity band {low}-{high} Hz has no mel bins")
        pred_log_mean = _band_log_mean_linear(prediction, frame_mask, bins)
        target_log_mean = _band_log_mean_linear(target, frame_mask, bins)
        under = torch.relu(target_log_mean - pred_log_mean)
        guard_terms.append(F.smooth_l1_loss(under, torch.zeros_like(under)))
        guard_weights.append(float(weight))
    if guard_terms:
        weight_sum = max(sum(guard_weights), 1e-12)
        clarity_underpresence = sum(
            term * weight for term, weight in zip(guard_terms, guard_weights, strict=True)
        ) / weight_sum
    else:
        clarity_underpresence = prediction.sum() * 0.0

    total = (
        mel_l1
        + centered_shape_weight * centered_shape
        + spectral_delta_weight * spectral_delta
        + temporal_delta_weight * temporal_delta
        + clarity_guard_weight * clarity_underpresence
    )
    return AcousticMelFidelityLossResult(
        total=total,
        mel_l1=mel_l1,
        centered_shape=centered_shape,
        spectral_delta=spectral_delta,
        temporal_delta=temporal_delta,
        clarity_underpresence=clarity_underpresence,
    )
