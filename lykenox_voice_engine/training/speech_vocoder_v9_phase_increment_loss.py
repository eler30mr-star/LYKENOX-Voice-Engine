"""Direct supervision for the V9 magnitude + phase-increment representation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


V9_PHASE_INCREMENT_LOSS_VERSION = "vocoder-v9-phase-increment-loss-v1"


@dataclass(frozen=True)
class V9PhaseIncrementLoss:
    total: torch.Tensor
    log_magnitude_l1: torch.Tensor
    phase_increment_circular: torch.Tensor
    waveform_l1: torch.Tensor


def v9_phase_increment_loss(
    predicted_magnitude: torch.Tensor,
    target_magnitude: torch.Tensor,
    predicted_residual_phase: torch.Tensor,
    target_residual_phase: torch.Tensor,
    predicted_waveform: torch.Tensor,
    target_waveform: torch.Tensor,
    *,
    phase_weight: float = 0.75,
    waveform_weight: float = 0.05,
) -> V9PhaseIncrementLoss:
    """Supervise magnitude and inter-frame phase advance directly.

    Phase is compared on the unit circle and weighted by target spectral magnitude so
    near-silent bins do not dominate with arbitrary phase. The waveform term is light;
    V9's primary contract is its differential spectral representation.
    """
    if predicted_magnitude.shape != target_magnitude.shape:
        raise ValueError("predicted and target magnitude shapes must match")
    if predicted_residual_phase.shape != target_residual_phase.shape:
        raise ValueError("predicted and target residual phase shapes must match")
    if not torch.is_complex(predicted_residual_phase) or not torch.is_complex(target_residual_phase):
        raise ValueError("phase residuals must be complex unit factors")
    if predicted_waveform.shape != target_waveform.shape:
        raise ValueError("predicted and target waveforms must match")
    if phase_weight < 0.0 or waveform_weight < 0.0:
        raise ValueError("loss weights must be non-negative")

    target_magnitude = target_magnitude.detach()
    target_residual_phase = target_residual_phase.detach()
    log_magnitude_l1 = F.l1_loss(
        torch.log1p(predicted_magnitude),
        torch.log1p(target_magnitude),
    )

    agreement = (
        predicted_residual_phase * target_residual_phase.conj()
    ).real.clamp(-1.0, 1.0)
    weights = torch.log1p(target_magnitude).clamp_min(0.0)
    weights = weights / weights.mean().clamp_min(1e-4)
    phase_increment_circular = (
        (1.0 - agreement) * weights
    ).sum() / weights.sum().clamp_min(1e-6)

    waveform_l1 = F.l1_loss(predicted_waveform, target_waveform)
    total = (
        log_magnitude_l1
        + float(phase_weight) * phase_increment_circular
        + float(waveform_weight) * waveform_l1
    )
    return V9PhaseIncrementLoss(
        total=total,
        log_magnitude_l1=log_magnitude_l1,
        phase_increment_circular=phase_increment_circular,
        waveform_l1=waveform_l1,
    )
