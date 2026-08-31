"""Direct complex-spectral supervision for the V8 fixed-iSTFT vocoder candidate."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


V8_COMPLEX_SPECTRAL_LOSS_VERSION = "vocoder-v8-complex-spectral-loss-v1"


@dataclass(frozen=True)
class V8ComplexSpectralLoss:
    total: torch.Tensor
    complex_relative_l1: torch.Tensor
    log_magnitude_l1: torch.Tensor
    waveform_l1: torch.Tensor


def v8_complex_spectral_loss(
    predicted_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    predicted_waveform: torch.Tensor,
    target_waveform: torch.Tensor,
    *,
    log_magnitude_weight: float = 0.50,
    waveform_weight: float = 0.10,
) -> V8ComplexSpectralLoss:
    """Supervise complex coefficients directly while retaining perceptual magnitude pressure.

    The complex term is target-scale relative so amplitude cannot dominate the objective.
    It remains phase-sensitive, unlike a magnitude-only STFT loss. The waveform term is
    intentionally light; V8's main representation contract is the complex STFT itself.
    """
    if predicted_spectrum.shape != target_spectrum.shape:
        raise ValueError("predicted and target complex spectra must share shape")
    if predicted_spectrum.ndim != 3 or not torch.is_complex(predicted_spectrum):
        raise ValueError("predicted_spectrum must be complex [batch, frequency, frames]")
    if not torch.is_complex(target_spectrum):
        raise ValueError("target_spectrum must be complex")
    if predicted_waveform.shape != target_waveform.shape or predicted_waveform.ndim != 2:
        raise ValueError("predicted and target waveform must share [batch, samples]")
    if log_magnitude_weight < 0.0 or waveform_weight < 0.0:
        raise ValueError("loss weights must be non-negative")

    target = target_spectrum.detach()
    target_scale = target.abs().mean().clamp_min(1e-4)
    complex_relative_l1 = (predicted_spectrum - target).abs().mean() / target_scale
    log_magnitude_l1 = F.l1_loss(
        torch.log1p(predicted_spectrum.abs()),
        torch.log1p(target.abs()),
    )
    waveform_l1 = F.l1_loss(predicted_waveform, target_waveform)
    total = (
        complex_relative_l1
        + float(log_magnitude_weight) * log_magnitude_l1
        + float(waveform_weight) * waveform_l1
    )
    return V8ComplexSpectralLoss(
        total=total,
        complex_relative_l1=complex_relative_l1,
        log_magnitude_l1=log_magnitude_l1,
        waveform_l1=waveform_l1,
    )
