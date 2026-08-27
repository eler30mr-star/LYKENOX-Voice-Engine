"""Differentiable periodicity control for LYKENOX vocoder training.

The failed v0/v2 listening gates showed that a generator can improve STFT reconstruction
while still inventing a strong waveform carrier at exactly one mel hop (256 samples,
93.75 Hz at 24 kHz). A fixed notch is unacceptable because a real speaker may genuinely
have F0 near that frequency.

This loss is therefore target-referenced: it compares the generated waveform's *excess*
correlation at the hop grid against the paired real waveform. If the reference genuinely
contains ~93.75 Hz periodicity, the target signature permits it. What is penalized is only
extra frame-grid periodicity not present in the real signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


VOCODER_PERIODICITY_CONTROL_VERSION = "vocoder-periodicity-v1"


@dataclass(frozen=True)
class PeriodicityControlResult:
    loss: torch.Tensor
    generated_signature: torch.Tensor
    target_signature: torch.Tensor


def _normalized_lag_correlation(waveform: torch.Tensor, lag: int) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [batch, samples]")
    if lag < 1 or waveform.shape[1] <= lag + 1:
        raise ValueError("lag must be positive and smaller than waveform length")

    centered = waveform - waveform.mean(dim=1, keepdim=True)
    left = centered[:, :-lag]
    right = centered[:, lag:]
    numerator = (left * right).mean(dim=1)
    denominator = torch.sqrt(
        left.square().mean(dim=1) * right.square().mean(dim=1) + 1e-8
    )
    return numerator / denominator.clamp_min(1e-6)


def hop_periodicity_signature(
    waveform: torch.Tensor,
    *,
    hop_length: int = 256,
) -> torch.Tensor:
    """Return differentiable exact-hop excess correlation for each batch item.

    We do not use raw correlation at lag 256 because normal voiced speech may naturally
    have a similar pitch period. Instead we subtract nearby-lag correlation so only an
    unusually privileged sample-grid period is emphasized. A second signature at 2x hop
    catches a carrier whose strongest autocorrelation lands on the first harmonic.
    """

    sample_count = int(waveform.shape[1])
    offsets = (6, 12)
    signatures: list[torch.Tensor] = []
    for base_lag in (hop_length, hop_length * 2):
        if sample_count <= base_lag + max(offsets) + 1:
            continue
        exact = _normalized_lag_correlation(waveform, base_lag)
        neighbors = [
            _normalized_lag_correlation(waveform, base_lag - offset)
            for offset in offsets
        ] + [
            _normalized_lag_correlation(waveform, base_lag + offset)
            for offset in offsets
        ]
        neighbor_mean = torch.stack(neighbors, dim=0).mean(dim=0)
        signatures.append(exact - neighbor_mean)

    if not signatures:
        raise ValueError("waveform is too short for hop-periodicity control")
    return torch.stack(signatures, dim=1)


def target_referenced_periodicity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    hop_length: int = 256,
) -> PeriodicityControlResult:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [batch, samples]")

    generated_signature = hop_periodicity_signature(
        prediction,
        hop_length=hop_length,
    )
    with torch.no_grad():
        target_signature = hop_periodicity_signature(
            target,
            hop_length=hop_length,
        )
    loss = F.smooth_l1_loss(generated_signature, target_signature)
    return PeriodicityControlResult(
        loss=loss,
        generated_signature=generated_signature,
        target_signature=target_signature,
    )
