"""Frame-grid/checkerboard artifact diagnostics for neural waveform decoders.

A vocoder may report acceptable RMS or spectral losses while emitting a nearly periodic
upsampling-grid tone. This module detects that failure directly in waveform space. It is a
rejection metric only; it never modifies or post-processes audio.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


VOCODER_GRID_ARTIFACT_VERSION = "vocoder-frame-grid-artifact-v1"


@dataclass(frozen=True)
class FrameGridArtifactResult:
    frame_rate_hz: float
    hop_autocorrelation: torch.Tensor
    double_hop_autocorrelation: torch.Tensor
    grid_harmonic_power_fraction: torch.Tensor
    severe_grid_artifact: torch.Tensor


def _as_batch(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [samples] or [batch, samples]")
    if waveform.shape[-1] < 1024:
        raise ValueError("waveform is too short for frame-grid analysis")
    return waveform


def _normalized_autocorrelation(waveform: torch.Tensor, lag: int) -> torch.Tensor:
    if lag < 1 or lag >= waveform.shape[-1]:
        raise ValueError("invalid autocorrelation lag")
    centered = waveform - waveform.mean(dim=-1, keepdim=True)
    left = centered[..., :-lag]
    right = centered[..., lag:]
    numerator = (left * right).sum(dim=-1)
    denominator = torch.sqrt(
        left.square().sum(dim=-1) * right.square().sum(dim=-1)
    ).clamp_min(1e-12)
    return numerator / denominator


def frame_grid_artifact_metrics(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    max_harmonics: int = 8,
    autocorrelation_threshold: float = 0.95,
    harmonic_power_threshold: float = 0.35,
) -> FrameGridArtifactResult:
    """Measure repetition at the mel hop and energy locked to the frame-rate comb.

    ``severe_grid_artifact`` is true when either hop-period repetition is almost exact or
    the frame-rate harmonic comb owns an implausibly large fraction of 20--4000 Hz power.
    The thresholds are deliberately conservative: V7 measured about 0.99 at the 256-sample
    hop, while natural/reference speech in the same audit was far lower.
    """

    if sample_rate < 1 or hop_length < 2:
        raise ValueError("sample_rate and hop_length must be positive")
    if max_harmonics < 1:
        raise ValueError("max_harmonics must be positive")
    if not 0.0 < autocorrelation_threshold < 1.0:
        raise ValueError("autocorrelation_threshold must be between zero and one")
    if not 0.0 < harmonic_power_threshold < 1.0:
        raise ValueError("harmonic_power_threshold must be between zero and one")

    batch = _as_batch(waveform)
    frame_rate = float(sample_rate) / float(hop_length)
    hop_corr = _normalized_autocorrelation(batch, hop_length)
    double_hop_corr = _normalized_autocorrelation(batch, hop_length * 2)

    window = torch.hann_window(
        batch.shape[-1],
        device=batch.device,
        dtype=batch.dtype,
    )
    spectrum = torch.fft.rfft(batch * window, dim=-1)
    power = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(
        batch.shape[-1],
        d=1.0 / float(sample_rate),
        device=batch.device,
    )
    audible = (frequencies >= 20.0) & (frequencies <= 4000.0)
    total_power = power[..., audible].sum(dim=-1).clamp_min(1e-12)

    # Integrate a narrow band around exact frame-rate harmonics. The width is at least one
    # FFT bin, so the detector is stable for both crop and full-utterance probes.
    bin_width = float(sample_rate) / float(batch.shape[-1])
    half_width_hz = max(bin_width * 1.5, frame_rate * 0.01)
    comb_mask = torch.zeros_like(frequencies, dtype=torch.bool)
    for harmonic in range(1, max_harmonics + 1):
        center = harmonic * frame_rate
        if center > 4000.0:
            break
        comb_mask |= (frequencies >= center - half_width_hz) & (
            frequencies <= center + half_width_hz
        )
    comb_power = power[..., comb_mask].sum(dim=-1)
    comb_fraction = comb_power / total_power

    severe = (
        torch.maximum(hop_corr.abs(), double_hop_corr.abs())
        >= autocorrelation_threshold
    ) | (comb_fraction >= harmonic_power_threshold)
    return FrameGridArtifactResult(
        frame_rate_hz=frame_rate,
        hop_autocorrelation=hop_corr,
        double_hop_autocorrelation=double_hop_corr,
        grid_harmonic_power_fraction=comb_fraction,
        severe_grid_artifact=severe,
    )
