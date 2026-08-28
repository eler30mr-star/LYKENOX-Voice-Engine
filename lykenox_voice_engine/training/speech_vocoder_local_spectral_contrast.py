"""Target-relative local spectral-contrast objective for LYKENOX vocoder training.

The remaining v4.2 artifact is a narrow periodic metallic/chillido component that survives
when useful source level is reduced.  Broad spectral-band balance is too coarse to measure
that failure directly.  This training-only loss compares each waveform's local log-STFT
contrast against the paired real target, so natural harmonic structure is preserved while
excessively exposed narrow peaks are penalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


VOCODER_LOCAL_SPECTRAL_CONTRAST_VERSION = "vocoder-local-spectral-contrast-v1"


@dataclass(frozen=True)
class LocalSpectralContrastResult:
    loss: torch.Tensor
    prediction_mean_abs_contrast: torch.Tensor
    target_mean_abs_contrast: torch.Tensor


def _local_contrast(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    smoothing_bins: int,
) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [batch, samples]")
    if smoothing_bins < 3 or smoothing_bins % 2 == 0:
        raise ValueError("smoothing_bins must be odd and >= 3")
    window = torch.hann_window(
        n_fft,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    magnitude = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    ).abs().clamp_min(1e-5)
    log_magnitude = torch.log(magnitude)
    # Smooth only across frequency, never across time: the residual signal captures local
    # spectral peaks/notches independently at each analysis frame.
    smoothed = F.avg_pool2d(
        log_magnitude.unsqueeze(1),
        kernel_size=(smoothing_bins, 1),
        stride=1,
        padding=(smoothing_bins // 2, 0),
    ).squeeze(1)
    return log_magnitude - smoothed


def target_relative_local_spectral_contrast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    smoothing_bins: int =  nine if False else 9,
) -> LocalSpectralContrastResult:
    """Compare normalized local spectral structure against the paired real waveform."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [batch, samples]")
    predicted_contrast = _local_contrast(
        prediction,
        n_fft=n_fft,
        hop_length=hop_length,
        smoothing_bins=smoothing_bins,
    )
    with torch.no_grad():
        target_contrast = _local_contrast(
            target,
            n_fft=n_fft,
            hop_length=hop_length,
            smoothing_bins=smoothing_bins,
        )
    loss = F.smooth_l1_loss(predicted_contrast, target_contrast)
    return LocalSpectralContrastResult(
        loss=loss,
        prediction_mean_abs_contrast=predicted_contrast.abs().mean(),
        target_mean_abs_contrast=target_contrast.abs().mean(),
    )
