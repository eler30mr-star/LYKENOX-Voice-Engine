"""F0-locked target-relative harmonic exposure objective for LYKENOX vocoder training.

The perceptual failure is described as radio-mistuned / metallic carrier interference.
Generic mel-envelope and broad-band balance losses can improve while that fine periodic
structure still sounds worse.  This training-only objective directly compares how exposed
F0 harmonics are relative to nearby inter-harmonic energy in prediction versus the paired
real waveform.

Because the target defines the desired contrast, natural speech harmonics are preserved;
only excess or deficient carrier exposure relative to the real recording is penalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


VOCODER_HARMONIC_EXPOSURE_VERSION = "vocoder-harmonic-exposure-v1"


@dataclass(frozen=True)
class HarmonicExposureResult:
    loss: torch.Tensor
    prediction_mean_exposure: torch.Tensor
    target_mean_exposure: torch.Tensor
    valid_fraction: torch.Tensor


def _log_magnitude(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    return torch.log(
        torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs().clamp_min(1e-5)
    )


def target_relative_harmonic_exposure_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    f0_hz: torch.Tensor,
    *,
    sample_rate: int = 24_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    harmonics: int = 8,
) -> HarmonicExposureResult:
    """Match F0-locked harmonic/inter-harmonic contrast to the paired real target."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [batch, samples]")
    if f0_hz.ndim != 2 or f0_hz.shape[0] != prediction.shape[0]:
        raise ValueError("f0_hz must have shape [batch, frames]")
    if harmonics < 1:
        raise ValueError("harmonics must be positive")

    pred_log = _log_magnitude(prediction, n_fft=n_fft, hop_length=hop_length)
    with torch.no_grad():
        target_log = _log_magnitude(target, n_fft=n_fft, hop_length=hop_length)

    time_frames = pred_log.shape[-1]
    f0 = F.interpolate(
        f0_hz.unsqueeze(1),
        size=time_frames,
        mode="linear",
        align_corners=False,
    ).squeeze(1).clamp_min(0.0)
    bin_hz = float(sample_rate) / float(n_fft)
    freq_bins = pred_log.shape[1]

    pred_exposures: list[torch.Tensor] = []
    target_exposures: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for harmonic in range(1, harmonics + 1):
        harmonic_hz = f0 * float(harmonic)
        harmonic_bin = torch.round(harmonic_hz / bin_hz).long()
        side_offset = torch.round((0.35 * f0) / bin_hz).long().clamp_min(1)
        left_bin = harmonic_bin - side_offset
        right_bin = harmonic_bin + side_offset
        valid = (
            (f0 >= 50.0)
            & (harmonic_hz < 0.46 * float(sample_rate))
            & (left_bin >= 1)
            & (right_bin < freq_bins - 1)
        )

        center_idx = harmonic_bin.clamp(0, freq_bins - 1).unsqueeze(1)
        left_idx = left_bin.clamp(0, freq_bins - 1).unsqueeze(1)
        right_idx = right_bin.clamp(0, freq_bins - 1).unsqueeze(1)
        pred_center = torch.gather(pred_log, 1, center_idx).squeeze(1)
        pred_left = torch.gather(pred_log, 1, left_idx).squeeze(1)
        pred_right = torch.gather(pred_log, 1, right_idx).squeeze(1)
        target_center = torch.gather(target_log, 1, center_idx).squeeze(1)
        target_left = torch.gather(target_log, 1, left_idx).squeeze(1)
        target_right = torch.gather(target_log, 1, right_idx).squeeze(1)

        pred_exposures.append(pred_center - 0.5 * (pred_left + pred_right))
        target_exposures.append(target_center - 0.5 * (target_left + target_right))
        masks.append(valid)

    pred_stack = torch.stack(pred_exposures, dim=1)
    target_stack = torch.stack(target_exposures, dim=1)
    mask = torch.stack(masks, dim=1)
    valid_values = mask.sum().clamp_min(1)
    element_loss = F.smooth_l1_loss(pred_stack, target_stack, reduction="none")
    loss = (element_loss * mask.to(element_loss.dtype)).sum() / valid_values
    mask_float = mask.to(pred_stack.dtype)
    pred_mean = (pred_stack * mask_float).sum() / valid_values
    target_mean = (target_stack * mask_float).sum() / valid_values
    return HarmonicExposureResult(
        loss=loss,
        prediction_mean_exposure=pred_mean,
        target_mean_exposure=target_mean,
        valid_fraction=mask_float.mean(),
    )
