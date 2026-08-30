"""Target-relative level and spectral-presence objectives for LYKENOX vocoder training.

These losses address two perceptual failure modes that broad reconstruction metrics did not
reliably prevent:

1. weak perceived voice level / dynamics;
2. nasal or muffled spectral collapse caused by excessive low-frequency concentration and
   insufficient 1-8 kHz formant/consonant energy.

They compare prediction against the paired target recording. They do not normalize or boost
waveforms at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


VOCODER_LEVEL_PRESENCE_VERSION = "vocoder-level-presence-v2"


@dataclass(frozen=True)
class LevelLossResult:
    loss: torch.Tensor
    global_log_rms_loss: torch.Tensor
    frame_log_rms_loss: torch.Tensor
    prediction_rms_db: torch.Tensor
    target_rms_db: torch.Tensor
    rms_error_db: torch.Tensor
    active_frame_fraction: torch.Tensor


@dataclass(frozen=True)
class PresenceLossResult:
    loss: torch.Tensor
    prediction_band_fractions: torch.Tensor
    target_band_fractions: torch.Tensor
    presence_1k_8k_error_db: torch.Tensor


def _validate_wave_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("waveforms must have shape [batch, samples]")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target waveform shapes must match")


def _log_rms(x: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    return 0.5 * torch.log(torch.mean(x.square(), dim=-1).clamp_min(eps))


def target_relative_level_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    frame_length: int = 512,
    frame_hop: int = 256,
    eps: float = 1e-7,
    global_weight: float = 0.60,
    active_margin_db: float = 24.0,
    activity_softness_db: float = 3.0,
) -> LevelLossResult:
    """Match global and active-speech short-time amplitude.

    V1 gave every short-time frame equal authority. Long quiet regions could therefore
    dominate the frame term even when the audible speech itself was becoming too weak.
    V2 keeps a stronger global-RMS anchor and softly weights the frame objective toward
    frames that contain meaningful target energy.
    """

    _validate_wave_pair(prediction, target)
    if frame_length < 64 or frame_hop < 1:
        raise ValueError("invalid level-loss framing")
    if not (0.0 < global_weight < 1.0):
        raise ValueError("global_weight must be between 0 and 1")
    if active_margin_db <= 0.0 or activity_softness_db <= 0.0:
        raise ValueError("activity weighting parameters must be positive")

    pred_global = _log_rms(prediction, eps=eps)
    target_global = _log_rms(target, eps=eps)
    global_loss = F.smooth_l1_loss(pred_global, target_global)

    def framed_log_rms(x: torch.Tensor) -> torch.Tensor:
        power = F.avg_pool1d(
            x.unsqueeze(1).square(),
            kernel_size=frame_length,
            stride=frame_hop,
            ceil_mode=False,
        ).squeeze(1)
        return 0.5 * torch.log(power.clamp_min(eps))

    pred_frames = framed_log_rms(prediction)
    target_frames = framed_log_rms(target)

    log10 = torch.log(
        torch.tensor(10.0, device=prediction.device, dtype=prediction.dtype)
    )
    db_scale = 20.0 / log10
    pred_global_db = pred_global * db_scale
    target_global_db = target_global * db_scale
    target_frame_db = target_frames * db_scale

    # The threshold follows each target example instead of imposing one arbitrary
    # recording level. The -60 dBFS floor only prevents nearly silent targets from
    # assigning high weight to numerical noise.
    threshold_db = torch.maximum(
        target_global_db.unsqueeze(1) - active_margin_db,
        torch.full_like(target_frame_db, -60.0),
    )
    activity = torch.sigmoid(
        (target_frame_db - threshold_db) / activity_softness_db
    )
    frame_error = F.smooth_l1_loss(
        pred_frames,
        target_frames,
        reduction="none",
    )
    frame_loss = (frame_error * activity).sum() / activity.sum().clamp_min(1e-6)

    loss = global_weight * global_loss + (1.0 - global_weight) * frame_loss
    rms_error_db = (pred_global_db - target_global_db).abs().mean()

    return LevelLossResult(
        loss=loss,
        global_log_rms_loss=global_loss,
        frame_log_rms_loss=frame_loss,
        prediction_rms_db=pred_global_db.mean(),
        target_rms_db=target_global_db.mean(),
        rms_error_db=rms_error_db,
        active_frame_fraction=activity.mean(),
    )


def _band_fractions(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    eps: float,
) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    ).abs().square()
    frequencies = torch.fft.rfftfreq(
        n_fft,
        d=1.0 / float(sample_rate),
        device=waveform.device,
    )
    bands = ((80.0, 300.0), (300.0, 1000.0), (1000.0, 3000.0), (3000.0, 8000.0))
    energies: list[torch.Tensor] = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        if not bool(mask.any()):
            raise RuntimeError("presence-loss FFT has no bins for a requested band")
        energies.append(spectrum[:, mask, :].mean(dim=(1, 2)))
    stacked = torch.stack(energies, dim=1).clamp_min(eps)
    return stacked / stacked.sum(dim=1, keepdim=True).clamp_min(eps)


def target_relative_presence_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    eps: float = 1e-8,
) -> PresenceLossResult:
    """Match target band distribution with extra authority in formant/consonant bands."""

    _validate_wave_pair(prediction, target)
    pred = _band_fractions(
        prediction,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        eps=eps,
    )
    ref = _band_fractions(
        target,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        eps=eps,
    )

    # Higher bands carry much of the perceptual clarity that collapsed in the rejected v5
    # oracle outputs. The comparison remains target-relative; it does not impose a generic EQ.
    weights = torch.tensor(
        [0.75, 1.0, 1.5, 1.5],
        device=prediction.device,
        dtype=prediction.dtype,
    )
    log_delta = torch.log(pred.clamp_min(eps)) - torch.log(ref.clamp_min(eps))
    loss = (
        F.smooth_l1_loss(
            log_delta,
            torch.zeros_like(log_delta),
            reduction="none",
        )
        * weights
    ).mean()

    pred_presence = (pred[:, 2] + pred[:, 3]).clamp_min(eps)
    ref_presence = (ref[:, 2] + ref[:, 3]).clamp_min(eps)
    presence_error_db = (
        10.0 * torch.log10(pred_presence / ref_presence)
    ).abs().mean()

    return PresenceLossResult(
        loss=loss,
        prediction_band_fractions=pred.mean(dim=0),
        target_band_fractions=ref.mean(dim=0),
        presence_1k_8k_error_db=presence_error_db,
    )
