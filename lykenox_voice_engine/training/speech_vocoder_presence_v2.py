"""Owned target-relative spectral-presence objective on valid crop context only.

The historical presence objective used centered STFT analysis over waveform crops and
averaged every analysis frame, including frames whose FFT support extended beyond the crop.
That reintroduced reflected boundary context after Loss V2 had removed it elsewhere.

Presence V2 keeps the target-relative 80 Hz-8 kHz semantics and the one-sided clarity guard,
but scores only centered STFT frames whose full window is contained in the crop.  It applies
no EQ, gain, normalization, or absolute spectral target; the paired LYKENOX recording remains
the authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lykenox_voice_engine.training.speech_vocoder_loss_v2 import valid_centered_frame_mask


OWNED_VOCODER_PRESENCE_V2_VERSION = "owned-vocoder-presence-v2-valid-context-target-relative"
PRESENCE_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (80.0, 300.0),
    (300.0, 1000.0),
    (1000.0, 3000.0),
    (3000.0, 8000.0),
)
PRESENCE_BAND_WEIGHTS: tuple[float, ...] = (0.75, 1.0, 1.5, 1.5)
PRESENCE_CLARITY_WEIGHTS: tuple[float, ...] = (0.0, 0.0, 1.0, 2.0)
PRESENCE_CLARITY_GUARD_WEIGHT = 0.75


@dataclass(frozen=True)
class OwnedPresenceLossV2Result:
    loss: torch.Tensor
    prediction_band_fractions: torch.Tensor
    target_band_fractions: torch.Tensor
    presence_1k_8k_error_db: torch.Tensor
    valid_frame_count: int
    analysis_frame_count: int


def _valid_context_band_fractions(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    eps: float,
) -> tuple[torch.Tensor, int, int]:
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [batch, samples]")
    if int(waveform.shape[1]) < n_fft:
        raise ValueError("waveform crop is too short for presence V2 analysis")

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
    analysis_frames = int(spectrum.shape[-1])
    valid_mask = valid_centered_frame_mask(
        sample_count=int(waveform.shape[1]),
        frame_count=analysis_frames,
        n_fft=n_fft,
        hop_length=hop_length,
        device=waveform.device,
    )
    if not bool(valid_mask.any()):
        raise ValueError("presence V2 has no valid centered frames")
    spectrum = spectrum[..., valid_mask]

    frequencies = torch.fft.rfftfreq(
        n_fft,
        d=1.0 / float(sample_rate),
        device=waveform.device,
        dtype=waveform.dtype,
    )
    energies: list[torch.Tensor] = []
    for low, high in PRESENCE_BANDS_HZ:
        band_mask = (frequencies >= low) & (frequencies < high)
        if not bool(band_mask.any()):
            raise RuntimeError("presence V2 FFT has no bins for a requested band")
        energies.append(spectrum[:, band_mask, :].mean(dim=(1, 2)))
    stacked = torch.stack(energies, dim=1).clamp_min(eps)
    fractions = stacked / stacked.sum(dim=1, keepdim=True).clamp_min(eps)
    return fractions, int(valid_mask.sum()), analysis_frames


def target_relative_presence_loss_v2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    eps: float = 1e-8,
) -> OwnedPresenceLossV2Result:
    """Match paired spectral presence using only analysis windows with real crop context."""

    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape [batch, samples]")

    pred, pred_valid, pred_analysis = _valid_context_band_fractions(
        prediction,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        eps=eps,
    )
    with torch.no_grad():
        ref, ref_valid, ref_analysis = _valid_context_band_fractions(
            target,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            eps=eps,
        )
    if pred_valid != ref_valid or pred_analysis != ref_analysis:
        raise RuntimeError("prediction/target presence V2 frame geometry differs")

    band_weights = torch.tensor(
        PRESENCE_BAND_WEIGHTS,
        device=prediction.device,
        dtype=prediction.dtype,
    ).view(1, -1)
    log_delta = torch.log(pred.clamp_min(eps)) - torch.log(ref.clamp_min(eps))
    symmetric = (
        F.smooth_l1_loss(
            log_delta,
            torch.zeros_like(log_delta),
            reduction="none",
        )
        * band_weights
    ).sum(dim=1) / band_weights.sum()

    underpresence = torch.relu(-log_delta)
    underpresence_error = F.smooth_l1_loss(
        underpresence,
        torch.zeros_like(underpresence),
        reduction="none",
    )
    clarity_weights = torch.tensor(
        PRESENCE_CLARITY_WEIGHTS,
        device=prediction.device,
        dtype=prediction.dtype,
    ).view(1, -1)
    clarity_guard = (
        underpresence_error * clarity_weights
    ).sum(dim=1) / clarity_weights.sum().clamp_min(eps)

    loss = symmetric.mean() + PRESENCE_CLARITY_GUARD_WEIGHT * clarity_guard.mean()
    pred_presence = (pred[:, 2] + pred[:, 3]).clamp_min(eps)
    ref_presence = (ref[:, 2] + ref[:, 3]).clamp_min(eps)
    presence_error_db = (
        10.0 * torch.log10(pred_presence / ref_presence)
    ).abs().mean()

    return OwnedPresenceLossV2Result(
        loss=loss,
        prediction_band_fractions=pred.mean(dim=0),
        target_band_fractions=ref.mean(dim=0),
        presence_1k_8k_error_db=presence_error_db,
        valid_frame_count=pred_valid,
        analysis_frame_count=pred_analysis,
    )
