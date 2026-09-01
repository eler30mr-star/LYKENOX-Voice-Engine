"""Owned LYKENOX vocoder loss V2 with valid crop context only.

Historical V1 reconstruction/envelope losses used centered spectral transforms directly on
waveform crops.  The centered transforms reflected samples beyond the crop boundaries and
therefore supervised synthetic boundary context.  The historical mel envelope path also
produced T+1 analysis frames for a T-hop waveform crop even though the vocoder conditioning
contains exactly T cached full-utterance mel frames.

V2 keeps the historical centered analysis grid but removes those invalid targets:

* multi-resolution STFT terms use only frames whose complete FFT window is contained in
  the waveform crop;
* the envelope objective compares generated waveform analysis directly with the owned
  cached full-utterance conditioning mel, not with a mel re-analysis of the cropped target;
* generated mel is aligned to exactly the conditioning frame count before scoring;
* frames whose centered window would require samples outside the crop are masked out;
* V1 losses remain untouched for artifact reproducibility.

This module contains training objectives only.  It does not select an architecture, load a
checkpoint, alter duration, normalize output, or authorize persistent training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_vocoder_losses import STFT_RESOLUTIONS


OWNED_VOCODER_LOSS_V2_VERSION = (
    "owned-vocoder-loss-v2-valid-context-conditioning-aligned"
)


@dataclass(frozen=True)
class OwnedVocoderReconstructionLossV2:
    total: torch.Tensor
    waveform_l1: torch.Tensor
    spectral_convergence: torch.Tensor
    log_magnitude: torch.Tensor
    valid_frame_counts: tuple[int, ...]
    analysis_frame_counts: tuple[int, ...]


@dataclass(frozen=True)
class OwnedVocoderEnvelopeLossV2:
    total: torch.Tensor
    log_mel_l1: torch.Tensor
    spectral_slope_l1: torch.Tensor
    temporal_delta_l1: torch.Tensor
    conditioning_frames: int
    analysis_frames: int
    valid_conditioning_frames: int


def valid_centered_frame_mask(
    *,
    sample_count: int,
    frame_count: int,
    n_fft: int,
    hop_length: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return frames whose centered FFT support lies fully inside the crop.

    ``torch.stft(center=True)`` places analysis center ``i * hop_length`` at frame ``i``.
    A frame is valid only when the full ``n_fft`` support is available from the crop itself;
    no reflected/zero context outside the crop is allowed to contribute to V2 losses.
    """

    if sample_count < 1 or frame_count < 1:
        raise ValueError("sample_count and frame_count must be positive")
    if n_fft < 2 or hop_length < 1:
        raise ValueError("invalid spectral geometry")
    half = int(n_fft) // 2
    centers = torch.arange(frame_count, dtype=torch.long, device=device) * int(hop_length)
    return (centers >= half) & (centers + half <= int(sample_count))


def _centered_stft_magnitude(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    window = torch.hann_window(
        win_length,
        dtype=waveform.dtype,
        device=waveform.device,
    )
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    ).abs().clamp_min(1e-5)


def valid_context_multi_resolution_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    waveform_weight: float = 0.10,
) -> OwnedVocoderReconstructionLossV2:
    """Waveform + multi-resolution STFT loss without artificial crop-boundary frames."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [batch, samples]")
    if waveform_weight < 0.0:
        raise ValueError("waveform_weight must be non-negative")

    waveform_l1 = F.l1_loss(prediction, target)
    convergence_terms: list[torch.Tensor] = []
    logmag_terms: list[torch.Tensor] = []
    valid_counts: list[int] = []
    analysis_counts: list[int] = []
    sample_count = int(prediction.shape[1])

    for n_fft, hop_length, win_length in STFT_RESOLUTIONS:
        predicted = _centered_stft_magnitude(
            prediction,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        truth = _centered_stft_magnitude(
            target,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        if predicted.shape != truth.shape:
            raise RuntimeError("prediction/target STFT shapes differ")

        frame_count = int(predicted.shape[-1])
        mask = valid_centered_frame_mask(
            sample_count=sample_count,
            frame_count=frame_count,
            n_fft=n_fft,
            hop_length=hop_length,
            device=prediction.device,
        )
        if not bool(mask.any()):
            raise ValueError("waveform crop is too short for a valid V2 STFT frame")

        predicted_valid = predicted[..., mask]
        truth_valid = truth[..., mask]
        difference = torch.linalg.vector_norm(truth_valid - predicted_valid)
        reference = torch.linalg.vector_norm(truth_valid).clamp_min(1e-6)
        convergence_terms.append(difference / reference)
        logmag_terms.append(
            F.l1_loss(torch.log(predicted_valid), torch.log(truth_valid))
        )
        valid_counts.append(int(mask.sum()))
        analysis_counts.append(frame_count)

    spectral_convergence = torch.stack(convergence_terms).mean()
    log_magnitude = torch.stack(logmag_terms).mean()
    total = waveform_weight * waveform_l1 + spectral_convergence + log_magnitude
    return OwnedVocoderReconstructionLossV2(
        total=total,
        waveform_l1=waveform_l1,
        spectral_convergence=spectral_convergence,
        log_magnitude=log_magnitude,
        valid_frame_counts=tuple(valid_counts),
        analysis_frame_counts=tuple(analysis_counts),
    )


class ConditioningAlignedLogMelEnvelopeLossV2(nn.Module):
    """Generated waveform -> owned cached conditioning-mel objective on valid frames."""

    def __init__(
        self,
        config: LykenoxSpeechConfig | None = None,
        *,
        spectral_slope_weight: float = 0.50,
        temporal_delta_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxSpeechConfig()
        if spectral_slope_weight < 0.0 or temporal_delta_weight < 0.0:
            raise ValueError("envelope weights must be non-negative")
        self.spectral_slope_weight = float(spectral_slope_weight)
        self.temporal_delta_weight = float(temporal_delta_weight)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.mel_bins,
            power=1.0,
            center=True,
        )

    def _generated_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        # [batch, mel, frames] -> [batch, frames, mel]
        return torch.log(self.mel(waveform).clamp_min(1e-5)).transpose(1, 2)

    def forward(
        self,
        prediction: torch.Tensor,
        conditioning_log_mel: torch.Tensor,
    ) -> OwnedVocoderEnvelopeLossV2:
        if prediction.ndim != 2:
            raise ValueError("prediction must have shape [batch, samples]")
        if conditioning_log_mel.ndim != 3:
            raise ValueError(
                "conditioning_log_mel must have shape [batch, frames, mel_bins]"
            )
        if int(conditioning_log_mel.shape[0]) != int(prediction.shape[0]):
            raise ValueError("conditioning batch size must match prediction")
        if int(conditioning_log_mel.shape[2]) != int(self.config.mel_bins):
            raise ValueError("conditioning mel-bin count does not match speech config")
        if not bool(torch.isfinite(conditioning_log_mel).all()):
            raise ValueError("conditioning mel contains non-finite values")

        conditioning_frames = int(conditioning_log_mel.shape[1])
        expected_samples = conditioning_frames * int(self.config.hop_length)
        if int(prediction.shape[1]) != expected_samples:
            raise ValueError(
                "prediction length must equal conditioning_frames * hop_length"
            )

        generated_all = self._generated_log_mel(prediction)
        analysis_frames = int(generated_all.shape[1])
        if analysis_frames < conditioning_frames:
            raise RuntimeError("generated mel has fewer frames than conditioning")
        # Historical center=True geometry yields T+1 for exactly T hops.  V2 explicitly
        # aligns the generated analysis to the T conditioning slots and never scores the
        # unconditioned terminal frame.
        generated = generated_all[:, :conditioning_frames, :]
        if generated.shape != conditioning_log_mel.shape:
            raise RuntimeError("generated/conditioning mel shapes differ after alignment")

        mask = valid_centered_frame_mask(
            sample_count=int(prediction.shape[1]),
            frame_count=conditioning_frames,
            n_fft=int(self.config.n_fft),
            hop_length=int(self.config.hop_length),
            device=prediction.device,
        )
        if not bool(mask.any()):
            raise ValueError("waveform crop is too short for a valid V2 mel frame")

        generated_valid = generated[:, mask, :]
        conditioning_valid = conditioning_log_mel[:, mask, :]
        level = F.l1_loss(generated_valid, conditioning_valid)

        if int(generated_valid.shape[2]) > 1:
            spectral_slope = F.l1_loss(
                torch.diff(generated_valid, dim=2),
                torch.diff(conditioning_valid, dim=2),
            )
        else:
            spectral_slope = level.new_zeros(())

        pair_mask = mask[:-1] & mask[1:]
        if bool(pair_mask.any()):
            generated_delta = torch.diff(generated, dim=1)[:, pair_mask, :]
            conditioning_delta = torch.diff(conditioning_log_mel, dim=1)[:, pair_mask, :]
            temporal_delta = F.l1_loss(generated_delta, conditioning_delta)
        else:
            temporal_delta = level.new_zeros(())

        total = (
            level
            + self.spectral_slope_weight * spectral_slope
            + self.temporal_delta_weight * temporal_delta
        )
        return OwnedVocoderEnvelopeLossV2(
            total=total,
            log_mel_l1=level,
            spectral_slope_l1=spectral_slope,
            temporal_delta_l1=temporal_delta,
            conditioning_frames=conditioning_frames,
            analysis_frames=analysis_frames,
            valid_conditioning_frames=int(mask.sum()),
        )
