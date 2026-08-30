"""Content-sensitive mel re-encoding loss for the LYKENOX V7 vocoder.

V6 showed that global band fractions and RMS can improve while intelligibility collapses. V7
therefore carries an explicit differentiable waveform -> log-mel consistency objective against
the conditioning mel itself. The loss is target-relative; it does not impose a fixed EQ curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig


VOCODER_V7_CONTENT_LOSS_VERSION = "vocoder-v7-mel-content-v1"


@dataclass(frozen=True)
class V7MelContentLossResult:
    total: torch.Tensor
    log_mel_l1: torch.Tensor
    centered_shape_l1: torch.Tensor
    spectral_delta_l1: torch.Tensor
    temporal_delta_l1: torch.Tensor
    temporal_acceleration_l1: torch.Tensor


class V7MelContentConsistencyLoss(nn.Module):
    """Require generated waveform to re-encode the mel content used to condition V7."""

    def __init__(
        self,
        config: LykenoxSpeechConfig | None = None,
        *,
        centered_shape_weight: float = 0.50,
        spectral_delta_weight: float = 0.40,
        temporal_delta_weight: float = 0.35,
        temporal_acceleration_weight: float = 0.15,
        boundary_margin_frames: int = 2,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxSpeechConfig()
        weights = (
            centered_shape_weight,
            spectral_delta_weight,
            temporal_delta_weight,
            temporal_acceleration_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v7 content-loss weights must be non-negative")
        if boundary_margin_frames < 0:
            raise ValueError("boundary_margin_frames must be non-negative")
        self.centered_shape_weight = float(centered_shape_weight)
        self.spectral_delta_weight = float(spectral_delta_weight)
        self.temporal_delta_weight = float(temporal_delta_weight)
        self.temporal_acceleration_weight = float(temporal_acceleration_weight)
        self.boundary_margin_frames = int(boundary_margin_frames)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.mel_bins,
            power=1.0,
            center=True,
        )

    def _prediction_log_mel(
        self,
        waveform: torch.Tensor,
        conditioning_frames: int,
    ) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        log_mel = torch.log(self.mel(waveform).clamp_min(1e-5))
        if log_mel.shape[-1] < conditioning_frames:
            raise RuntimeError("v7 predicted mel is shorter than conditioning mel")
        return log_mel[..., :conditioning_frames]

    def _interior(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        margin = self.boundary_margin_frames
        frames = int(target.shape[-1])
        if margin > 0 and frames > 2 * margin + 2:
            return prediction[..., margin:-margin], target[..., margin:-margin]
        return prediction, target

    def forward(
        self,
        waveform: torch.Tensor,
        conditioning_mel: torch.Tensor,
    ) -> V7MelContentLossResult:
        if conditioning_mel.ndim != 3:
            raise ValueError("conditioning_mel must have shape [batch, frames, mel_bins]")
        if conditioning_mel.shape[-1] != self.config.mel_bins:
            raise ValueError("v7 conditioning mel-bin mismatch")
        if waveform.shape[0] != conditioning_mel.shape[0]:
            raise ValueError("v7 waveform/mel batch mismatch")

        target = conditioning_mel.transpose(1, 2)
        prediction = self._prediction_log_mel(waveform, int(conditioning_mel.shape[1]))
        prediction, target = self._interior(prediction, target)

        level = F.l1_loss(prediction, target)

        prediction_centered = prediction - prediction.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        centered_shape = F.l1_loss(prediction_centered, target_centered)

        if prediction.shape[1] > 1:
            spectral_delta = F.l1_loss(
                torch.diff(prediction, dim=1),
                torch.diff(target, dim=1),
            )
        else:
            spectral_delta = level.new_zeros(())

        if prediction.shape[2] > 1:
            prediction_delta = torch.diff(prediction, dim=2)
            target_delta = torch.diff(target, dim=2)
            temporal_delta = F.l1_loss(prediction_delta, target_delta)
        else:
            prediction_delta = target_delta = None
            temporal_delta = level.new_zeros(())

        if prediction_delta is not None and prediction_delta.shape[2] > 1:
            temporal_acceleration = F.l1_loss(
                torch.diff(prediction_delta, dim=2),
                torch.diff(target_delta, dim=2),
            )
        else:
            temporal_acceleration = level.new_zeros(())

        total = (
            level
            + self.centered_shape_weight * centered_shape
            + self.spectral_delta_weight * spectral_delta
            + self.temporal_delta_weight * temporal_delta
            + self.temporal_acceleration_weight * temporal_acceleration
        )
        return V7MelContentLossResult(
            total=total,
            log_mel_l1=level,
            centered_shape_l1=centered_shape,
            spectral_delta_l1=spectral_delta,
            temporal_delta_l1=temporal_delta,
            temporal_acceleration_l1=temporal_acceleration,
        )
