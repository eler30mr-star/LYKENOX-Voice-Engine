"""Perceptually targeted spectral-envelope loss for LYKENOX vocoder training.

The v4.1 oracle failure is not a simple loudness problem: generated speech retains too much
periodic metallic character and too little natural formant/consonant detail even when the
conditioning mel is real.  Multi-resolution STFT loss remains useful, but v4.2 also needs a
direct contract that the generated waveform reproduces the mel envelope it was conditioned
on.

This loss compares log-mel level, spectral slope across mel bands, and temporal change.  It
is training-only and introduces no runtime dependency or inference input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig


VOCODER_ENVELOPE_LOSS_VERSION = "vocoder-envelope-loss-v1"


@dataclass(frozen=True)
class VocoderEnvelopeLoss:
    total: torch.Tensor
    log_mel_l1: torch.Tensor
    spectral_slope_l1: torch.Tensor
    temporal_delta_l1: torch.Tensor


class LogMelEnvelopeLoss(nn.Module):
    """Differentiable waveform -> log-mel envelope matching objective."""

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
            raise ValueError("envelope loss weights must be non-negative")
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

    def _log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        return torch.log(self.mel(waveform).clamp_min(1e-5))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> VocoderEnvelopeLoss:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target waveform shapes must match")
        predicted = self._log_mel(prediction)
        truth = self._log_mel(target)
        if predicted.shape != truth.shape:
            raise RuntimeError("prediction/target log-mel shapes differ")

        level = F.l1_loss(predicted, truth)
        if predicted.shape[1] > 1:
            spectral_slope = F.l1_loss(
                torch.diff(predicted, dim=1),
                torch.diff(truth, dim=1),
            )
        else:
            spectral_slope = level.new_zeros(())
        if predicted.shape[2] > 1:
            temporal_delta = F.l1_loss(
                torch.diff(predicted, dim=2),
                torch.diff(truth, dim=2),
            )
        else:
            temporal_delta = level.new_zeros(())

        total = (
            level
            + self.spectral_slope_weight * spectral_slope
            + self.temporal_delta_weight * temporal_delta
        )
        return VocoderEnvelopeLoss(
            total=total,
            log_mel_l1=level,
            spectral_slope_l1=spectral_slope,
            temporal_delta_l1=temporal_delta,
        )
