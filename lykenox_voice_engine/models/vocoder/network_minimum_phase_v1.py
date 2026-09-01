"""Owned frame-rate cepstral envelope predictor for the minimum-phase renderer.

This module is intentionally predictor-only.  It never creates an optimizer, loads a
checkpoint, renders waveform samples, predicts phase, or performs learned temporal
upsampling.  The output is a 64-coefficient real cepstrum per conditioning frame.

The final projection is exactly zero-initialized.  Therefore a newly instantiated predictor
produces a flat spectral envelope and leaves the already-proven fixed renderer at its exact
identity operating point until learning is explicitly authorized by a later gate.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


PREDICTOR_ARCHITECTURE = "lykenox_owned_frame_rate_cepstral_predictor_v1"
MEL_BINS = 80
CEPSTRAL_ORDER = 64
DEFAULT_HIDDEN_CHANNELS = 128
DEFAULT_DILATIONS = (1, 2, 4, 8)


class _FrameResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        if channels < 16:
            raise ValueError("channels must be >=16")
        if dilation < 1:
            raise ValueError("dilation must be positive")
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            dilation=dilation,
            padding=2 * dilation,
            groups=channels,
        )
        self.pointwise_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.pointwise_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(F.gelu(x))
        activation, gate = self.pointwise_in(y).chunk(2, dim=1)
        y = torch.tanh(activation) * torch.sigmoid(gate)
        y = self.pointwise_out(y)
        return (residual + y) * (2.0 ** -0.5)


class LykenoxFrameRateCepstralPredictorV1(nn.Module):
    """Predict frame-rate real-cepstral log-spectral envelopes from owned conditioning."""

    architecture = PREDICTOR_ARCHITECTURE

    def __init__(
        self,
        *,
        mel_bins: int = MEL_BINS,
        cepstral_order: int = CEPSTRAL_ORDER,
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        if mel_bins != MEL_BINS:
            raise ValueError(f"mel_bins must remain {MEL_BINS} for architecture contract v1")
        if cepstral_order != CEPSTRAL_ORDER:
            raise ValueError(
                f"cepstral_order must remain {CEPSTRAL_ORDER} for architecture contract v1"
            )
        if hidden_channels < 32:
            raise ValueError("hidden_channels must be >=32")
        if not dilations or any(int(value) < 1 for value in dilations):
            raise ValueError("dilations must be a non-empty tuple of positive integers")

        self.mel_bins = int(mel_bins)
        self.cepstral_order = int(cepstral_order)
        self.hidden_channels = int(hidden_channels)
        self.dilations = tuple(int(value) for value in dilations)

        # mel80 + bounded log-F0 + voiced + periodicity.  Every feature remains at the
        # conditioning frame rate; there is no stride, transpose convolution, or sample-rate
        # representation in this model.
        input_channels = self.mel_bins + 3
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_FrameResidualBlock(hidden_channels, dilation) for dilation in self.dilations]
        )
        self.pre_output = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.cepstral_projection = nn.Conv1d(
            hidden_channels,
            self.cepstral_order,
            kernel_size=1,
        )

        # Neutral initialization is a hard structural contract: c[k]=0 -> H(z)=1.
        nn.init.zeros_(self.cepstral_projection.weight)
        nn.init.zeros_(self.cepstral_projection.bias)

    def _validate_inputs(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> None:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, frames, mel_bins]")
        if mel.shape[-1] != self.mel_bins:
            raise ValueError(f"mel bin mismatch: {mel.shape[-1]} != {self.mel_bins}")
        frame_shape = mel.shape[:2]
        if f0_hz.shape != frame_shape:
            raise ValueError("f0_hz must match mel [batch, frames]")
        if voiced.shape != frame_shape:
            raise ValueError("voiced must match mel [batch, frames]")
        if periodicity.shape != frame_shape:
            raise ValueError("periodicity must match mel [batch, frames]")
        for name, value in (
            ("mel", mel),
            ("f0_hz", f0_hz),
            ("voiced", voiced),
            ("periodicity", periodicity),
        ):
            if value.is_complex() or not value.is_floating_point():
                raise ValueError(f"{name} must be a real floating tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")

    def conditioning_features(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(mel, f0_hz, voiced, periodicity)
        # Bound pitch conditioning without changing the cached pitch semantics.  1000 Hz is
        # only a numerical scale; no F0 is re-estimated or modified in the data contract.
        log_f0 = torch.log1p(f0_hz.clamp_min(0.0)) / math.log(1001.0)
        features = torch.cat(
            (
                mel,
                log_f0.unsqueeze(-1),
                voiced.clamp(0.0, 1.0).unsqueeze(-1),
                periodicity.clamp(0.0, 1.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        return features

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> torch.Tensor:
        features = self.conditioning_features(mel, f0_hz, voiced, periodicity)
        x = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.pre_output(F.gelu(x)))
        cepstrum = self.cepstral_projection(x).transpose(1, 2)
        if not torch.isfinite(cepstrum).all():
            raise ValueError("cepstral predictor produced non-finite values")
        return cepstrum
