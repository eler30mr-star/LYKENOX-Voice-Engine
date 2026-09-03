"""LYKENOX-owned residual-statistics source model.

Direct residual forensics closed deterministic waveform regression at the acoustic-frame grid:
both overlapping 512/256 vectors and unique 256-sample blocks collapse toward a repeated per-frame
waveform template, producing a 93.75 Hz comb.  The real Step-3f residual instead retains high-band
aperiodic phase that is not determined by mel/F0 conditioning.

This model therefore predicts only source statistics that are identifiable from acoustic context:
- one-sided real cepstral spectral shape of the residual source;
- explicit residual log-RMS;
- residual periodic energy fraction.

It predicts no waveform samples and owns no random phase.  A separate deterministic LYKENOX source
synthesizer turns these statistics into a continuous carrier/filter stream with no per-frame phase
reset.  No external model/weight/service, codebook, post-hoc enhancement or duration modification is
used. Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE = "lykenox_owned_residual_statistics_source_v1"
MEL_BINS = 80
SOURCE_CEPSTRAL_ORDER = 64
DEFAULT_CONTEXT_CHANNELS = 160
DEFAULT_STATE_CHANNELS = 192
DEFAULT_DILATIONS = (1, 2, 4, 8)
MIN_LOG_RMS = -9.0
MAX_LOG_RMS = 0.5
INITIAL_LOG_RMS = -3.5
MAX_SOURCE_CEPSTRUM = 2.5


def _conditioning_features(
    mel: torch.Tensor,
    f0_track_hz: torch.Tensor,
    energy_confidence: torch.Tensor,
    periodic_strength: torch.Tensor,
) -> torch.Tensor:
    if mel.ndim != 3 or mel.shape[-1] != MEL_BINS:
        raise ValueError(f"mel must have shape [batch, frames, {MEL_BINS}]")
    shape = mel.shape[:2]
    if f0_track_hz.shape != shape or energy_confidence.shape != shape or periodic_strength.shape != shape:
        raise ValueError("conditioning must match mel [batch, frames]")
    for name, value in (
        ("mel", mel),
        ("f0_track_hz", f0_track_hz),
        ("energy_confidence", energy_confidence),
        ("periodic_strength", periodic_strength),
    ):
        if not value.is_floating_point() or value.is_complex() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite real floating data")
    log_f0 = torch.log1p(f0_track_hz.clamp_min(0.0)) / math.log(1001.0)
    return torch.cat(
        (
            mel,
            log_f0.unsqueeze(-1),
            energy_confidence.clamp(0.0, 1.0).unsqueeze(-1),
            periodic_strength.clamp(0.0, 1.0).unsqueeze(-1),
        ),
        dim=-1,
    )


class _ContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.in_gate = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(F.gelu(x))
        value, gate = self.in_gate(y).chunk(2, dim=1)
        y = torch.tanh(value) * torch.sigmoid(gate)
        return (residual + self.out(y)) * (2.0 ** -0.5)


def _initial_level_logit() -> float:
    probability = (INITIAL_LOG_RMS - MIN_LOG_RMS) / (MAX_LOG_RMS - MIN_LOG_RMS)
    probability = min(1.0 - 1.0e-6, max(1.0e-6, probability))
    return math.log(probability / (1.0 - probability))


class LykenoxResidualStatisticsSourceV1(nn.Module):
    """Predict residual distribution statistics at acoustic frame rate, never waveform samples."""

    architecture = RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE

    def __init__(
        self,
        *,
        context_channels: int = DEFAULT_CONTEXT_CHANNELS,
        state_channels: int = DEFAULT_STATE_CHANNELS,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        if context_channels < 64 or state_channels < 64:
            raise ValueError("residual-statistics dimensions are below architecture floor")
        if not dilations or any(int(value) < 1 for value in dilations):
            raise ValueError("dilations must be positive")
        self.context_channels = int(context_channels)
        self.state_channels = int(state_channels)
        self.dilations = tuple(int(value) for value in dilations)

        self.conditioning_projection = nn.Conv1d(MEL_BINS + 3, context_channels, kernel_size=1)
        self.context_blocks = nn.ModuleList(
            [_ContextBlock(context_channels, dilation) for dilation in self.dilations]
        )
        self.recurrent = nn.GRU(
            input_size=context_channels,
            hidden_size=state_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.pre_output = nn.Sequential(
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
        )
        self.source_cepstrum_projection = nn.Linear(state_channels, SOURCE_CEPSTRAL_ORDER - 1)
        self.level_projection = nn.Linear(state_channels + context_channels, 1)
        self.periodicity_projection = nn.Linear(state_channels + context_channels, 1)

        nn.init.zeros_(self.source_cepstrum_projection.weight)
        nn.init.zeros_(self.source_cepstrum_projection.bias)
        nn.init.zeros_(self.level_projection.weight)
        nn.init.constant_(self.level_projection.bias, _initial_level_logit())
        nn.init.zeros_(self.periodicity_projection.weight)
        nn.init.constant_(self.periodicity_projection.bias, -2.0)

    @staticmethod
    def _bounded_log_rms(raw: torch.Tensor) -> torch.Tensor:
        return MIN_LOG_RMS + (MAX_LOG_RMS - MIN_LOG_RMS) * torch.sigmoid(raw)

    def encode_conditioning(
        self,
        mel: torch.Tensor,
        f0_track_hz: torch.Tensor,
        energy_confidence: torch.Tensor,
        periodic_strength: torch.Tensor,
    ) -> torch.Tensor:
        features = _conditioning_features(mel, f0_track_hz, energy_confidence, periodic_strength)
        x = self.conditioning_projection(features.transpose(1, 2))
        for block in self.context_blocks:
            x = block(x)
        return x.transpose(1, 2).contiguous()

    def forward(
        self,
        mel: torch.Tensor,
        f0_track_hz: torch.Tensor,
        energy_confidence: torch.Tensor,
        periodic_strength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_conditioning(mel, f0_track_hz, energy_confidence, periodic_strength)
        state, _ = self.recurrent(context)
        hidden = self.pre_output(state)
        head_input = torch.cat((hidden, context), dim=-1)

        positive_quefrency = MAX_SOURCE_CEPSTRUM * torch.tanh(self.source_cepstrum_projection(hidden))
        zero = torch.zeros(*positive_quefrency.shape[:-1], 1, dtype=positive_quefrency.dtype, device=positive_quefrency.device)
        source_cepstrum = torch.cat((zero, positive_quefrency), dim=-1)
        log_rms = self._bounded_log_rms(self.level_projection(head_input)).squeeze(-1)
        residual_periodicity = torch.sigmoid(self.periodicity_projection(head_input)).squeeze(-1)

        batch, frames = mel.shape[:2]
        if source_cepstrum.shape != (batch, frames, SOURCE_CEPSTRAL_ORDER):
            raise RuntimeError("source cepstrum geometry changed")
        if log_rms.shape != (batch, frames) or residual_periodicity.shape != (batch, frames):
            raise RuntimeError("residual statistics scalar geometry changed")
        if not bool(torch.isfinite(source_cepstrum).all() and torch.isfinite(log_rms).all() and torch.isfinite(residual_periodicity).all()):
            raise RuntimeError("residual statistics source produced non-finite values")
        return source_cepstrum.contiguous(), log_rms.contiguous(), residual_periodicity.contiguous()


__all__ = [
    "MAX_SOURCE_CEPSTRUM",
    "RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE",
    "SOURCE_CEPSTRAL_ORDER",
    "LykenoxResidualStatisticsSourceV1",
]
