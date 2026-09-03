"""LYKENOX-owned unified phase-aware residual source.

This model replaces the rejected hybrid handoff between an independently trained pitch-synchronous
source and V2.  Periodic and aperiodic behavior are coordinates of one acoustic state and one jointly
optimized residual source:

- one conditioning encoder and one recurrent state;
- a phase-harmonic frame head evaluated on accumulated F0 phase;
- an aperiodic 512/256 overlap-add head from the same hidden state;
- explicit level factorization for both coordinates;
- no second source checkpoint, source handoff, bridge, codebook, stochastic innovation, external
  model/weight/service, or post-hoc enhancement.

The final residual is synthesized by the trainer/renderer with complementary energy weights
sqrt(periodicity) and sqrt(1-periodicity).  Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


UNIFIED_PHASE_SOURCE_ARCHITECTURE = "lykenox_owned_unified_phase_residual_source_v1"
MEL_BINS = 80
HOP_LENGTH = 256
RESIDUAL_VECTOR_SAMPLES = 512
HARMONIC_COUNT = 24
DEFAULT_CONTEXT_CHANNELS = 160
DEFAULT_STATE_CHANNELS = 192
DEFAULT_DILATIONS = (1, 2, 4, 8)
MIN_LOG_RMS = -9.0
MAX_LOG_RMS = 0.5
INITIAL_LOG_RMS = -3.5
EPSILON = 1.0e-8


def _conditioning_features(
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
) -> torch.Tensor:
    if mel.ndim != 3 or mel.shape[-1] != MEL_BINS:
        raise ValueError(f"mel must have shape [batch, frames, {MEL_BINS}]")
    shape = mel.shape[:2]
    if f0_hz.shape != shape or voiced.shape != shape or periodicity.shape != shape:
        raise ValueError("F0/voiced/periodicity must match mel [batch, frames]")
    for name, value in (
        ("mel", mel),
        ("f0_hz", f0_hz),
        ("voiced", voiced),
        ("periodicity", periodicity),
    ):
        if not value.is_floating_point() or value.is_complex() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite real floating data")
    log_f0 = torch.log1p(f0_hz.clamp_min(0.0)) / math.log(1001.0)
    return torch.cat(
        (
            mel,
            log_f0.unsqueeze(-1),
            voiced.clamp(0.0, 1.0).unsqueeze(-1),
            periodicity.clamp(0.0, 1.0).unsqueeze(-1),
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
        y = self.out(y)
        return (residual + y) * (2.0 ** -0.5)


def _initial_level_logit() -> float:
    probability = (INITIAL_LOG_RMS - MIN_LOG_RMS) / (MAX_LOG_RMS - MIN_LOG_RMS)
    probability = min(1.0 - 1.0e-6, max(1.0e-6, probability))
    return math.log(probability / (1.0 - probability))


class LykenoxUnifiedPhaseResidualSourceV1(nn.Module):
    """One acoustic state generating periodic phase structure and aperiodic residual texture."""

    architecture = UNIFIED_PHASE_SOURCE_ARCHITECTURE

    def __init__(
        self,
        *,
        context_channels: int = DEFAULT_CONTEXT_CHANNELS,
        state_channels: int = DEFAULT_STATE_CHANNELS,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        if context_channels < 64 or state_channels < 64:
            raise ValueError("unified source dimensions are below the architecture floor")
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

        # Periodic coordinate: cosine/sine pairs normalized to unit Fourier RMS, then scaled by
        # an explicit predicted log-RMS.  The source phase itself is never predicted; it comes from
        # accumulated owned F0 conditioning in the fixed synthesis function.
        self.harmonic_projection = nn.Linear(state_channels, HARMONIC_COUNT * 2)
        self.periodic_level_projection = nn.Linear(state_channels + context_channels, 1)

        # Aperiodic coordinate: the proven 512/256 OLA representation with explicit level.  It is
        # generated from the SAME hidden state as the harmonic coordinate, not from V2 or another
        # checkpoint.  A dedicated terminal vector preserves the exact frames+1 OLA geometry.
        self.aperiodic_shape_projection = nn.Linear(state_channels, RESIDUAL_VECTOR_SAMPLES)
        self.aperiodic_level_projection = nn.Linear(state_channels + context_channels, 1)
        self.terminal_shape_projection = nn.Linear(state_channels, RESIDUAL_VECTOR_SAMPLES)
        self.terminal_level_projection = nn.Linear(state_channels + context_channels, 1)

        nn.init.normal_(self.harmonic_projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.harmonic_projection.bias)
        nn.init.normal_(self.aperiodic_shape_projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.aperiodic_shape_projection.bias)
        nn.init.normal_(self.terminal_shape_projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.terminal_shape_projection.bias)
        for layer in (
            self.periodic_level_projection,
            self.aperiodic_level_projection,
            self.terminal_level_projection,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, _initial_level_logit())

    @staticmethod
    def _bounded_log_rms(raw: torch.Tensor) -> torch.Tensor:
        return MIN_LOG_RMS + (MAX_LOG_RMS - MIN_LOG_RMS) * torch.sigmoid(raw)

    @staticmethod
    def _unit_rms_shape(value: torch.Tensor) -> torch.Tensor:
        mean_square = value.square().mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(mean_square + EPSILON)

    @staticmethod
    def _unit_fourier_rms(pairs: torch.Tensor) -> torch.Tensor:
        # For sum_k a_k cos(k phi)+b_k sin(k phi), RMS^2 over phase is
        # 0.5 * sum_k(a_k^2+b_k^2).  Normalize that quantity to one.
        power = 0.5 * pairs.square().sum(dim=(-1, -2), keepdim=True)
        return pairs * torch.rsqrt(power + EPSILON)

    def encode_conditioning(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> torch.Tensor:
        features = _conditioning_features(mel, f0_hz, voiced, periodicity)
        x = self.conditioning_projection(features.transpose(1, 2))
        for block in self.context_blocks:
            x = block(x)
        return x.transpose(1, 2).contiguous()

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_conditioning(mel, f0_hz, voiced, periodicity)
        state, _ = self.recurrent(context)
        hidden = self.pre_output(state)
        batch, frames, _ = hidden.shape

        raw_pairs = self.harmonic_projection(hidden).view(batch, frames, HARMONIC_COUNT, 2)
        unit_pairs = self._unit_fourier_rms(raw_pairs)
        periodic_log_rms = self._bounded_log_rms(
            self.periodic_level_projection(torch.cat((hidden, context), dim=-1))
        ).squeeze(-1)
        harmonic_pairs = unit_pairs * torch.exp(periodic_log_rms).unsqueeze(-1).unsqueeze(-1)

        unit_aperiodic = self._unit_rms_shape(self.aperiodic_shape_projection(hidden))
        aperiodic_log_rms_frames = self._bounded_log_rms(
            self.aperiodic_level_projection(torch.cat((hidden, context), dim=-1))
        ).squeeze(-1)
        aperiodic_frames = unit_aperiodic * torch.exp(aperiodic_log_rms_frames).unsqueeze(-1)

        last_hidden = hidden[:, -1]
        last_context = context[:, -1]
        terminal_shape = self._unit_rms_shape(self.terminal_shape_projection(last_hidden))
        terminal_log_rms = self._bounded_log_rms(
            self.terminal_level_projection(torch.cat((last_hidden, last_context), dim=-1))
        ).squeeze(-1)
        terminal = terminal_shape * torch.exp(terminal_log_rms).unsqueeze(-1)
        aperiodic_vectors = torch.cat((aperiodic_frames, terminal.unsqueeze(1)), dim=1).contiguous()
        aperiodic_log_rms = torch.cat((aperiodic_log_rms_frames, terminal_log_rms.unsqueeze(1)), dim=1)

        if harmonic_pairs.shape != (batch, frames, HARMONIC_COUNT, 2):
            raise RuntimeError("unified harmonic geometry changed")
        if periodic_log_rms.shape != (batch, frames):
            raise RuntimeError("unified periodic level geometry changed")
        if aperiodic_vectors.shape != (batch, frames + 1, RESIDUAL_VECTOR_SAMPLES):
            raise RuntimeError("unified aperiodic vector geometry changed")
        if aperiodic_log_rms.shape != (batch, frames + 1):
            raise RuntimeError("unified aperiodic level geometry changed")
        if not bool(
            torch.isfinite(harmonic_pairs).all()
            and torch.isfinite(periodic_log_rms).all()
            and torch.isfinite(aperiodic_vectors).all()
            and torch.isfinite(aperiodic_log_rms).all()
        ):
            raise RuntimeError("unified phase source produced non-finite values")
        return harmonic_pairs, periodic_log_rms, aperiodic_vectors, aperiodic_log_rms


__all__ = [
    "HARMONIC_COUNT",
    "HOP_LENGTH",
    "MEL_BINS",
    "RESIDUAL_VECTOR_SAMPLES",
    "UNIFIED_PHASE_SOURCE_ARCHITECTURE",
    "LykenoxUnifiedPhaseResidualSourceV1",
]
