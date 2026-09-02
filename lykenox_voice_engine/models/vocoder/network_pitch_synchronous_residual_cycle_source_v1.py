"""LYKENOX-owned pitch-synchronous residual-cycle source.

This source addresses the remaining robotic timbre of the level-factored continuous source V2 by
removing absolute sample-phase ambiguity from voiced residual learning.  Voiced Step-3f residual is
represented one F0 cycle at a time in a fixed phase coordinate.  The model predicts real owned cycle
shape and explicit cycle RMS; it does not impose Rosenberg or any other parametric pulse family.

The acoustic frame encoder may be warm-started from the owned V2 source checkpoint.  Cycle dynamics
are learned from scratch from owned TRAIN residual cycles.  There is no codebook, external model,
pretrained third-party weight, remote service, post-hoc enhancement, or duration modification.
Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE = "lykenox_owned_pitch_synchronous_residual_cycle_source_v1"
MEL_BINS = 80
CYCLE_PHASE_BINS = 128
DEFAULT_CONTEXT_CHANNELS = 160
DEFAULT_STATE_CHANNELS = 192
DEFAULT_PREVIOUS_EMBED = 96
DEFAULT_DILATIONS = (1, 2, 4, 8)
MIN_LOG_RMS = -9.0
MAX_LOG_RMS = 0.5
INITIAL_LOG_RMS = -3.5
SHAPE_EPSILON = 1.0e-8


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


class LykenoxPitchSynchronousResidualCycleSourceV1(nn.Module):
    """Predict real residual cycles in canonical F0 phase coordinates."""

    architecture = PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE

    def __init__(
        self,
        *,
        context_channels: int = DEFAULT_CONTEXT_CHANNELS,
        state_channels: int = DEFAULT_STATE_CHANNELS,
        previous_embed: int = DEFAULT_PREVIOUS_EMBED,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        if context_channels < 64 or state_channels < 64 or previous_embed < 32:
            raise ValueError("pitch-synchronous source dimensions are below the architecture floor")
        if not dilations or any(int(value) < 1 for value in dilations):
            raise ValueError("dilations must be positive")
        self.context_channels = int(context_channels)
        self.state_channels = int(state_channels)
        self.previous_embed = int(previous_embed)
        self.dilations = tuple(int(value) for value in dilations)

        self.conditioning_projection = nn.Conv1d(MEL_BINS + 3, context_channels, kernel_size=1)
        self.context_blocks = nn.ModuleList(
            [_ContextBlock(context_channels, dilation) for dilation in self.dilations]
        )
        self.previous_cycle_projection = nn.Linear(CYCLE_PHASE_BINS, previous_embed)
        self.recurrent = nn.GRUCell(context_channels + previous_embed, state_channels)
        self.pre_output = nn.Sequential(
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
        )
        self.cycle_shape_projection = nn.Linear(state_channels, CYCLE_PHASE_BINS)
        self.cycle_level_projection = nn.Linear(state_channels + context_channels, 1)

        # Start with a finite nonzero cycle level. Cycle shape is intentionally not zero-initialized:
        # zero followed by unit-RMS normalization would create a degenerate gradient origin.
        nn.init.normal_(self.cycle_shape_projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.cycle_shape_projection.bias)
        nn.init.zeros_(self.cycle_level_projection.weight)
        probability = (INITIAL_LOG_RMS - MIN_LOG_RMS) / (MAX_LOG_RMS - MIN_LOG_RMS)
        probability = min(1.0 - 1.0e-6, max(1.0e-6, probability))
        nn.init.constant_(self.cycle_level_projection.bias, math.log(probability / (1.0 - probability)))

    @staticmethod
    def unit_rms_shape(value: torch.Tensor) -> torch.Tensor:
        mean_square = value.square().mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(mean_square + SHAPE_EPSILON)

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

    def _bounded_log_rms(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        raw = self.cycle_level_projection(torch.cat((hidden, context), dim=-1))
        return MIN_LOG_RMS + (MAX_LOG_RMS - MIN_LOG_RMS) * torch.sigmoid(raw)

    def forward_cycles(
        self,
        frame_context: torch.Tensor,
        cycle_frame_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate canonical cycles for one utterance.

        ``frame_context`` is [frames, context_channels] and ``cycle_frame_indices`` maps every real
        F0 cycle to the nearest acoustic frame.  Generation is fully autoregressive over canonical
        cycle SHAPE only; absolute cycle level is explicit and never fed back.
        """
        if frame_context.ndim != 2 or frame_context.shape[-1] != self.context_channels:
            raise ValueError("frame_context has wrong geometry")
        if cycle_frame_indices.ndim != 1 or cycle_frame_indices.dtype != torch.long:
            raise ValueError("cycle_frame_indices must be a one-dimensional long tensor")
        if cycle_frame_indices.numel() < 1:
            raise ValueError("at least one pitch-synchronous cycle is required")
        if int(cycle_frame_indices.min()) < 0 or int(cycle_frame_indices.max()) >= int(frame_context.shape[0]):
            raise ValueError("cycle frame index outside acoustic context")

        state = torch.zeros(1, self.state_channels, dtype=frame_context.dtype, device=frame_context.device)
        previous = torch.zeros(1, CYCLE_PHASE_BINS, dtype=frame_context.dtype, device=frame_context.device)
        cycles: list[torch.Tensor] = []
        levels: list[torch.Tensor] = []
        for frame_index in cycle_frame_indices.tolist():
            context = frame_context[int(frame_index)].unsqueeze(0)
            previous_shape = self.unit_rms_shape(previous)
            previous_embed = F.gelu(self.previous_cycle_projection(previous_shape))
            state = self.recurrent(torch.cat((context, previous_embed), dim=-1), state)
            hidden = self.pre_output(state)
            shape = self.unit_rms_shape(self.cycle_shape_projection(hidden))
            log_rms = self._bounded_log_rms(hidden, context)
            current = shape * torch.exp(log_rms)
            cycles.append(current.squeeze(0))
            levels.append(log_rms.squeeze())
            previous = current
        cycle_tensor = torch.stack(cycles, dim=0).contiguous()
        level_tensor = torch.stack(levels, dim=0).contiguous()
        if cycle_tensor.shape != (cycle_frame_indices.numel(), CYCLE_PHASE_BINS):
            raise RuntimeError("pitch-synchronous cycle geometry changed")
        if level_tensor.shape != (cycle_frame_indices.numel(),):
            raise RuntimeError("pitch-synchronous level geometry changed")
        if not bool(torch.isfinite(cycle_tensor).all() and torch.isfinite(level_tensor).all()):
            raise RuntimeError("pitch-synchronous source produced non-finite values")
        return cycle_tensor, level_tensor

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
        cycle_frame_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mel.shape[0] != 1:
            raise ValueError("pitch-synchronous v1 currently operates one utterance at a time")
        context = self.encode_conditioning(mel, f0_hz, voiced, periodicity)[0]
        return self.forward_cycles(context, cycle_frame_indices)


__all__ = [
    "CYCLE_PHASE_BINS",
    "PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE",
    "LykenoxPitchSynchronousResidualCycleSourceV1",
]
