"""LYKENOX-owned non-overlapping continuous residual stream source.

The previous Continuous Source V2 predicts 512-sample vectors every 256 samples. Direct forensics
showed that learned adjacent vectors disagree about their duplicated overlap and produce a 93.75 Hz
hop-grid comb before the fixed minimum-phase renderer. The real Step-3f target vectors do not have
that defect.

This source removes duplicated sample authority completely:
- one acoustic frame predicts exactly one contiguous 256-sample residual block;
- blocks are concatenated directly, with no analysis/synthesis window and no overlap-add;
- one causal recurrent acoustic state carries continuity across blocks;
- previous waveform blocks are not fed back into the recurrent input;
- residual shape and absolute RMS remain explicitly factorized.

Conditioning uses the owned pitch-conditioning-v2 semantic contract through the same three numeric
slots (continuous F0 track, energy confidence, periodic strength). No external model/weight/service,
codebook, stochastic post-processing, gain normalization, EQ, denoise or duration modification is
used. Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


CONTINUOUS_STREAM_SOURCE_ARCHITECTURE = "lykenox_owned_continuous_residual_stream_source_v1"
MEL_BINS = 80
HOP_LENGTH = 256
BLOCK_SAMPLES = HOP_LENGTH
DEFAULT_CONTEXT_CHANNELS = 160
DEFAULT_STATE_CHANNELS = 192
DEFAULT_DILATIONS = (1, 2, 4, 8)
MIN_LOG_RMS = -9.0
MAX_LOG_RMS = 0.5
INITIAL_LOG_RMS = -3.5
EPSILON = 1.0e-8


def _conditioning_features(
    mel: torch.Tensor,
    f0_track_hz: torch.Tensor,
    energy_confidence: torch.Tensor,
    periodic_strength: torch.Tensor,
) -> torch.Tensor:
    if mel.ndim != 3 or mel.shape[-1] != MEL_BINS:
        raise ValueError(f"mel must have shape [batch, frames, {MEL_BINS}]")
    shape = mel.shape[:2]
    if (
        f0_track_hz.shape != shape
        or energy_confidence.shape != shape
        or periodic_strength.shape != shape
    ):
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
        y = self.out(y)
        return (residual + y) * (2.0 ** -0.5)


def _initial_level_logit() -> float:
    probability = (INITIAL_LOG_RMS - MIN_LOG_RMS) / (MAX_LOG_RMS - MIN_LOG_RMS)
    probability = min(1.0 - 1.0e-6, max(1.0e-6, probability))
    return math.log(probability / (1.0 - probability))


class LykenoxContinuousResidualStreamSourceV1(nn.Module):
    """Causal frame-state model producing one unique contiguous residual block per frame."""

    architecture = CONTINUOUS_STREAM_SOURCE_ARCHITECTURE

    def __init__(
        self,
        *,
        context_channels: int = DEFAULT_CONTEXT_CHANNELS,
        state_channels: int = DEFAULT_STATE_CHANNELS,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        if context_channels < 64 or state_channels < 64:
            raise ValueError("continuous stream dimensions are below the architecture floor")
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
        self.block_shape_projection = nn.Linear(state_channels, BLOCK_SAMPLES)
        self.block_level_projection = nn.Linear(state_channels + context_channels, 1)

        nn.init.normal_(self.block_shape_projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.block_shape_projection.bias)
        nn.init.zeros_(self.block_level_projection.weight)
        nn.init.constant_(self.block_level_projection.bias, _initial_level_logit())

    @staticmethod
    def _unit_rms_shape(value: torch.Tensor) -> torch.Tensor:
        mean_square = value.square().mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(mean_square + EPSILON)

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
        features = _conditioning_features(
            mel, f0_track_hz, energy_confidence, periodic_strength
        )
        x = self.conditioning_projection(features.transpose(1, 2))
        for block in self.context_blocks:
            x = block(x)
        return x.transpose(1, 2).contiguous()

    def forward_with_log_rms(
        self,
        mel: torch.Tensor,
        f0_track_hz: torch.Tensor,
        energy_confidence: torch.Tensor,
        periodic_strength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_conditioning(
            mel, f0_track_hz, energy_confidence, periodic_strength
        )
        state, _ = self.recurrent(context)
        hidden = self.pre_output(state)
        unit_shape = self._unit_rms_shape(self.block_shape_projection(hidden))
        log_rms = self._bounded_log_rms(
            self.block_level_projection(torch.cat((hidden, context), dim=-1))
        ).squeeze(-1)
        blocks = unit_shape * torch.exp(log_rms).unsqueeze(-1)
        batch, frames = mel.shape[:2]
        if blocks.shape != (batch, frames, BLOCK_SAMPLES):
            raise RuntimeError("continuous stream block geometry changed")
        if log_rms.shape != (batch, frames):
            raise RuntimeError("continuous stream level geometry changed")
        if not bool(torch.isfinite(blocks).all() and torch.isfinite(log_rms).all()):
            raise RuntimeError("continuous stream source produced non-finite values")
        return blocks.contiguous(), log_rms.contiguous()

    def forward(
        self,
        mel: torch.Tensor,
        f0_track_hz: torch.Tensor,
        energy_confidence: torch.Tensor,
        periodic_strength: torch.Tensor,
    ) -> torch.Tensor:
        blocks, _ = self.forward_with_log_rms(
            mel, f0_track_hz, energy_confidence, periodic_strength
        )
        return blocks

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        f0_track_hz: torch.Tensor,
        energy_confidence: torch.Tensor,
        periodic_strength: torch.Tensor,
    ) -> torch.Tensor:
        return self(mel, f0_track_hz, energy_confidence, periodic_strength)


def blocks_to_residual(blocks: torch.Tensor) -> torch.Tensor:
    """Concatenate unique blocks into the continuous residual stream; no overlap/add exists."""
    if blocks.ndim != 3 or blocks.shape[-1] != BLOCK_SAMPLES:
        raise ValueError("blocks must be [batch, frames, 256]")
    residual = blocks.reshape(blocks.shape[0], blocks.shape[1] * BLOCK_SAMPLES)
    if not bool(torch.isfinite(residual).all()):
        raise RuntimeError("continuous stream residual contains non-finite values")
    return residual.contiguous()


__all__ = [
    "BLOCK_SAMPLES",
    "CONTINUOUS_STREAM_SOURCE_ARCHITECTURE",
    "HOP_LENGTH",
    "LykenoxContinuousResidualStreamSourceV1",
    "blocks_to_residual",
]
