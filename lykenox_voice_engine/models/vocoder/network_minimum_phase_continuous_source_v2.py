"""LYKENOX-owned continuous residual source with explicit residual level factorization.

V1 predicted each 512-sample residual vector with one unconstrained head.  That coupled fine
structure and absolute level, allowing a low-energy solution even while shape/spectral terms
improved.  V2 factorizes every residual vector into:

    residual_vector = unit_rms_shape * exp(predicted_log_rms)

The shape remains autoregressive at frame rate for phase/fine-structure continuity.  Absolute
level is predicted by a separate acoustically-conditioned head and is not fed back from the
previous vector amplitude, preventing recursive level collapse.  There is no codebook, sample-rate
neural generator, post-hoc gain, external model, pretrained weight, or remote service.
Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


CONTINUOUS_SOURCE_ARCHITECTURE_V2 = "lykenox_owned_continuous_residual_source_v2_level_factored"
MEL_BINS = 80
HOP_LENGTH = 256
RESIDUAL_VECTOR_SAMPLES = 512
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


class LykenoxContinuousResidualSourceV2(nn.Module):
    """Frame-rate autoregressive residual-shape model with explicit acoustic level head."""

    architecture = CONTINUOUS_SOURCE_ARCHITECTURE_V2

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
            raise ValueError("continuous source dimensions are below the architecture floor")
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

        # Only previous SHAPE is recurrent. Previous absolute level is deliberately excluded so
        # a low-energy frame cannot recursively pull all following frames toward silence.
        self.previous_projection = nn.Linear(RESIDUAL_VECTOR_SAMPLES, previous_embed)
        self.recurrent = nn.GRUCell(context_channels + previous_embed, state_channels)
        self.pre_output = nn.Sequential(
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
        )
        self.shape_projection = nn.Linear(state_channels, RESIDUAL_VECTOR_SAMPLES)

        # Level is its own supervised variable. It sees current acoustic context + recurrent
        # shape state, but not previous residual amplitude.
        self.level_projection = nn.Linear(state_channels + context_channels, 1)
        nn.init.zeros_(self.level_projection.weight)
        probability = (INITIAL_LOG_RMS - MIN_LOG_RMS) / (MAX_LOG_RMS - MIN_LOG_RMS)
        probability = min(1.0 - 1.0e-6, max(1.0e-6, probability))
        initial_logit = math.log(probability / (1.0 - probability))
        nn.init.constant_(self.level_projection.bias, initial_logit)

    @staticmethod
    def _unit_rms_shape(vector: torch.Tensor) -> torch.Tensor:
        mean_square = vector.square().mean(dim=-1, keepdim=True)
        return vector * torch.rsqrt(mean_square + SHAPE_EPSILON)

    @staticmethod
    def _teacher_mask(step: int, ratio: float, seed: int) -> bool:
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        threshold = int(round(float(ratio) * 1000.0))
        value = (int(seed) + int(step) * 977 + 431) % 1000
        return value < threshold

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

    def _bounded_log_rms(self, state_hidden: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        raw = self.level_projection(torch.cat((state_hidden, conditioning), dim=-1))
        return MIN_LOG_RMS + (MAX_LOG_RMS - MIN_LOG_RMS) * torch.sigmoid(raw)

    def forward_with_log_rms(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
        *,
        teacher_vectors: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        teacher_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= float(teacher_forcing_ratio) <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0,1]")
        context = self.encode_conditioning(mel, f0_hz, voiced, periodicity)
        batch, frames, _ = context.shape
        vector_count = frames + 1
        if teacher_vectors is not None:
            expected = (batch, vector_count, RESIDUAL_VECTOR_SAMPLES)
            if teacher_vectors.shape != expected:
                raise ValueError(f"teacher_vectors must have shape {expected}")
            if not bool(torch.isfinite(teacher_vectors).all()):
                raise ValueError("teacher_vectors contain non-finite values")
            teacher_vectors = teacher_vectors.to(context.dtype)

        state = torch.zeros(batch, self.state_channels, dtype=context.dtype, device=context.device)
        previous = torch.zeros(batch, RESIDUAL_VECTOR_SAMPLES, dtype=context.dtype, device=context.device)
        outputs: list[torch.Tensor] = []
        levels: list[torch.Tensor] = []

        for step in range(vector_count):
            if step > 0 and teacher_vectors is not None and self._teacher_mask(
                step, float(teacher_forcing_ratio), int(teacher_seed)
            ):
                previous = teacher_vectors[:, step - 1]
            previous_shape = self._unit_rms_shape(previous)
            previous_embed = F.gelu(self.previous_projection(previous_shape))
            conditioning = context[:, min(step, frames - 1)]
            state = self.recurrent(torch.cat((conditioning, previous_embed), dim=-1), state)
            hidden = self.pre_output(state)
            unit_shape = self._unit_rms_shape(self.shape_projection(hidden))
            log_rms = self._bounded_log_rms(hidden, conditioning)
            current = unit_shape * torch.exp(log_rms)
            outputs.append(current)
            levels.append(log_rms.squeeze(-1))
            previous = current

        vectors = torch.stack(outputs, dim=1).contiguous()
        log_rms = torch.stack(levels, dim=1).contiguous()
        if vectors.shape != (batch, vector_count, RESIDUAL_VECTOR_SAMPLES):
            raise RuntimeError("continuous residual source v2 geometry changed")
        if log_rms.shape != (batch, vector_count):
            raise RuntimeError("continuous residual source v2 level geometry changed")
        if not bool(torch.isfinite(vectors).all() and torch.isfinite(log_rms).all()):
            raise RuntimeError("continuous residual source v2 produced non-finite values")
        return vectors, log_rms

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
        *,
        teacher_vectors: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        teacher_seed: int = 0,
    ) -> torch.Tensor:
        vectors, _ = self.forward_with_log_rms(
            mel,
            f0_hz,
            voiced,
            periodicity,
            teacher_vectors=teacher_vectors,
            teacher_forcing_ratio=teacher_forcing_ratio,
            teacher_seed=teacher_seed,
        )
        return vectors

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> torch.Tensor:
        return self(
            mel,
            f0_hz,
            voiced,
            periodicity,
            teacher_vectors=None,
            teacher_forcing_ratio=0.0,
        )


__all__ = [
    "CONTINUOUS_SOURCE_ARCHITECTURE_V2",
    "HOP_LENGTH",
    "INITIAL_LOG_RMS",
    "MAX_LOG_RMS",
    "MEL_BINS",
    "MIN_LOG_RMS",
    "RESIDUAL_VECTOR_SAMPLES",
    "LykenoxContinuousResidualSourceV2",
]
