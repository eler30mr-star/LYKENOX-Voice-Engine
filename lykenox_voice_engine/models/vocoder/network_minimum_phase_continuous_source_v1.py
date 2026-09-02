"""LYKENOX-owned continuous residual source for the fixed minimum-phase renderer.

This replaces discrete residual-codebook retrieval.  It predicts the owned Step-3f real residual
continuously as a sequence of 512-sample sqrt-Hann analysis vectors at the existing 256-sample hop.
The sequence is autoregressive at *frame rate*: the previous residual vector is encoded and carried
through a GRU state, so phase/fine-structure continuity is modeled directly instead of approximated by
nearest-neighbour codewords.

The model consumes only owned acoustic conditioning available to the vocoder (mel, F0, voiced,
periodicity).  It contains no external model, pretrained weight, remote service, codebook index,
sample-rate neural upsampler, transpose convolution, post-hoc enhancement, or duration modification.
Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


CONTINUOUS_SOURCE_ARCHITECTURE = "lykenox_owned_continuous_residual_source_v1"
MEL_BINS = 80
HOP_LENGTH = 256
RESIDUAL_VECTOR_SAMPLES = 512
DEFAULT_CONTEXT_CHANNELS = 160
DEFAULT_STATE_CHANNELS = 192
DEFAULT_PREVIOUS_EMBED = 96
DEFAULT_DILATIONS = (1, 2, 4, 8)


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
    for name, value in (("mel", mel), ("f0_hz", f0_hz), ("voiced", voiced), ("periodicity", periodicity)):
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


class LykenoxContinuousResidualSourceV1(nn.Module):
    """Frame-rate autoregressive generator of continuous residual analysis vectors."""

    architecture = CONTINUOUS_SOURCE_ARCHITECTURE

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
        # Previous-vector input is shape-normalized plus explicit log-RMS.  This preserves phase/
        # fine structure while preventing absolute level from dominating the recurrent state.
        self.previous_projection = nn.Linear(RESIDUAL_VECTOR_SAMPLES + 1, previous_embed)
        self.recurrent = nn.GRUCell(context_channels + previous_embed, state_channels)
        self.pre_output = nn.Sequential(
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
        )
        self.vector_projection = nn.Linear(state_channels, RESIDUAL_VECTOR_SAMPLES)

        # Start from neutral excitation rather than a random high-energy source.
        nn.init.zeros_(self.vector_projection.weight)
        nn.init.zeros_(self.vector_projection.bias)

    @staticmethod
    def _previous_features(vector: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(vector.square().mean(dim=-1, keepdim=True).clamp_min(1.0e-10))
        shape = vector / rms
        log_rms = torch.log(rms.clamp_min(1.0e-7))
        return torch.cat((shape, log_rms), dim=-1)

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

    @staticmethod
    def _teacher_mask(step: int, ratio: float, seed: int) -> bool:
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        threshold = int(round(float(ratio) * 1000.0))
        value = (int(seed) + int(step) * 977 + 431) % 1000
        return value < threshold

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
        """Return [B, T+1, 512] residual analysis vectors.

        ``teacher_vectors`` is legal only during owned training.  Product generation omits it and is
        fully autoregressive from the model's own previous vector.
        """
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
        previous = torch.zeros(
            batch,
            RESIDUAL_VECTOR_SAMPLES,
            dtype=context.dtype,
            device=context.device,
        )
        outputs: list[torch.Tensor] = []
        for step in range(vector_count):
            if step > 0 and teacher_vectors is not None and self._teacher_mask(
                step, float(teacher_forcing_ratio), int(teacher_seed)
            ):
                previous = teacher_vectors[:, step - 1]
            previous_embed = F.gelu(self.previous_projection(self._previous_features(previous)))
            conditioning = context[:, min(step, frames - 1)]
            state = self.recurrent(torch.cat((conditioning, previous_embed), dim=-1), state)
            current = self.vector_projection(self.pre_output(state))
            outputs.append(current)
            previous = current

        result = torch.stack(outputs, dim=1).contiguous()
        if result.shape != (batch, vector_count, RESIDUAL_VECTOR_SAMPLES):
            raise RuntimeError("continuous residual source geometry changed")
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("continuous residual source produced non-finite values")
        return result

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
    "CONTINUOUS_SOURCE_ARCHITECTURE",
    "HOP_LENGTH",
    "MEL_BINS",
    "RESIDUAL_VECTOR_SAMPLES",
    "LykenoxContinuousResidualSourceV1",
]
