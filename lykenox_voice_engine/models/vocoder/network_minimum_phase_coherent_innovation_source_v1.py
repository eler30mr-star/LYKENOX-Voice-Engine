"""LYKENOX-owned coherent + stochastic-innovation residual source.

This is the root architectural correction after continuous residual source V2 recovered level,
pronunciation and removed the gangoso/chillido defect but remained perceptually robotic.  V2 was
fully deterministic and therefore had to regress both repeatable/coherent residual structure and
irreducibly aperiodic fine structure to one deterministic trajectory.  This source preserves V2's
explicit level factorization and coherent frame-rate recurrence, while representing aperiodic source
energy as a separate owned stochastic innovation whose amount and spectral color are predicted at
frame rate.

The innovation is generated from stateless local pseudo-noise and a learned frame-rate spectral
coloring envelope.  There is no external model, pretrained third-party weight, codebook, sample-rate
neural autoregression, transpose convolution, post-hoc gain, EQ, denoise, or duration modification.
Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v2 import (
    DEFAULT_CONTEXT_CHANNELS,
    DEFAULT_DILATIONS,
    DEFAULT_PREVIOUS_EMBED,
    DEFAULT_STATE_CHANNELS,
    HOP_LENGTH,
    INITIAL_LOG_RMS,
    MAX_LOG_RMS,
    MEL_BINS,
    MIN_LOG_RMS,
    RESIDUAL_VECTOR_SAMPLES,
    SHAPE_EPSILON,
    _ContextBlock,
    _conditioning_features,
)


COHERENT_INNOVATION_ARCHITECTURE = "lykenox_owned_coherent_innovation_residual_source_v1"
INNOVATION_BANDS = 16
MAX_INNOVATION_LOG_COLOR = 2.0
INITIAL_INNOVATION_MIX = 0.03


class LykenoxCoherentInnovationResidualSourceV1(nn.Module):
    """Frame-rate coherent residual generator plus learned aperiodic innovation distribution."""

    architecture = COHERENT_INNOVATION_ARCHITECTURE

    def __init__(
        self,
        *,
        context_channels: int = DEFAULT_CONTEXT_CHANNELS,
        state_channels: int = DEFAULT_STATE_CHANNELS,
        previous_embed: int = DEFAULT_PREVIOUS_EMBED,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
        innovation_bands: int = INNOVATION_BANDS,
    ) -> None:
        super().__init__()
        if context_channels < 64 or state_channels < 64 or previous_embed < 32:
            raise ValueError("coherent-innovation source dimensions are below the architecture floor")
        if not dilations or any(int(value) < 1 for value in dilations):
            raise ValueError("dilations must be positive")
        if innovation_bands < 4 or innovation_bands > 64:
            raise ValueError("innovation_bands must be in [4,64]")
        self.context_channels = int(context_channels)
        self.state_channels = int(state_channels)
        self.previous_embed = int(previous_embed)
        self.dilations = tuple(int(value) for value in dilations)
        self.innovation_bands = int(innovation_bands)

        self.conditioning_projection = nn.Conv1d(MEL_BINS + 3, context_channels, kernel_size=1)
        self.context_blocks = nn.ModuleList(
            [_ContextBlock(context_channels, dilation) for dilation in self.dilations]
        )
        self.previous_projection = nn.Linear(RESIDUAL_VECTOR_SAMPLES, previous_embed)
        self.recurrent = nn.GRUCell(context_channels + previous_embed, state_channels)
        self.pre_output = nn.Sequential(
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
            nn.Linear(state_channels, state_channels),
            nn.GELU(),
        )
        self.shape_projection = nn.Linear(state_channels, RESIDUAL_VECTOR_SAMPLES)
        self.level_projection = nn.Linear(state_channels + context_channels, 1)

        # These two heads are the new source degrees of freedom.  Mix starts near zero so a
        # warm-start from V2 initially reproduces its already-good coherent path and level.
        self.innovation_mix_projection = nn.Linear(state_channels + context_channels, 1)
        self.innovation_color_projection = nn.Linear(
            state_channels + context_channels, self.innovation_bands
        )
        nn.init.zeros_(self.innovation_mix_projection.weight)
        mix = min(1.0 - 1.0e-6, max(1.0e-6, INITIAL_INNOVATION_MIX))
        nn.init.constant_(self.innovation_mix_projection.bias, math.log(mix / (1.0 - mix)))
        nn.init.zeros_(self.innovation_color_projection.weight)
        nn.init.zeros_(self.innovation_color_projection.bias)

        # When a V2 warm-start is not available this matches V2's safe finite level initialization.
        nn.init.zeros_(self.level_projection.weight)
        probability = (INITIAL_LOG_RMS - MIN_LOG_RMS) / (MAX_LOG_RMS - MIN_LOG_RMS)
        probability = min(1.0 - 1.0e-6, max(1.0e-6, probability))
        nn.init.constant_(self.level_projection.bias, math.log(probability / (1.0 - probability)))

    @staticmethod
    def _unit_rms(vector: torch.Tensor) -> torch.Tensor:
        return vector * torch.rsqrt(vector.square().mean(dim=-1, keepdim=True) + SHAPE_EPSILON)

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

    def _bounded_log_rms(self, hidden: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        raw = self.level_projection(torch.cat((hidden, conditioning), dim=-1))
        return MIN_LOG_RMS + (MAX_LOG_RMS - MIN_LOG_RMS) * torch.sigmoid(raw)

    @staticmethod
    def _stateless_noise(
        batch: int,
        *,
        step: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        index = torch.arange(RESIDUAL_VECTOR_SAMPLES, device=device, dtype=dtype)
        phase = (
            index.unsqueeze(0)
            + float(seed) * 131.0
            + float(step) * 977.0
            + torch.arange(batch, device=device, dtype=dtype).unsqueeze(1) * 271.0
        ) * 12.9898 + 78.233
        hashed = torch.sin(phase) * 43758.5453123
        return (hashed - torch.floor(hashed)).mul(2.0).sub(1.0)

    def _colored_innovation(
        self,
        raw_noise: torch.Tensor,
        raw_color: torch.Tensor,
    ) -> torch.Tensor:
        color = MAX_INNOVATION_LOG_COLOR * torch.tanh(raw_color)
        bins = RESIDUAL_VECTOR_SAMPLES // 2 + 1
        interpolated = F.interpolate(
            color.unsqueeze(1), size=bins, mode="linear", align_corners=True
        ).squeeze(1)
        spectrum = torch.fft.rfft(raw_noise, n=RESIDUAL_VECTOR_SAMPLES, dim=-1)
        shaped = torch.fft.irfft(
            spectrum * torch.exp(interpolated).to(spectrum.dtype),
            n=RESIDUAL_VECTOR_SAMPLES,
            dim=-1,
        )
        return self._unit_rms(shaped)

    def forward_with_components(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
        *,
        innovation_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_conditioning(mel, f0_hz, voiced, periodicity)
        batch, frames, _ = context.shape
        vector_count = frames + 1
        state = torch.zeros(batch, self.state_channels, dtype=context.dtype, device=context.device)
        previous_coherent = torch.zeros(
            batch, RESIDUAL_VECTOR_SAMPLES, dtype=context.dtype, device=context.device
        )
        outputs: list[torch.Tensor] = []
        coherent_shapes: list[torch.Tensor] = []
        innovations: list[torch.Tensor] = []
        levels: list[torch.Tensor] = []
        mixes: list[torch.Tensor] = []

        for step in range(vector_count):
            previous_shape = self._unit_rms(previous_coherent)
            previous_embed = F.gelu(self.previous_projection(previous_shape))
            conditioning = context[:, min(step, frames - 1)]
            state = self.recurrent(torch.cat((conditioning, previous_embed), dim=-1), state)
            hidden = self.pre_output(state)
            coherent = self._unit_rms(self.shape_projection(hidden))
            head_input = torch.cat((hidden, conditioning), dim=-1)
            log_rms = self._bounded_log_rms(hidden, conditioning)
            mix = torch.sigmoid(self.innovation_mix_projection(head_input))
            raw_noise = self._stateless_noise(
                batch,
                step=step,
                seed=int(innovation_seed),
                device=context.device,
                dtype=context.dtype,
            )
            innovation = self._colored_innovation(
                raw_noise, self.innovation_color_projection(head_input)
            )
            coherent_weight = torch.sqrt((1.0 - mix.square()).clamp_min(0.0))
            combined_shape = self._unit_rms(coherent_weight * coherent + mix * innovation)
            current = combined_shape * torch.exp(log_rms)

            outputs.append(current)
            coherent_shapes.append(coherent)
            innovations.append(innovation)
            levels.append(log_rms.squeeze(-1))
            mixes.append(mix.squeeze(-1))
            # Only the coherent trajectory is recurrent.  Random innovation is intentionally not
            # allowed to destabilize future deterministic phonetic structure.
            previous_coherent = coherent

        vectors = torch.stack(outputs, dim=1).contiguous()
        coherent = torch.stack(coherent_shapes, dim=1).contiguous()
        innovation = torch.stack(innovations, dim=1).contiguous()
        log_rms = torch.stack(levels, dim=1).contiguous()
        innovation_mix = torch.stack(mixes, dim=1).contiguous()
        expected = (batch, vector_count, RESIDUAL_VECTOR_SAMPLES)
        if vectors.shape != expected or coherent.shape != expected or innovation.shape != expected:
            raise RuntimeError("coherent-innovation source vector geometry changed")
        if log_rms.shape != (batch, vector_count) or innovation_mix.shape != (batch, vector_count):
            raise RuntimeError("coherent-innovation source scalar geometry changed")
        values = (vectors, coherent, innovation, log_rms, innovation_mix)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise RuntimeError("coherent-innovation source produced non-finite values")
        return values

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
        *,
        innovation_seed: int = 0,
    ) -> torch.Tensor:
        vectors, _, _, _, _ = self.forward_with_components(
            mel, f0_hz, voiced, periodicity, innovation_seed=innovation_seed
        )
        return vectors

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
        *,
        innovation_seed: int = 0,
    ) -> torch.Tensor:
        return self(mel, f0_hz, voiced, periodicity, innovation_seed=innovation_seed)


__all__ = [
    "COHERENT_INNOVATION_ARCHITECTURE",
    "HOP_LENGTH",
    "INNOVATION_BANDS",
    "RESIDUAL_VECTOR_SAMPLES",
    "LykenoxCoherentInnovationResidualSourceV1",
]
