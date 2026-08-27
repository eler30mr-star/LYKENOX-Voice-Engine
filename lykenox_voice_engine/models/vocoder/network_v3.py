"""LYKENOX vocoder v3: smooth base upsampling plus controlled phase residuals.

The first three generator experiments exposed two opposite failure modes:

- transposed-convolution v0 learned a strong carrier locked to the 24 kHz / 256 mel
  frame rate (93.75 Hz);
- resize-convolution v1 removed that carrier but collapsed most generated energy into
  sub-bass;
- free polyphase v2 restored spectral capacity but again learned a generated-specific
  93.75 Hz carrier.

V3 keeps the useful parts of v1 and v2 while structurally constraining the phase path.
Each stage has two branches:

1. a smooth linearly resized base branch that carries slowly varying conditioning; and
2. a learned polyphase *residual* whose phase channels are forced to have zero mean for
   every output channel/time position and are multiplied by a learned bounded gate.

The zero-mean residual cannot create a per-frame DC phase pattern by itself, while the
phase branch still has explicit sample-phase capacity for high-frequency waveform detail.
There is no ConvTranspose1d and no external vocoder implementation.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V3_ARCHITECTURE = "lykenox_smooth_phase_residual_v3"


class _ResidualUnitV3(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        return x + residual


class _ControlledPhaseStage(nn.Module):
    """Smooth integer upsampling with a zero-mean learned phase-detail branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        *,
        residual_kernel_size: int,
        residual_dilations: tuple[int, ...],
        initial_phase_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_phase_scale < 1.0:
            raise ValueError("initial_phase_scale must be between 0 and 1")
        self.out_channels = int(out_channels)
        self.factor = int(factor)

        self.base = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2,
        )
        # Bias is deliberately disabled: an independent bias for every phase is a direct
        # route to a frame-synchronous carrier. The detail filters start at zero so the
        # initial network is the smooth base path only.
        self.phase_detail = nn.Conv1d(
            in_channels,
            out_channels * factor,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        nn.init.zeros_(self.phase_detail.weight)

        initial_logit = math.log(initial_phase_scale / (1.0 - initial_phase_scale))
        self.phase_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

        self.refine = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.residual = nn.Sequential(
            *[
                _ResidualUnitV3(
                    out_channels,
                    residual_kernel_size,
                    dilation,
                )
                for dilation in residual_dilations
            ]
        )

    def phase_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.phase_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x, negative_slope=0.1)

        base = self.base(x)
        base = F.interpolate(
            base,
            scale_factor=self.factor,
            mode="linear",
            align_corners=False,
        )

        detail = self.phase_detail(x)
        batch, channels, steps = detail.shape
        expected_channels = self.out_channels * self.factor
        if channels != expected_channels:
            raise RuntimeError(
                f"Controlled phase channel contract failed: {channels} != {expected_channels}"
            )
        detail = detail.reshape(batch, self.out_channels, self.factor, steps)
        # Crucial periodicity control: phase residuals may redistribute energy within one
        # low-rate frame but cannot change the average value of that frame.
        detail = detail - detail.mean(dim=2, keepdim=True)
        detail = detail.permute(0, 1, 3, 2).contiguous()
        detail = detail.reshape(batch, self.out_channels, steps * self.factor)

        if detail.shape[-1] != base.shape[-1]:
            raise RuntimeError(
                "Controlled phase/base length mismatch: "
                f"{detail.shape[-1]} != {base.shape[-1]}"
            )

        combined = base + self.phase_scale() * detail
        combined = F.leaky_relu(combined, negative_slope=0.1)
        combined = self.refine(combined)
        return self.residual(combined)


class LykenoxVocoderGeneratorV3(nn.Module):
    """Compact mel-to-waveform generator with explicit periodicity control.

    Input shape: ``[batch, mel_frames, mel_bins]``.
    Output shape: ``[batch, mel_frames * hop_length]``.

    The upsample-factor product remains exactly the speech hop length, so V3 stays fully
    sample-count compatible with the current LYKENOX acoustic model.
    """

    architecture = VOCODER_GENERATOR_V3_ARCHITECTURE

    def __init__(self, config: LykenoxVocoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        self.pre = nn.Conv1d(
            self.config.mel_bins,
            self.config.channels,
            kernel_size=7,
            padding=3,
        )

        stages: list[_ControlledPhaseStage] = []
        in_channels = self.config.channels
        for factor in self.config.upsample_factors:
            # Keep the proven CPU-friendly schedule used by v2.
            out_channels = max(16, in_channels // 4)
            stages.append(
                _ControlledPhaseStage(
                    in_channels,
                    out_channels,
                    factor,
                    residual_kernel_size=self.config.residual_kernel_size,
                    residual_dilations=self.config.residual_dilations,
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.post = nn.Conv1d(in_channels, 1, kernel_size=7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, mel_frames, mel_bins]")
        if mel.shape[-1] != self.config.mel_bins:
            raise ValueError(
                f"mel bin mismatch: {mel.shape[-1]} != {self.config.mel_bins}"
            )

        x = self.pre(mel.transpose(1, 2))
        for stage in self.stages:
            x = stage(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        waveform = torch.tanh(self.post(x)).squeeze(1)

        expected = int(mel.shape[1]) * self.config.hop_length
        if int(waveform.shape[1]) != expected:
            raise RuntimeError(
                "LYKENOX v3 vocoder output length contract failed: "
                f"{waveform.shape[1]} != {expected}"
            )
        return waveform

    def phase_scales(self) -> list[float]:
        return [float(stage.phase_scale().detach().cpu()) for stage in self.stages]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
