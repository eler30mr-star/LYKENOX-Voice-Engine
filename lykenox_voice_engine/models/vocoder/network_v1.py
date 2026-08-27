"""Second LYKENOX-owned vocoder generator: resize-convolution upsampling.

The v0 transposed-convolution prototype proved CPU feasibility but its first held-out
listening gate exposed a strong periodic artifact locked to the mel frame rate
(sample_rate / hop_length).  This v1 generator removes strided transposed convolutions
from the waveform path.  Each stage performs deterministic linear interpolation followed
by ordinary Conv1d refinement and residual units.

This is still a compact experimental LYKENOX architecture, not a third-party vocoder.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V1_ARCHITECTURE = "lykenox_compact_resize_conv_v1"


class _ResidualUnitV1(nn.Module):
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


class _ResizeConvStage(nn.Module):
    """Upsample without transposed-convolution overlap periodicity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        *,
        residual_kernel_size: int,
        residual_dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.factor = int(factor)
        self.refine = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2,
        )
        self.residual = nn.Sequential(
            *[
                _ResidualUnitV1(
                    out_channels,
                    residual_kernel_size,
                    dilation,
                )
                for dilation in residual_dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=self.factor,
            mode="linear",
            align_corners=False,
        )
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.refine(x)
        return self.residual(x)


class LykenoxVocoderGeneratorV1(nn.Module):
    """Compact resize-convolution mel-to-waveform generator.

    Input: ``[batch, mel_frames, mel_bins]``.
    Output: ``[batch, mel_frames * hop_length]``.

    The same LYKENOX mel/audio contract is preserved; only the learned upsampling
    mechanism changes.  Linear resize establishes the exact temporal grid first, then
    stride-1 convolutions learn waveform detail without transposed-convolution overlap.
    """

    architecture = VOCODER_GENERATOR_V1_ARCHITECTURE

    def __init__(self, config: LykenoxVocoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        self.pre = nn.Conv1d(
            self.config.mel_bins,
            self.config.channels,
            kernel_size=7,
            padding=3,
        )

        stages: list[nn.Module] = []
        in_channels = self.config.channels
        for factor in self.config.upsample_factors:
            out_channels = max(16, in_channels // 2)
            stages.append(
                _ResizeConvStage(
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
                "LYKENOX resize-conv vocoder output length contract failed: "
                f"{waveform.shape[1]} != {expected}"
            )
        return waveform

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
