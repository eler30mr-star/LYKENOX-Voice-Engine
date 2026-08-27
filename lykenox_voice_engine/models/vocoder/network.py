"""Compact LYKENOX-owned non-autoregressive mel-to-waveform generator.

The purpose of this first implementation is to establish an owned, local CPU vocoder
path and measure its feasibility before committing to long training. It intentionally
uses only generic PyTorch layers and does not wrap or import a third-party vocoder.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


class _ResidualUnit(nn.Module):
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


class LykenoxVocoderGenerator(nn.Module):
    """Turn log-mel frames into one 24 kHz waveform sample stream.

    Input shape: ``[batch, mel_frames, mel_bins]``.
    Output shape: ``[batch, mel_frames * hop_length]``.

    The upsampling factors are part of the versioned configuration and must multiply
    exactly to ``hop_length``. This makes acoustic-model/vocoder length compatibility
    explicit instead of repairing mismatched lengths after synthesis.
    """

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
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=factor * 2,
                    stride=factor,
                    padding=factor // 2,
                )
            )
            stages.append(
                nn.Sequential(
                    *[
                        _ResidualUnit(
                            out_channels,
                            self.config.residual_kernel_size,
                            dilation,
                        )
                        for dilation in self.config.residual_dilations
                    ]
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
            if isinstance(stage, nn.ConvTranspose1d):
                x = F.leaky_relu(x, negative_slope=0.1)
            x = stage(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        waveform = torch.tanh(self.post(x)).squeeze(1)

        expected = int(mel.shape[1]) * self.config.hop_length
        if int(waveform.shape[1]) != expected:
            raise RuntimeError(
                "LYKENOX vocoder output length contract failed: "
                f"{waveform.shape[1]} != {expected}"
            )
        return waveform

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
