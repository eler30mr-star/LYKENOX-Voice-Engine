"""Third LYKENOX-owned vocoder generator: learned polyphase upsampling.

The resize-convolution v1 probe removed the transposed-convolution frame-rate buzz, but
its first held-out listening gate collapsed almost all generated energy into sub-bass.
Linear interpolation creates a very smooth sample grid and gives the network no explicit
learned phase channels at each upsampling step.

This v2 generator uses 1-D subpixel/polyphase upsampling. A stride-1 convolution predicts
``factor`` learned phase channels for every output channel, then a deterministic channel-
to-time shuffle expands the sequence. There is no overlapping transposed convolution and
no interpolation bottleneck. Generic PyTorch layers only; this remains a LYKENOX model.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V2_ARCHITECTURE = "lykenox_compact_polyphase_v2"


class _ResidualUnitV2(nn.Module):
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


def _phase_equal_initialization(conv: nn.Conv1d, out_channels: int, factor: int) -> None:
    """Initialize every phase of one output channel identically.

    Polyphase/subpixel layers can begin with a periodic phase pattern if each phase filter
    is initialized independently. Repeating one Kaiming-initialized base filter across the
    ``factor`` phases makes the initial stage phase-neutral while keeping every phase free
    to learn independently afterward.
    """

    with torch.no_grad():
        base = torch.empty(
            out_channels,
            conv.in_channels,
            conv.kernel_size[0],
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        nn.init.kaiming_normal_(base, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
        conv.weight.copy_(base.repeat_interleave(factor, dim=0))


class _PolyphaseStage(nn.Module):
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
        self.out_channels = int(out_channels)
        self.factor = int(factor)
        # No bias: a constant learned per-phase bias is an easy route to a periodic
        # carrier unrelated to the mel conditioning.
        self.expand = nn.Conv1d(
            in_channels,
            out_channels * factor,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        _phase_equal_initialization(self.expand, out_channels, factor)
        self.refine = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = nn.Sequential(
            *[
                _ResidualUnitV2(
                    out_channels,
                    residual_kernel_size,
                    dilation,
                )
                for dilation in residual_dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.expand(x)
        batch, channels, steps = x.shape
        expected_channels = self.out_channels * self.factor
        if channels != expected_channels:
            raise RuntimeError(
                f"Polyphase channel contract failed: {channels} != {expected_channels}"
            )
        # [B, out*factor, T] -> [B, out, T, factor] -> [B, out, T*factor]
        x = x.reshape(batch, self.out_channels, self.factor, steps)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(batch, self.out_channels, steps * self.factor)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.refine(x)
        return self.residual(x)


class LykenoxVocoderGeneratorV2(nn.Module):
    """Compact learned-polyphase mel-to-waveform generator.

    Input: ``[batch, mel_frames, mel_bins]``.
    Output: ``[batch, mel_frames * hop_length]``.

    Every stage expands time by an exact integer factor using learned phase channels. The
    product of all factors remains the versioned speech hop length, so the acoustic model
    and vocoder stay sample-count compatible without post-hoc trimming or padding.
    """

    architecture = VOCODER_GENERATOR_V2_ARCHITECTURE

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
                _PolyphaseStage(
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
                "LYKENOX polyphase vocoder output length contract failed: "
                f"{waveform.shape[1]} != {expected}"
            )
        return waveform

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
