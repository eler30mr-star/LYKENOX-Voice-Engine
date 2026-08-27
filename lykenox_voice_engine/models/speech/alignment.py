"""Compact LYKENOX-owned acoustic-to-text CTC aligner.

This module exists only to derive training-time frame durations. The final
speech runtime does not need the aligner once the acoustic model's duration
predictor has learned from cached alignments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LykenoxCTCAlignerConfig:
    """CPU-oriented configuration for the first LYKENOX speech aligner."""

    num_symbols: int
    mel_bins: int = 80
    hidden_size: int = 128
    recurrent_layers: int = 2
    dropout: float = 0.1
    frame_stride: int = 2

    @property
    def blank_id(self) -> int:
        return self.num_symbols


class LykenoxCTCAligner(nn.Module):
    """Mel -> Spanish token posterior model trained with CTC."""

    def __init__(self, config: LykenoxCTCAlignerConfig) -> None:
        super().__init__()
        if config.num_symbols < 2:
            raise ValueError("num_symbols must be >= 2")
        if config.hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even for bidirectional GRU")
        if config.hidden_size % 8 != 0:
            raise ValueError("hidden_size must be divisible by 8 for GroupNorm")
        if config.frame_stride != 2:
            raise ValueError("The current aligner implementation supports frame_stride=2")

        self.config = config
        self.acoustic_frontend = nn.Sequential(
            nn.Conv1d(
                config.mel_bins,
                config.hidden_size,
                kernel_size=5,
                stride=config.frame_stride,
                padding=2,
            ),
            nn.GroupNorm(8, config.hidden_size),
            nn.GELU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.encoder = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=config.recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(config.hidden_size, config.num_symbols + 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Return unnormalized CTC logits [batch, time, classes]."""

        if mel.ndim != 3 or mel.shape[-1] != self.config.mel_bins:
            raise ValueError(
                f"mel must have shape [batch, time, {self.config.mel_bins}]"
            )
        x = self.acoustic_frontend(mel.transpose(1, 2)).transpose(1, 2)
        x, _ = self.encoder(x)
        return self.projection(x)

    def output_lengths(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        """Map original mel lengths to CTC timestep lengths."""

        return torch.div(
            mel_lengths + self.config.frame_stride - 1,
            self.config.frame_stride,
            rounding_mode="floor",
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
