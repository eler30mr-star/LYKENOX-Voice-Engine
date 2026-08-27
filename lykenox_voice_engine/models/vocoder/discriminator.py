"""LYKENOX-owned lightweight waveform discriminator for vocoder training.

This module is training-only. The installed speech runtime needs only the generator.
It uses generic PyTorch convolutions and does not wrap a third-party GAN/vocoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


DISCRIMINATOR_ARCHITECTURE = "lykenox_multiscale_waveform_discriminator_v0"


@dataclass(frozen=True)
class DiscriminatorOutput:
    scores: list[torch.Tensor]
    feature_maps: list[list[torch.Tensor]]


class _WaveDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
                nn.Conv1d(16, 32, kernel_size=41, stride=4, padding=20, groups=4),
                nn.Conv1d(32, 64, kernel_size=41, stride=4, padding=20, groups=4),
                nn.Conv1d(64, 128, kernel_size=41, stride=4, padding=20, groups=4),
                nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            ]
        )
        self.post = nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [batch, 1, samples]")
        x = waveform
        features: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            features.append(x)
        score = self.post(x)
        features.append(score)
        return score, features


class LykenoxMultiScaleWaveformDiscriminator(nn.Module):
    """Two-scale CPU-oriented discriminator used only while training the vocoder."""

    architecture = DISCRIMINATOR_ARCHITECTURE

    def __init__(self, scales: int = 2) -> None:
        super().__init__()
        if scales < 1 or scales > 3:
            raise ValueError("scales must be between 1 and 3")
        self.scales = int(scales)
        self.discriminators = nn.ModuleList([_WaveDiscriminator() for _ in range(scales)])

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        if waveform.ndim == 2:
            x = waveform.unsqueeze(1)
        elif waveform.ndim == 3 and waveform.shape[1] == 1:
            x = waveform
        else:
            raise ValueError("waveform must have shape [batch, samples] or [batch, 1, samples]")

        scores: list[torch.Tensor] = []
        feature_maps: list[list[torch.Tensor]] = []
        for scale_index, discriminator in enumerate(self.discriminators):
            if scale_index:
                x = F.avg_pool1d(x, kernel_size=4, stride=2, padding=1)
            score, features = discriminator(x)
            scores.append(score)
            feature_maps.append(features)
        return DiscriminatorOutput(scores=scores, feature_maps=feature_maps)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
