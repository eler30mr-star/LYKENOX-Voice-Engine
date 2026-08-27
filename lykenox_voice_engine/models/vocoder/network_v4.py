"""LYKENOX vocoder v4: explicit pitch-conditioned neural source-filter.

V0/v2/v3 showed that a tiny mel-only waveform generator can improve spectral loss while
inventing a carrier locked to the mel hop. V1 avoided the carrier but lacked sample-rate
excitation and collapsed spectrally. V4 stops asking the upsampler to invent pitch.

The waveform path receives an explicit F0/voicing excitation extracted from real audio
while training. The final speech acoustic model will predict the same F0/voicing contract;
for singing, score-conditioned pitch can feed this interface directly.

There is no learned temporal upsampling. Mel/F0 conditioning is resized deterministically
to sample rate, harmonic excitation supplies voiced periodic structure, a deterministic
aperiodic source supplies unvoiced detail, and inexpensive depthwise-separable residual
filters shape the final waveform. Generic PyTorch only; this is a LYKENOX-owned model.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V4_ARCHITECTURE = "lykenox_pitch_source_filter_v4"


class _SourceFilterBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * 3
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            dilation=dilation,
            padding=padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.depthwise(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.pointwise(x)
        return residual + x


class LykenoxVocoderGeneratorV4(nn.Module):
    """Pitch-conditioned sample-rate neural source-filter.

    Inputs:
    - ``mel``: ``[batch, mel_frames, mel_bins]``
    - ``f0_hz``: ``[batch, mel_frames]``; zero for unvoiced frames
    - ``voiced``: ``[batch, mel_frames]`` in ``[0, 1]``

    Output: ``[batch, mel_frames * hop_length]``.
    """

    architecture = VOCODER_GENERATOR_V4_ARCHITECTURE

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        hidden_channels: int = 32,
        harmonics: int = 8,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if hidden_channels < 16:
            raise ValueError("hidden_channels must be >= 16")
        if harmonics < 1 or harmonics > 16:
            raise ValueError("harmonics must be between 1 and 16")
        self.hidden_channels = int(hidden_channels)
        self.harmonics = int(harmonics)

        input_channels = self.config.mel_bins + self.harmonics + 3
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_SourceFilterBlock(hidden_channels, dilation) for dilation in (1, 3, 9, 27, 81)]
        )
        self.post = nn.Conv1d(hidden_channels, 1, kernel_size=7, padding=3)

    def _validate_inputs(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> None:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, mel_frames, mel_bins]")
        if mel.shape[-1] != self.config.mel_bins:
            raise ValueError(
                f"mel bin mismatch: {mel.shape[-1]} != {self.config.mel_bins}"
            )
        if f0_hz.shape != mel.shape[:2] or voiced.shape != mel.shape[:2]:
            raise ValueError("f0_hz and voiced must match mel [batch, mel_frames]")

    def _harmonic_source(
        self,
        f0_samples: torch.Tensor,
        voiced_samples: torch.Tensor,
    ) -> torch.Tensor:
        # Integrating instantaneous frequency means the oscillator follows the supplied
        # pitch rather than any mel-frame clock. A fixed per-harmonic phase offset avoids
        # every harmonic starting at the same zero crossing.
        phase = torch.cumsum(
            2.0 * math.pi * f0_samples / float(self.config.sample_rate),
            dim=2,
        )
        sources: list[torch.Tensor] = []
        for harmonic_index in range(1, self.harmonics + 1):
            offset = (harmonic_index * 0.61803398875 % 1.0) * 2.0 * math.pi
            source = torch.sin(phase * harmonic_index + offset)
            source = source * voiced_samples / float(harmonic_index)
            sources.append(source)
        stacked = torch.cat(sources, dim=1)
        normalization = math.sqrt(sum(1.0 / (index * index) for index in range(1, self.harmonics + 1)))
        return stacked / normalization

    @staticmethod
    def _aperiodic_source(
        batch: int,
        samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Deterministic high-rate pseudo-noise basis. It is independent of the mel hop,
        # reproducible for validation, and introduces no external RNG/runtime state.
        index = torch.arange(samples, device=device, dtype=dtype)
        raw = torch.sin(index * 12.9898 + 78.233) * 43758.5453
        noise = (raw - torch.floor(raw)) * 2.0 - 1.0
        return noise.view(1, 1, samples).expand(batch, 1, samples)

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(mel, f0_hz, voiced)
        batch, mel_frames, _ = mel.shape
        samples = int(mel_frames) * self.config.hop_length

        mel_samples = F.interpolate(
            mel.transpose(1, 2),
            size=samples,
            mode="linear",
            align_corners=False,
        )
        f0_samples = F.interpolate(
            f0_hz.unsqueeze(1),
            size=samples,
            mode="linear",
            align_corners=False,
        ).clamp_min(0.0)
        voiced_samples = F.interpolate(
            voiced.unsqueeze(1),
            size=samples,
            mode="linear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        harmonic = self._harmonic_source(f0_samples, voiced_samples)
        noise = self._aperiodic_source(
            batch,
            samples,
            device=mel.device,
            dtype=mel.dtype,
        )
        # Keep some aperiodic excitation in voiced regions for aspiration/frication while
        # emphasizing it in unvoiced regions.
        noise = noise * (0.08 + 0.92 * (1.0 - voiced_samples))
        log_f0 = torch.log1p(f0_samples) / math.log1p(500.0)

        x = torch.cat(
            [mel_samples, harmonic, voiced_samples, log_f0, noise],
            dim=1,
        )
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        waveform = torch.tanh(self.post(x)).squeeze(1)

        if int(waveform.shape[1]) != samples:
            raise RuntimeError(
                "LYKENOX v4 vocoder output length contract failed: "
                f"{waveform.shape[1]} != {samples}"
            )
        return waveform

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
