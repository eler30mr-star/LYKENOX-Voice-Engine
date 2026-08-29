"""LYKENOX vocoder v4.4 candidate: 8-harmonic hybrid excitation + mel-selected filter bank.

V4.3 removed the additive source shortcut, but full-utterance listening regressed into a
radio-mistuned / metallic carrier texture.  A post-training carrier ablation found that the
8-harmonic equal-RMS variant was the best of the tested 24/16/12/8-harmonic and voiced-noise
variants, although it did not solve the artifact.

V4.4 keeps the causal improvement (eight harmonics) but changes the overly restrictive
v4.3 mel control.  Mel no longer applies only a scalar multiplicative gain.  Instead it
selects among multiple bias-free convolutional filter bases at every residual block, while
an independent broadband aperiodic excitation branch supplies non-periodic detail.  There
is still no mel-only waveform path and no raw source-to-waveform bypass.

Every audible path is excitation-dependent.  Setting the complete excitation scale to zero
therefore yields exactly zero waveform even with non-zero mel conditioning.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig
from .network_v4_1 import _design_highpass_fir


VOCODER_GENERATOR_V4_4_ARCHITECTURE = "lykenox_dynamic_filter_hybrid_v4_4"


class _DynamicFilterBankBlockV44(nn.Module):
    """Mel-selected bias-free filter bank with an excitation-dependent noise residual."""

    def __init__(
        self,
        channels: int,
        conditioning_channels: int,
        dilation: int,
        *,
        filter_bases: int = 3,
    ) -> None:
        super().__init__()
        if filter_bases < 2:
            raise ValueError("filter_bases must be >= 2")
        self.channels = int(channels)
        self.filter_bases = int(filter_bases)
        self.filters = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=7,
                    dilation=dilation,
                    padding=dilation * 3,
                    groups=channels,
                    bias=False,
                )
                for _ in range(filter_bases)
            ]
        )
        self.filter_selector = nn.Conv1d(
            conditioning_channels,
            channels * filter_bases,
            kernel_size=1,
            bias=True,
        )
        self.channel_mix = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.aperiodic_projection = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.aperiodic_gate = nn.Conv1d(
            conditioning_channels,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.residual_projection = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.skip_projection = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # Start as an equal filter-bank mixture with modest aperiodic injection.  Mel learns
        # spectral/time-varying selection rather than creating an additive waveform feature.
        nn.init.zeros_(self.filter_selector.weight)
        nn.init.zeros_(self.filter_selector.bias)
        nn.init.zeros_(self.aperiodic_gate.weight)
        nn.init.constant_(self.aperiodic_gate.bias, -1.0)

    def forward(
        self,
        x: torch.Tensor,
        aperiodic_state: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        activated = F.leaky_relu(x, negative_slope=0.1)
        candidates = torch.stack([layer(activated) for layer in self.filters], dim=2)
        batch, channels, _bases, samples = candidates.shape
        selector = self.filter_selector(conditioning).view(
            batch,
            channels,
            self.filter_bases,
            samples,
        )
        selector = torch.softmax(selector, dim=2)
        dynamic = (candidates * selector).sum(dim=2)
        dynamic = self.channel_mix(F.leaky_relu(dynamic, negative_slope=0.1))

        noise_gate = torch.sigmoid(self.aperiodic_gate(conditioning))
        noise_detail = self.aperiodic_projection(aperiodic_state) * noise_gate
        y = torch.tanh(dynamic + noise_detail)
        residual = (x + self.residual_projection(y)) * (2.0 ** -0.5)
        skip = self.skip_projection(y)
        return residual, skip


class LykenoxVocoderGeneratorV44(nn.Module):
    """CPU-bounded hybrid source/filter candidate for persistent LYKENOX speech."""

    architecture = VOCODER_GENERATOR_V4_4_ARCHITECTURE

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        hidden_channels: int = 64,
        conditioning_channels: int = 96,
        harmonics: int = 8,
        filter_bases: int = 3,
        highpass_cutoff_hz: float = 30.0,
        highpass_kernel_size: int = 513,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if hidden_channels < 32:
            raise ValueError("hidden_channels must be >= 32")
        if conditioning_channels < hidden_channels:
            raise ValueError("conditioning_channels must be >= hidden_channels")
        if harmonics != 8:
            raise ValueError("v4.4 causal candidate fixes harmonics=8 from the v4.3 ablation")
        if filter_bases < 2 or filter_bases > 4:
            raise ValueError("filter_bases must be between 2 and 4")

        self.hidden_channels = int(hidden_channels)
        self.conditioning_channels = int(conditioning_channels)
        self.harmonics = int(harmonics)
        self.filter_bases = int(filter_bases)
        self.highpass_cutoff_hz = float(highpass_cutoff_hz)
        self.highpass_kernel_size = int(highpass_kernel_size)
        self.dilations = (1, 2, 4, 8, 16, 32, 64, 128)

        self.frame_conditioner = nn.Sequential(
            nn.Conv1d(self.config.mel_bins, conditioning_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(conditioning_channels, conditioning_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Periodic and aperiodic excitation are encoded independently.  All convolutions
        # touching audible state are bias-free, so mel cannot synthesize waveform by itself.
        periodic_channels = self.harmonics + 2  # harmonics + voiced + log-F0
        self.periodic_stem = nn.Sequential(
            nn.Conv1d(periodic_channels, hidden_channels, kernel_size=15, padding=7, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=15, padding=7, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.aperiodic_stem = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=15, padding=7, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=15, padding=7, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.initial_mix = nn.Conv1d(
            hidden_channels * 2,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )

        self.blocks = nn.ModuleList(
            [
                _DynamicFilterBankBlockV44(
                    hidden_channels,
                    conditioning_channels,
                    dilation,
                    filter_bases=filter_bases,
                )
                for dilation in self.dilations
            ]
        )
        self.post = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(hidden_channels, 1, kernel_size=7, padding=3, bias=False),
        )

        highpass = _design_highpass_fir(
            sample_rate=self.config.sample_rate,
            cutoff_hz=self.highpass_cutoff_hz,
            kernel_size=self.highpass_kernel_size,
        )
        self.register_buffer("output_highpass_fir", highpass, persistent=True)

        harmonic_index = torch.arange(1, self.harmonics + 1, dtype=torch.float32)
        weights = torch.rsqrt(harmonic_index)
        weights = weights / torch.sqrt(weights.square().sum()).clamp_min(1e-8)
        self.register_buffer("harmonic_weights", weights, persistent=True)

    def _validate_inputs(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> None:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, mel_frames, mel_bins]")
        if mel.shape[-1] != self.config.mel_bins:
            raise ValueError("mel bin mismatch")
        if f0_hz.shape != mel.shape[:2] or voiced.shape != mel.shape[:2]:
            raise ValueError("f0_hz and voiced must match mel [batch, mel_frames]")

    @staticmethod
    def _aperiodic_source(
        batch: int,
        samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        index = torch.arange(samples, device=device, dtype=dtype)
        raw = torch.sin(index * 12.9898 + 78.233) * 43758.5453
        noise = (raw - torch.floor(raw)) * 2.0 - 1.0
        return noise.view(1, 1, samples).expand(batch, 1, samples)

    def _harmonic_carrier(
        self,
        f0_samples: torch.Tensor,
        voiced_samples: torch.Tensor,
    ) -> torch.Tensor:
        phase = torch.cumsum(
            2.0 * math.pi * f0_samples / float(self.config.sample_rate),
            dim=2,
        )
        guard_hz = 0.46 * float(self.config.sample_rate)
        transition_hz = 350.0
        weights = self.harmonic_weights.to(device=f0_samples.device, dtype=f0_samples.dtype)
        channels: list[torch.Tensor] = []
        for harmonic_index in range(1, self.harmonics + 1):
            offset = (harmonic_index * 0.61803398875 % 1.0) * 2.0 * math.pi
            frequency = f0_samples * float(harmonic_index)
            anti_alias = torch.sigmoid((guard_hz - frequency) / transition_hz)
            channels.append(
                torch.sin(phase * harmonic_index + offset)
                * voiced_samples
                * anti_alias
                * weights[harmonic_index - 1]
            )
        return torch.cat(channels, dim=1)

    def _forward_impl(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        *,
        excitation_scale: float,
    ) -> torch.Tensor:
        self._validate_inputs(mel, f0_hz, voiced)
        batch, mel_frames, _ = mel.shape
        samples = int(mel_frames) * self.config.hop_length

        conditioning_frames = self.frame_conditioner(mel.transpose(1, 2))
        conditioning = F.interpolate(
            conditioning_frames,
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

        harmonic = self._harmonic_carrier(f0_samples, voiced_samples)
        log_f0 = torch.log1p(f0_samples) / math.log1p(500.0)
        periodic_input = torch.cat([harmonic, voiced_samples, log_f0], dim=1)

        noise = self._aperiodic_source(
            batch,
            samples,
            device=mel.device,
            dtype=mel.dtype,
        )
        # Preserve useful aperiodicity during voiced speech instead of v4.3's near-pure
        # deterministic carrier.  The learned filter decides how much reaches the waveform.
        voiced_noise_floor = 0.12
        shaped_noise = noise * (
            voiced_noise_floor + (1.0 - voiced_noise_floor) * (1.0 - voiced_samples)
        )
        aperiodic_input = torch.cat([shaped_noise, 1.0 - voiced_samples], dim=1)

        scale = float(excitation_scale)
        periodic_state = self.periodic_stem(periodic_input * scale)
        aperiodic_state = self.aperiodic_stem(aperiodic_input * scale)
        x = self.initial_mix(torch.cat([periodic_state, aperiodic_state], dim=1))

        skips: list[torch.Tensor] = []
        for block in self.blocks:
            x, skip = block(x, aperiodic_state, conditioning)
            skips.append(skip)
        x = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(float(len(skips)))
        raw = self.post(x)
        filtered = F.conv1d(
            raw,
            self.output_highpass_fir.to(device=raw.device, dtype=raw.dtype),
            padding=self.highpass_kernel_size // 2,
        )
        waveform = torch.tanh(filtered).squeeze(1)
        if tuple(waveform.shape) != (batch, samples):
            raise RuntimeError("LYKENOX v4.4 output length contract failed")
        return waveform

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(mel, f0_hz, voiced, excitation_scale=1.0)

    def diagnostic_zero_excitation(self, mel: torch.Tensor) -> torch.Tensor:
        """Prove that non-zero mel cannot create waveform without excitation."""
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, mel_frames, mel_bins]")
        f0 = torch.zeros(mel.shape[:2], device=mel.device, dtype=mel.dtype)
        voiced = torch.zeros_like(f0)
        return self._forward_impl(mel, f0, voiced, excitation_scale=0.0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def sample_receptive_field(self) -> int:
        return 29 + 6 * sum(self.dilations)
