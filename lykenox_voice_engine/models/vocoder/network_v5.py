"""LYKENOX vocoder v5 candidate: stochastic glottal-pulse excitation, no sinusoidal carrier.

V4.1-v4.4 repeatedly exposed a deterministic periodic carrier as a radio-mistuned / metallic
artifact.  V5 is an architectural break: F0 controls *when* stochastic voiced excitation is
released, but no sinusoidal/harmonic bank is ever added to the waveform path.

The only audible source is deterministic broadband noise shaped into voiced glottal-like
bursts plus an unvoiced broadband component.  Mel and phase/F0 features only select and gate
bias-free filters; they cannot create waveform from zero excitation.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig
from .network_v4_1 import _design_highpass_fir


VOCODER_GENERATOR_V5_ARCHITECTURE = "lykenox_stochastic_glottal_filter_v5"


class _StochasticDynamicBlockV5(nn.Module):
    """Bias-free dynamic filter block driven only by an excitation state."""

    def __init__(
        self,
        channels: int,
        conditioning_channels: int,
        dilation: int,
        *,
        filter_bases: int = 2,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.filter_bases = int(filter_bases)
        self.filters = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=7,
                    dilation=dilation,
                    padding=3 * dilation,
                    groups=channels,
                    bias=False,
                )
                for _ in range(filter_bases)
            ]
        )
        self.selector = nn.Conv1d(
            conditioning_channels,
            channels * filter_bases,
            kernel_size=1,
            bias=True,
        )
        self.gate = nn.Conv1d(
            conditioning_channels,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.channel_mix = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.residual = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        activated = F.leaky_relu(x, negative_slope=0.1)
        candidates = torch.stack([layer(activated) for layer in self.filters], dim=2)
        batch, channels, _bases, samples = candidates.shape
        selector = self.selector(conditioning).view(
            batch,
            channels,
            self.filter_bases,
            samples,
        )
        selector = torch.softmax(selector, dim=2)
        dynamic = (candidates * selector).sum(dim=2)
        dynamic = self.channel_mix(F.leaky_relu(dynamic, negative_slope=0.1))

        # Conditioning can only scale excitation-dependent state; it cannot inject signal.
        gate = 0.5 + torch.sigmoid(self.gate(conditioning))
        y = torch.tanh(dynamic * gate)
        residual = (x + self.residual(y)) * (2.0 ** -0.5)
        return residual, self.skip(y)


class LykenoxVocoderGeneratorV5(nn.Module):
    """CPU-bounded non-sinusoidal corrective vocoder candidate."""

    architecture = VOCODER_GENERATOR_V5_ARCHITECTURE
    source_family = "stochastic_glottal_pulse_noise"
    explicit_sinusoidal_carrier = False
    deterministic_harmonics = 0

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        hidden_channels: int = 48,
        mel_conditioning_channels: int = 64,
        filter_bases: int = 2,
        pulse_width_cycles: float = 0.09,
        voiced_noise_floor: float = 0.10,
        highpass_cutoff_hz: float = 30.0,
        highpass_kernel_size: int = 513,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if hidden_channels < 32:
            raise ValueError("hidden_channels must be >= 32")
        if mel_conditioning_channels < hidden_channels:
            raise ValueError("mel_conditioning_channels must be >= hidden_channels")
        if filter_bases < 2 or filter_bases > 3:
            raise ValueError("filter_bases must be 2 or 3")
        if not 0.03 <= pulse_width_cycles <= 0.20:
            raise ValueError("pulse_width_cycles outside safe range")
        if not 0.0 <= voiced_noise_floor <= 0.5:
            raise ValueError("voiced_noise_floor outside safe range")

        self.hidden_channels = int(hidden_channels)
        self.mel_conditioning_channels = int(mel_conditioning_channels)
        self.filter_bases = int(filter_bases)
        self.pulse_width_cycles = float(pulse_width_cycles)
        self.voiced_noise_floor = float(voiced_noise_floor)
        self.highpass_cutoff_hz = float(highpass_cutoff_hz)
        self.highpass_kernel_size = int(highpass_kernel_size)
        self.dilations = (1, 2, 4, 8, 16, 32, 64, 128)
        self.phase_feature_channels = 4  # pulse aperture, centered phase, voiced, log-F0
        conditioning_channels = self.mel_conditioning_channels + self.phase_feature_channels

        self.frame_conditioner = nn.Sequential(
            nn.Conv1d(self.config.mel_bins, mel_conditioning_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mel_conditioning_channels, mel_conditioning_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Three excitation channels: stochastic pulse bursts, voiced broadband floor,
        # and unvoiced broadband noise.  No deterministic periodic waveform is present.
        self.excitation_stem = nn.Sequential(
            nn.Conv1d(3, hidden_channels, kernel_size=15, padding=7, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=15, padding=7, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.blocks = nn.ModuleList(
            [
                _StochasticDynamicBlockV5(
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
    def _deterministic_noise(
        batch: int,
        samples: int,
        *,
        phase_offset: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        index = torch.arange(samples, device=device, dtype=dtype)
        raw = torch.sin(index * 12.9898 + float(phase_offset)) * 43758.5453
        noise = (raw - torch.floor(raw)) * 2.0 - 1.0
        return noise.view(1, 1, samples).expand(batch, 1, samples)

    def _sample_controls(
        self,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = F.interpolate(
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

        cycles = torch.cumsum(f0 / float(self.config.sample_rate), dim=2)
        phase = torch.remainder(cycles, 1.0)
        circular_distance = torch.minimum(phase, 1.0 - phase)
        pulse = torch.exp(
            -0.5 * (circular_distance / self.pulse_width_cycles).square()
        ) * voiced_samples
        centered_phase = (phase - 0.5) * voiced_samples
        log_f0 = torch.log1p(f0) / math.log1p(500.0)
        return f0, voiced_samples, pulse, torch.cat(
            [pulse, centered_phase, voiced_samples, log_f0], dim=1
        )

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

        mel_frames_conditioning = self.frame_conditioner(mel.transpose(1, 2))
        mel_conditioning = F.interpolate(
            mel_frames_conditioning,
            size=samples,
            mode="linear",
            align_corners=False,
        )
        _f0, voiced_samples, pulse, phase_features = self._sample_controls(
            f0_hz,
            voiced,
            samples,
        )
        conditioning = torch.cat([mel_conditioning, phase_features], dim=1)

        noise_a = self._deterministic_noise(
            batch,
            samples,
            phase_offset=78.233,
            device=mel.device,
            dtype=mel.dtype,
        )
        noise_b = self._deterministic_noise(
            batch,
            samples,
            phase_offset=19.417,
            device=mel.device,
            dtype=mel.dtype,
        )
        noise_c = self._deterministic_noise(
            batch,
            samples,
            phase_offset=131.71,
            device=mel.device,
            dtype=mel.dtype,
        )

        stochastic_pulses = noise_a * pulse
        voiced_floor = noise_b * voiced_samples * self.voiced_noise_floor
        unvoiced_noise = noise_c * (1.0 - voiced_samples)
        excitation = torch.cat(
            [stochastic_pulses, voiced_floor, unvoiced_noise],
            dim=1,
        ) * float(excitation_scale)

        x = self.excitation_stem(excitation)
        skips: list[torch.Tensor] = []
        for block in self.blocks:
            x, skip = block(x, conditioning)
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
            raise RuntimeError("LYKENOX v5 output length contract failed")
        return waveform

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(mel, f0_hz, voiced, excitation_scale=1.0)

    def diagnostic_zero_excitation(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(mel, f0_hz, voiced, excitation_scale=0.0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def sample_receptive_field(self) -> int:
        # Two 15-sample stem convolutions + eight kernel-7 dilated blocks.
        return 29 + 6 * sum(self.dilations)
