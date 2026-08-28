"""LYKENOX vocoder v4.3 candidate: mel-filtered carrier without additive source shortcut.

The v4.2 full-utterance oracle run materially improved v4.1, but a residual metallic/
insect-like periodic component remained.  A trained source-path ablation then showed that
reducing the complete source branch mainly removes voice, level and upper-band detail while
normalized periodicity remains similar.  With the source branch removed, useful speech
collapses into a weak subgrave residual.  Therefore v4.2 still uses the transformed source
as too much of the waveform itself.

V4.3 makes the architectural ownership explicit:

* deterministic carrier owns F0/phase and aperiodic excitation only;
* mel owns timbre/envelope through multiplicative filter controls only;
* there is no additive mel-to-waveform path and no additive source bypass;
* every carrier/filter/post convolution on the waveform path is bias-free;
* zero carrier therefore produces exactly zero waveform, even with non-zero mel;
* a richer anti-aliased harmonic carrier is forced through the mel-conditioned nonlinear
  filter before waveform projection;
* exact ``mel_frames * hop_length`` length, no transposed convolution and no learned
  temporal upsampling are preserved.

This module is an untrained bounded candidate.  Persistent training is not authorized until
its real-data architecture smoke passes.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig
from .network_v4_1 import _design_highpass_fir


VOCODER_GENERATOR_V4_3_ARCHITECTURE = "lykenox_mel_filtered_carrier_v4_3"


class _CarrierFilterBlockV43(nn.Module):
    """Bias-free carrier transform with multiplicative mel conditioning only."""

    def __init__(self, channels: int, conditioning_channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            dilation=dilation,
            padding=dilation * 3,
            groups=channels,
            bias=False,
        )
        self.channel_mix = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.condition_gain = nn.Conv1d(
            conditioning_channels,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.residual_projection = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.skip_projection = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.condition_gain.weight)
        nn.init.zeros_(self.condition_gain.bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = F.leaky_relu(x, negative_slope=0.1)
        y = self.depthwise(y)
        y = self.channel_mix(F.leaky_relu(y, negative_slope=0.1))
        # Initial gain is exactly 1.0.  Mel may attenuate/amplify the transformed carrier,
        # but it can never create a waveform feature from an all-zero carrier.
        gain = torch.exp(0.75 * torch.tanh(self.condition_gain(conditioning)))
        y = y * gain
        y = torch.tanh(y)
        residual = (x + self.residual_projection(y)) * (2.0 ** -0.5)
        skip = self.skip_projection(y)
        return residual, skip


class LykenoxVocoderGeneratorV43(nn.Module):
    """Mel-filtered deterministic carrier for CPU-feasible persistent-identity speech."""

    architecture = VOCODER_GENERATOR_V4_3_ARCHITECTURE

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        hidden_channels: int = 64,
        conditioning_channels: int = 96,
        harmonics: int = 24,
        highpass_cutoff_hz: float = 30.0,
        highpass_kernel_size: int = 513,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if hidden_channels < 32:
            raise ValueError("hidden_channels must be >= 32")
        if conditioning_channels < hidden_channels:
            raise ValueError("conditioning_channels must be >= hidden_channels")
        if harmonics < 4 or harmonics > 32:
            raise ValueError("harmonics must be between 4 and 32")

        self.hidden_channels = int(hidden_channels)
        self.conditioning_channels = int(conditioning_channels)
        self.harmonics = int(harmonics)
        self.highpass_cutoff_hz = float(highpass_cutoff_hz)
        self.highpass_kernel_size = int(highpass_kernel_size)
        self.dilations = (1, 2, 4, 8, 16, 32, 64, 128)

        # Mel is encoded at frame rate and reaches the waveform path only as a
        # multiplicative filter control.  It has no additive projection into x.
        self.frame_conditioner = nn.Sequential(
            nn.Conv1d(
                self.config.mel_bins,
                conditioning_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.GELU(),
            nn.Conv1d(
                conditioning_channels,
                conditioning_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )

        # Carrier channels are harmonic excitation + aperiodic excitation + voiced/logF0.
        # All waveform-path convolutions are bias-free, preserving the zero-carrier proof.
        carrier_channels = self.harmonics + 3
        self.carrier_stem = nn.Sequential(
            nn.Conv1d(
                carrier_channels,
                hidden_channels,
                kernel_size=15,
                padding=7,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=15,
                padding=7,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.blocks = nn.ModuleList(
            [
                _CarrierFilterBlockV43(
                    hidden_channels,
                    conditioning_channels,
                    dilation,
                )
                for dilation in self.dilations
            ]
        )
        self.post = nn.Sequential(
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(
                hidden_channels,
                1,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
        )

        highpass = _design_highpass_fir(
            sample_rate=self.config.sample_rate,
            cutoff_hz=self.highpass_cutoff_hz,
            kernel_size=self.highpass_kernel_size,
        )
        self.register_buffer("output_highpass_fir", highpass, persistent=True)

        # Richer than v4.1/v4.2's 8-harmonic source.  1/sqrt(h) retains useful upper
        # excitation while total harmonic RMS is normalized; mel-controlled filtering must
        # learn the vocal envelope instead of receiving a pre-shaped waveform shortcut.
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
            raise ValueError(
                f"mel bin mismatch: {mel.shape[-1]} != {self.config.mel_bins}"
            )
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

    def _carrier(
        self,
        f0_samples: torch.Tensor,
        voiced_samples: torch.Tensor,
    ) -> torch.Tensor:
        phase = torch.cumsum(
            2.0 * math.pi * f0_samples / float(self.config.sample_rate),
            dim=2,
        )
        harmonics: list[torch.Tensor] = []
        guard_hz = 0.46 * float(self.config.sample_rate)
        transition_hz = 350.0
        weights = self.harmonic_weights.to(
            device=f0_samples.device,
            dtype=f0_samples.dtype,
        )
        for harmonic_index in range(1, self.harmonics + 1):
            offset = (harmonic_index * 0.61803398875 % 1.0) * 2.0 * math.pi
            frequency = f0_samples * float(harmonic_index)
            anti_alias = torch.sigmoid((guard_hz - frequency) / transition_hz)
            source = torch.sin(phase * harmonic_index + offset)
            source = (
                source
                * voiced_samples
                * anti_alias
                * weights[harmonic_index - 1]
            )
            harmonics.append(source)
        return torch.cat(harmonics, dim=1)

    def _forward_impl(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        *,
        carrier_scale: float,
    ) -> torch.Tensor:
        self._validate_inputs(mel, f0_hz, voiced)
        batch, mel_frames, _ = mel.shape
        samples = int(mel_frames) * self.config.hop_length

        conditioning_frames = self.frame_conditioner(mel.transpose(1, 2))
        conditioning_samples = F.interpolate(
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

        harmonic = self._carrier(f0_samples, voiced_samples)
        noise = self._aperiodic_source(
            batch,
            samples,
            device=mel.device,
            dtype=mel.dtype,
        )
        noise = noise * (0.05 + 0.95 * (1.0 - voiced_samples))
        log_f0 = torch.log1p(f0_samples) / math.log1p(500.0)
        carrier = torch.cat(
            [harmonic, noise, voiced_samples, log_f0],
            dim=1,
        ) * float(carrier_scale)

        x = self.carrier_stem(carrier)
        skips: list[torch.Tensor] = []
        for block in self.blocks:
            x, skip = block(x, conditioning_samples)
            skips.append(skip)
        x = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(float(len(skips)))
        raw_waveform = self.post(x)
        filtered = F.conv1d(
            raw_waveform,
            self.output_highpass_fir.to(
                device=raw_waveform.device,
                dtype=raw_waveform.dtype,
            ),
            padding=self.highpass_kernel_size // 2,
        )
        waveform = torch.tanh(filtered).squeeze(1)
        if tuple(waveform.shape) != (batch, samples):
            raise RuntimeError(
                "LYKENOX v4.3 vocoder output length contract failed: "
                f"{tuple(waveform.shape)} != {(batch, samples)}"
            )
        return waveform

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(mel, f0_hz, voiced, carrier_scale=1.0)

    def diagnostic_zero_carrier(self, mel: torch.Tensor) -> torch.Tensor:
        """Prove that non-zero mel cannot create waveform without the carrier path."""

        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, mel_frames, mel_bins]")
        f0 = torch.zeros(mel.shape[:2], device=mel.device, dtype=mel.dtype)
        voiced = torch.zeros_like(f0)
        return self._forward_impl(mel, f0, voiced, carrier_scale=0.0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def sample_receptive_field(self) -> int:
        # Two kernel-15 stem layers + eight kernel-7 dilated filter blocks.
        return 29 + 6 * sum(self.dilations)
