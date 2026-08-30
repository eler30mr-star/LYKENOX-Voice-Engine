"""LYKENOX vocoder V7: source-free mel-latent waveform decoder.

V7 is a clean successor to the perceptually rejected V6 architecture. F0 and voicing are
accepted only at mel-frame rate and are fused into a learned latent representation before
waveform upsampling. No sample-rate pitch/phase control, deterministic excitation, harmonic
bank, pulse/aperture signal, noise source, raw-source bypass, or RMS shape normalization is
present.

The waveform path uses learned stride upsampling followed by multi-receptive-field residual
refinement. This gives the network enough within-frame degrees of freedom to synthesize speech
without exposing a hand-built periodic carrier. Objective metrics may reject V7, but only a
full-utterance listening A/B against v4.2 may grant perceptual acceptance.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V7_ARCHITECTURE = "lykenox_source_free_mel_latent_waveform_v7"


class _V7ResidualUnit(nn.Module):
    """Two-convolution residual unit used inside a multi-receptive-field stage."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("v7 residual kernel_size must be odd and >= 3")
        if dilation < 1:
            raise ValueError("v7 residual dilation must be positive")
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
        y = self.conv1(F.leaky_relu(x, negative_slope=0.1))
        y = self.conv2(F.leaky_relu(y, negative_slope=0.1))
        return (residual + y) * (2.0 ** -0.5)


class _V7MRFBlock(nn.Module):
    """Parallel residual stacks with different kernel widths."""

    def __init__(
        self,
        channels: int,
        *,
        kernels: tuple[int, ...],
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not kernels or not dilations:
            raise ValueError("v7 MRF kernels/dilations cannot be empty")
        branches: list[nn.Module] = []
        for kernel in kernels:
            branches.append(
                nn.Sequential(
                    *[
                        _V7ResidualUnit(channels, kernel, dilation)
                        for dilation in dilations
                    ]
                )
            )
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        return torch.stack(outputs, dim=0).mean(dim=0)


class _V7UpsampleStage(nn.Module):
    """Learned exact-ratio upsampling followed by anti-grid residual refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        *,
        kernels: tuple[int, ...],
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        if factor < 2:
            raise ValueError("v7 upsample factor must be >= 2")
        kernel_size = factor * 2
        padding = factor // 2
        self.factor = int(factor)
        self.upsample = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=factor,
            padding=padding,
        )
        self.refine = _V7MRFBlock(
            out_channels,
            kernels=kernels,
            dilations=dilations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(F.leaky_relu(x, negative_slope=0.1))
        return self.refine(x)


class LykenoxVocoderGeneratorV7(nn.Module):
    """Frame-conditioned, genuinely source-free neural waveform decoder."""

    architecture = VOCODER_GENERATOR_V7_ARCHITECTURE
    source_family = "source_free_mel_latent_waveform_decoder"
    source_free = True
    explicit_source = False
    explicit_sinusoidal_carrier = False
    deterministic_harmonics = 0
    voiced_noise_source = False
    deterministic_noise_conditioning = False
    raw_source_bypass = False
    sample_phase_conditioning = False
    sample_rate_pitch_features = False
    pitch_conditioning_scope = "frame_latent_only"
    local_unit_rms_shape_normalization = False
    global_unit_rms_shape_normalization = False
    level_rescue_branch = False
    posthoc_gain_normalization = False
    perceptually_accepted = False

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        frame_channels: int = 160,
        upsample_channels: tuple[int, ...] = (128, 96, 64),
        upsample_factors: tuple[int, ...] = (8, 8, 4),
        residual_kernels: tuple[int, ...] = (3, 7, 11),
        residual_dilations: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if math.prod(upsample_factors) != self.config.hop_length:
            raise ValueError("v7 upsample factors must multiply exactly to hop_length")
        if len(upsample_channels) != len(upsample_factors):
            raise ValueError("v7 channel/factor schedule length mismatch")
        if frame_channels < 64 or min(upsample_channels) < 32:
            raise ValueError("v7 channel schedule is below supported capacity")

        self.frame_channels = int(frame_channels)
        self.upsample_channels = tuple(int(value) for value in upsample_channels)
        self.upsample_factors = tuple(int(value) for value in upsample_factors)
        self.residual_kernels = tuple(int(value) for value in residual_kernels)
        self.residual_dilations = tuple(int(value) for value in residual_dilations)
        self.frame_feature_channels = self.config.mel_bins + 2

        self.frame_pre = nn.Conv1d(
            self.frame_feature_channels,
            self.frame_channels,
            kernel_size=7,
            padding=3,
        )
        self.frame_context = nn.Sequential(
            _V7ResidualUnit(self.frame_channels, 5, 1),
            _V7ResidualUnit(self.frame_channels, 5, 3),
            _V7ResidualUnit(self.frame_channels, 5, 9),
        )

        stages: list[nn.Module] = []
        in_channels = self.frame_channels
        for factor, out_channels in zip(
            self.upsample_factors,
            self.upsample_channels,
            strict=True,
        ):
            stages.append(
                _V7UpsampleStage(
                    in_channels,
                    out_channels,
                    factor,
                    kernels=self.residual_kernels,
                    dilations=self.residual_dilations,
                )
            )
            in_channels = out_channels
        self.upsampling = nn.ModuleList(stages)

        self.post = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=7, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(48, 1, kernel_size=7, padding=3),
        )
        final = self.post[-1]
        assert isinstance(final, nn.Conv1d)
        nn.init.normal_(final.weight, mean=0.0, std=0.002)
        nn.init.zeros_(final.bias)

    def _validate_inputs(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> None:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, mel_frames, mel_bins]")
        if mel.shape[-1] != self.config.mel_bins:
            raise ValueError("v7 mel bin mismatch")
        if f0_hz.shape != mel.shape[:2] or voiced.shape != mel.shape[:2]:
            raise ValueError("v7 f0_hz/voiced must match mel [batch, mel_frames]")

    def _frame_latent(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        # Pitch information exists only here, at frame rate. It is never expanded into
        # sample-by-sample phase, pulses, harmonics, apertures, or noise controls.
        voiced_clean = voiced.clamp(0.0, 1.0)
        log_f0 = torch.log1p(f0_hz.clamp_min(0.0)) / math.log1p(500.0)
        features = torch.cat(
            [mel, log_f0.unsqueeze(-1), voiced_clean.unsqueeze(-1)],
            dim=-1,
        ).transpose(1, 2)
        return self.frame_context(self.frame_pre(features))

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(mel, f0_hz, voiced)
        batch, mel_frames, _ = mel.shape
        expected_samples = int(mel_frames) * self.config.hop_length

        x = self._frame_latent(mel, f0_hz, voiced)
        for stage in self.upsampling:
            x = stage(x)
        waveform = torch.tanh(self.post(F.leaky_relu(x, negative_slope=0.1))).squeeze(1)

        if tuple(waveform.shape) != (batch, expected_samples):
            raise RuntimeError(
                "LYKENOX v7 vocoder output length contract failed: "
                f"{tuple(waveform.shape)} != {(batch, expected_samples)}"
            )
        return waveform

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
