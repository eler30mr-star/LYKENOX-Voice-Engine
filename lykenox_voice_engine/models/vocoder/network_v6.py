"""LYKENOX vocoder v6: direct conditional waveform decoder.

V5 proved that removing a sinusoidal bank is not enough when the entire audible waveform is
still forced through an explicit stochastic voiced source. V6 removes the source/filter
factorization itself. Mel, F0 and voicing are conditioning features for a learned waveform
decoder; there is no voiced noise source, harmonic bank, pulse train, or raw excitation
bypass.

Revision 2 also separates waveform *shape* from slow amplitude control. The sample decoder
cannot win reconstruction losses by collapsing its raw scale because its output is locally
RMS-normalized before a mel-conditioned frame-RMS envelope is applied. This is an internal
learned level path, not inference-time normalization of the final waveform.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V6_ARCHITECTURE = "lykenox_direct_conditional_waveform_v6_level_decoupled"


class _GatedDepthwiseResidual(nn.Module):
    """CPU-friendly residual block with wide temporal context."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=7,
            dilation=dilation,
            padding=3 * dilation,
            groups=channels,
        )
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.depthwise(F.leaky_relu(x, negative_slope=0.1)).chunk(2, dim=1)
        y = torch.tanh(a) * torch.sigmoid(b)
        y = self.mix(y)
        return (x + y) * (2.0 ** -0.5)


class _ResizeRefineStage(nn.Module):
    """Deterministic resize + learned refinement without phase-indexed kernels."""

    def __init__(self, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()
        self.factor = int(factor)
        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.smooth = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            padding=3,
            groups=out_channels,
            bias=False,
        )
        self.mix = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.residual = nn.Sequential(
            _GatedDepthwiseResidual(out_channels, 1),
            _GatedDepthwiseResidual(out_channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(F.leaky_relu(x, negative_slope=0.1))
        x = F.interpolate(
            x,
            scale_factor=self.factor,
            mode="linear",
            align_corners=False,
        )
        x = self.smooth(x)
        x = self.mix(F.leaky_relu(x, negative_slope=0.1))
        return self.residual(x)


class LykenoxVocoderGeneratorV6(nn.Module):
    """Direct mel/F0/voicing-to-waveform decoder with decoupled learned level control."""

    architecture = VOCODER_GENERATOR_V6_ARCHITECTURE
    source_family = "direct_conditional_waveform_decoder"
    explicit_source = False
    explicit_sinusoidal_carrier = False
    deterministic_harmonics = 0
    voiced_noise_source = False
    raw_source_bypass = False
    conditioning_only_waveform = True
    waveform_shape_level_decoupled = True
    level_control_family = "mel_conditioned_frame_rms_envelope"

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        frame_channels: int = 96,
        upsample_channels: tuple[int, ...] = (72, 48, 32, 24),
        sample_channels: int = 48,
        upsample_factors: tuple[int, ...] = (4, 4, 4, 4),
        sample_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        initial_frame_rms: float = 0.012,
        min_frame_rms: float = 1e-4,
        max_frame_rms: float = 0.35,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if math.prod(upsample_factors) != self.config.hop_length:
            raise ValueError("v6 upsample factors must multiply exactly to hop_length")
        if len(upsample_channels) != len(upsample_factors):
            raise ValueError("v6 channel/factor schedule length mismatch")
        if frame_channels < 64 or sample_channels < 32:
            raise ValueError("v6 channel schedule is below supported capacity")
        if not (0.0 < min_frame_rms < initial_frame_rms < max_frame_rms < 1.0):
            raise ValueError("v6 frame-RMS bounds/initialization are invalid")

        self.frame_channels = int(frame_channels)
        self.upsample_channels = tuple(int(value) for value in upsample_channels)
        self.upsample_factors = tuple(int(value) for value in upsample_factors)
        self.sample_channels = int(sample_channels)
        self.sample_dilations = tuple(int(value) for value in sample_dilations)
        self.frame_feature_channels = self.config.mel_bins + 2  # mel + log-F0 + voiced
        self.sample_control_channels = 5  # phase aperture, phase, voiced, log-F0, UV detail
        self.min_frame_rms = float(min_frame_rms)
        self.max_frame_rms = float(max_frame_rms)

        self.frame_pre = nn.Conv1d(
            self.frame_feature_channels,
            frame_channels,
            kernel_size=5,
            padding=2,
        )
        self.frame_context = nn.Sequential(
            _GatedDepthwiseResidual(frame_channels, 1),
            _GatedDepthwiseResidual(frame_channels, 3),
            _GatedDepthwiseResidual(frame_channels, 9),
        )

        stages: list[_ResizeRefineStage] = []
        in_channels = frame_channels
        for factor, out_channels in zip(
            self.upsample_factors,
            self.upsample_channels,
            strict=True,
        ):
            stages.append(_ResizeRefineStage(in_channels, out_channels, factor))
            in_channels = out_channels
        self.upsampling = nn.ModuleList(stages)

        self.sample_pre = nn.Conv1d(
            in_channels + self.sample_control_channels,
            sample_channels,
            kernel_size=7,
            padding=3,
        )
        self.sample_decoder = nn.Sequential(
            *[
                _GatedDepthwiseResidual(sample_channels, dilation)
                for dilation in self.sample_dilations
            ]
        )
        self.post = nn.Sequential(
            nn.Conv1d(sample_channels, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

        # This path owns slow waveform level. The decoder's raw amplitude is normalized
        # before this envelope is applied, so decoder scale and level scale cannot cancel.
        self.frame_level_head = nn.Conv1d(
            frame_channels,
            1,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        nn.init.zeros_(self.frame_level_head.weight)
        initial_ratio = (
            (float(initial_frame_rms) - self.min_frame_rms)
            / (self.max_frame_rms - self.min_frame_rms)
        )
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.level_logit_bias_parameter = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )

    @staticmethod
    def _deterministic_noise(
        batch: int,
        samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        index = torch.arange(samples, device=device, dtype=dtype)
        raw = torch.sin(index * 12.9898 + 41.371) * 43758.5453
        noise = (raw - torch.floor(raw)) * 2.0 - 1.0
        return noise.view(1, 1, samples).expand(batch, 1, samples)

    def _rms_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        span = self.max_frame_rms - self.min_frame_rms
        return self.min_frame_rms + span * torch.sigmoid(logits)

    def nominal_output_rms(self) -> torch.Tensor:
        """Return the learned frame-RMS baseline before mel-conditioned residuals."""
        return self._rms_from_logits(self.level_logit_bias_parameter)

    def output_gain(self) -> torch.Tensor:
        """Backward-compatible diagnostic alias for the nominal learned output RMS."""
        return self.nominal_output_rms()

    def level_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters dedicated to slow level control for optimizer grouping."""
        return (self.level_logit_bias_parameter, *tuple(self.frame_level_head.parameters()))

    def _validate_inputs(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> None:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, mel_frames, mel_bins]")
        if mel.shape[-1] != self.config.mel_bins:
            raise ValueError("v6 mel bin mismatch")
        if f0_hz.shape != mel.shape[:2] or voiced.shape != mel.shape[:2]:
            raise ValueError("v6 f0_hz/voiced must match mel [batch, mel_frames]")

    def _frame_features(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        voiced_clean = voiced.clamp(0.0, 1.0)
        f0_clean = f0_hz.clamp_min(0.0)
        log_f0 = torch.log1p(f0_clean) / math.log1p(500.0)
        features = torch.cat(
            [mel, log_f0.unsqueeze(-1), voiced_clean.unsqueeze(-1)],
            dim=-1,
        )
        return features.transpose(1, 2)

    def _sample_controls(
        self,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        samples: int,
    ) -> torch.Tensor:
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
        centered_phase = (phase - 0.5) * voiced_samples
        circular_distance = torch.minimum(phase, 1.0 - phase)
        # This is a conditioning feature, not an excitation waveform or bypass.
        phase_aperture = torch.exp(-0.5 * (circular_distance / 0.12).square()) * voiced_samples
        log_f0 = torch.log1p(f0) / math.log1p(500.0)
        unvoiced_detail = self._deterministic_noise(
            int(f0.shape[0]),
            samples,
            device=f0.device,
            dtype=f0.dtype,
        ) * (1.0 - voiced_samples)
        return torch.cat(
            [phase_aperture, centered_phase, voiced_samples, log_f0, unvoiced_detail],
            dim=1,
        )

    def _level_envelope(
        self,
        frame_state: torch.Tensor,
        samples: int,
    ) -> torch.Tensor:
        frame_logits = (
            self.frame_level_head(frame_state)
            + self.level_logit_bias_parameter.view(1, 1, 1)
        )
        frame_rms = self._rms_from_logits(frame_logits)
        return F.interpolate(
            frame_rms,
            size=samples,
            mode="linear",
            align_corners=False,
        )

    def _normalized_waveform_shape(self, raw: torch.Tensor) -> torch.Tensor:
        """Remove slow mean/scale so only the dedicated level path owns amplitude."""
        radius = int(self.config.hop_length)
        kernel_size = 2 * radius + 1
        padded = F.pad(raw, (radius, radius), mode="replicate")
        local_mean = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
        centered = raw - local_mean

        squared = F.pad(centered.square(), (radius, radius), mode="replicate")
        local_power = F.avg_pool1d(squared, kernel_size=kernel_size, stride=1)
        shape = centered / torch.sqrt(local_power.clamp_min(1e-6))
        global_rms = torch.sqrt(shape.square().mean(dim=-1, keepdim=True).clamp_min(1e-6))
        return shape / global_rms

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(mel, f0_hz, voiced)
        batch, frames, _ = mel.shape
        expected_samples = int(frames) * self.config.hop_length

        frame_state = self.frame_context(
            self.frame_pre(self._frame_features(mel, f0_hz, voiced))
        )
        x = frame_state
        for stage in self.upsampling:
            x = stage(x)
        if int(x.shape[-1]) != expected_samples:
            raise RuntimeError("v6 progressive upsampling length contract failed")

        controls = self._sample_controls(f0_hz, voiced, expected_samples)
        x = torch.cat([x, controls], dim=1)
        x = self.sample_pre(x)
        x = self.sample_decoder(x)
        raw = self.post(F.leaky_relu(x, negative_slope=0.1))

        shape = self._normalized_waveform_shape(raw)
        level = self._level_envelope(frame_state, expected_samples)
        waveform = torch.tanh(shape * level).squeeze(1)
        if tuple(waveform.shape) != (batch, expected_samples):
            raise RuntimeError("LYKENOX v6 output length contract failed")
        return waveform

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def sample_decoder_receptive_field(self) -> int:
        # Lower bound from the sample-rate decoder only. Earlier resize/refine context adds more.
        return 13 + 6 * sum(self.sample_dilations)
