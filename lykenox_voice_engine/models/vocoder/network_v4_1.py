"""LYKENOX vocoder v4.1: source-balanced pitch-conditioned source-filter.

V4 established the correct product direction: explicit F0/voicing removes the learned
mel-hop carrier and gives the waveform generator a real vocal excitation source.  The
first listening gate also exposed the next bounded problem: the fixed ``1 / harmonic``
source puts too much authority in the fundamental/subgrave region and gives the neural
filter little control over the spectral envelope.

V4.1 keeps the source-filter architecture and changes only that local balance:

* harmonic weights are predicted from the LYKENOX mel conditioning instead of being
  permanently fixed at ``1 / harmonic``;
* the learned weights are bounded and RMS-normalized so the source cannot win merely by
  becoming louder;
* the output uses a fixed 30 Hz linear-phase high-pass FIR to remove DC/subgrave drift
  while minimally disturbing the observed 80-100 Hz speech fundamentals.

There is still no learned temporal upsampling, no external vocoder, and no reference
audio requirement in the intended runtime.  Training uses target F0/voicing only until
the LYKENOX acoustic model learns to predict that same contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V4_1_ARCHITECTURE = "lykenox_pitch_source_filter_v4_1"


class _SourceFilterBlockV41(nn.Module):
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


def _design_highpass_fir(
    *,
    sample_rate: int,
    cutoff_hz: float,
    kernel_size: int,
) -> torch.Tensor:
    """Create a deterministic windowed-sinc high-pass FIR kernel.

    The filter is deliberately below the pitch-analysis floor (60 Hz).  It is not a
    93.75 Hz notch: genuine low speech F0 remains legal and is supplied explicitly by the
    source oscillator.
    """

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("high-pass kernel_size must be odd and >= 3")
    if not 0.0 < cutoff_hz < sample_rate / 2.0:
        raise ValueError("invalid high-pass cutoff")

    center = (kernel_size - 1) / 2.0
    n = torch.arange(kernel_size, dtype=torch.float32) - center
    normalized_cutoff = float(cutoff_hz) / float(sample_rate)
    lowpass = 2.0 * normalized_cutoff * torch.sinc(2.0 * normalized_cutoff * n)
    window = torch.hann_window(kernel_size, periodic=False, dtype=torch.float32)
    lowpass = lowpass * window
    lowpass = lowpass / lowpass.sum().clamp_min(1e-8)
    highpass = -lowpass
    highpass[kernel_size // 2] += 1.0
    return highpass.view(1, 1, kernel_size).contiguous()


class LykenoxVocoderGeneratorV41(nn.Module):
    """Pitch-conditioned source-filter with mel-conditioned harmonic balance.

    Inputs:
    - ``mel``: ``[batch, mel_frames, mel_bins]``
    - ``f0_hz``: ``[batch, mel_frames]``; zero for unvoiced frames
    - ``voiced``: ``[batch, mel_frames]`` in ``[0, 1]``

    Output: ``[batch, mel_frames * hop_length]``.
    """

    architecture = VOCODER_GENERATOR_V4_1_ARCHITECTURE

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        hidden_channels: int = 32,
        harmonics: int = 8,
        harmonic_log_range: float = 1.25,
        highpass_cutoff_hz: float = 30.0,
        highpass_kernel_size: int = 513,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if hidden_channels < 16:
            raise ValueError("hidden_channels must be >= 16")
        if harmonics < 1 or harmonics > 16:
            raise ValueError("harmonics must be between 1 and 16")
        if harmonic_log_range <= 0.0:
            raise ValueError("harmonic_log_range must be positive")
        self.hidden_channels = int(hidden_channels)
        self.harmonics = int(harmonics)
        self.harmonic_log_range = float(harmonic_log_range)
        self.highpass_cutoff_hz = float(highpass_cutoff_hz)
        self.highpass_kernel_size = int(highpass_kernel_size)

        # The final projection is zero-initialized: at step zero v4.1 reproduces the
        # proven v4 1/h harmonic envelope exactly, then learns bounded timbral deviations.
        self.harmonic_envelope = nn.Sequential(
            nn.Conv1d(self.config.mel_bins, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, self.harmonics, kernel_size=1),
        )
        final_envelope = self.harmonic_envelope[-1]
        assert isinstance(final_envelope, nn.Conv1d)
        nn.init.zeros_(final_envelope.weight)
        nn.init.zeros_(final_envelope.bias)

        input_channels = self.config.mel_bins + self.harmonics + 3
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_SourceFilterBlockV41(hidden_channels, dilation) for dilation in (1, 3, 9, 27, 81)]
        )
        self.post = nn.Conv1d(hidden_channels, 1, kernel_size=7, padding=3)

        highpass = _design_highpass_fir(
            sample_rate=self.config.sample_rate,
            cutoff_hz=self.highpass_cutoff_hz,
            kernel_size=self.highpass_kernel_size,
        )
        self.register_buffer("output_highpass_fir", highpass, persistent=True)

        baseline = torch.tensor(
            [1.0 / float(index) for index in range(1, self.harmonics + 1)],
            dtype=torch.float32,
        )
        self.register_buffer("baseline_harmonic_weights", baseline, persistent=True)

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

    def _harmonic_weight_frames(self, mel: torch.Tensor) -> torch.Tensor:
        """Return bounded, source-RMS-neutral harmonic weights at mel rate."""

        logits = self.harmonic_envelope(mel.transpose(1, 2))
        multiplier = torch.exp(self.harmonic_log_range * torch.tanh(logits))
        baseline = self.baseline_harmonic_weights.view(1, self.harmonics, 1).to(
            device=mel.device,
            dtype=mel.dtype,
        )
        weights = baseline * multiplier
        baseline_rms = torch.sqrt(baseline.square().sum(dim=1, keepdim=True)).clamp_min(1e-8)
        learned_rms = torch.sqrt(weights.square().sum(dim=1, keepdim=True)).clamp_min(1e-8)
        return weights * (baseline_rms / learned_rms)

    def _harmonic_source(
        self,
        f0_samples: torch.Tensor,
        voiced_samples: torch.Tensor,
        harmonic_weights_samples: torch.Tensor,
    ) -> torch.Tensor:
        phase = torch.cumsum(
            2.0 * math.pi * f0_samples / float(self.config.sample_rate),
            dim=2,
        )
        sources: list[torch.Tensor] = []
        for harmonic_index in range(1, self.harmonics + 1):
            offset = (harmonic_index * 0.61803398875 % 1.0) * 2.0 * math.pi
            source = torch.sin(phase * harmonic_index + offset)
            source = source * voiced_samples * harmonic_weights_samples[:, harmonic_index - 1 : harmonic_index]
            sources.append(source)
        return torch.cat(sources, dim=1)

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
        harmonic_weight_frames = self._harmonic_weight_frames(mel)
        harmonic_weights_samples = F.interpolate(
            harmonic_weight_frames,
            size=samples,
            mode="linear",
            align_corners=False,
        )

        harmonic = self._harmonic_source(
            f0_samples,
            voiced_samples,
            harmonic_weights_samples,
        )
        noise = self._aperiodic_source(
            batch,
            samples,
            device=mel.device,
            dtype=mel.dtype,
        )
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
        raw_waveform = self.post(x)
        filtered = F.conv1d(
            raw_waveform,
            self.output_highpass_fir.to(device=raw_waveform.device, dtype=raw_waveform.dtype),
            padding=self.highpass_kernel_size // 2,
        )
        waveform = torch.tanh(filtered).squeeze(1)

        if int(waveform.shape[1]) != samples:
            raise RuntimeError(
                "LYKENOX v4.1 vocoder output length contract failed: "
                f"{waveform.shape[1]} != {samples}"
            )
        return waveform

    def harmonic_weight_snapshot(self, mel: torch.Tensor) -> list[float]:
        """Median harmonic weights for one diagnostic batch."""

        with torch.no_grad():
            weights = self._harmonic_weight_frames(mel)
            medians = weights.median(dim=2).values.mean(dim=0)
        return [float(value) for value in medians.detach().cpu()]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
