"""LYKENOX vocoder v4.2 candidate: envelope-first gated source-filter.

V4.1 proved that explicit F0/voicing is the correct timing/pitch contract, but the
full-utterance oracle audit also proved that its 32-channel filter leaves a persistent
periodic metallic/buzz character even when mel, F0, voicing and durations are all target
features.  Source-shape ablations did not isolate one removable source component: the
baseline remains the clearest variant and the model collapses spectrally when harmonic
information is removed.

V4.2 therefore keeps the successful product constraints while changing the neural filter
itself:

* no learned temporal upsampling and no transposed convolution;
* exact ``mel_frames * hop_length`` waveform length;
* explicit F0/voicing and deterministic aperiodic excitation;
* a frame-rate mel encoder owns the spectral-envelope representation before interpolation;
* the periodic source enters through a separate temporal stem and a learned mel-controlled
  gate instead of being concatenated directly with raw mel as waveform authority;
* eight gated residual/skip blocks provide a wider nonlinear filter and about 43 ms of
  sample-rate receptive field, so the network can reshape source phase/timbre instead of
  exposing the oscillator comb directly;
* the same fixed 30 Hz linear-phase high-pass output contract is retained.

This module is a candidate architecture only.  It intentionally does not load or mutate
v4.1 weights.  Persistent training is gated behind a bounded real-data architecture smoke.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig
from .network_v4_1 import _design_highpass_fir


VOCODER_GENERATOR_V4_2_ARCHITECTURE = "lykenox_envelope_first_source_filter_v4_2"


class _EnvelopeFirstBlockV42(nn.Module):
    """Gated separable residual block with explicit mel-envelope conditioning."""

    def __init__(self, channels: int, conditioning_channels: int, dilation: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.conditioning_projection = nn.Conv1d(
            conditioning_channels,
            channels * 2,
            kernel_size=1,
        )
        self.depthwise = nn.Conv1d(
            channels * 2,
            channels * 2,
            kernel_size=5,
            dilation=dilation,
            padding=dilation * 2,
            groups=channels * 2,
        )
        self.output_projection = nn.Conv1d(channels, channels * 2, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        y = F.leaky_relu(x, negative_slope=0.1)
        y = self.input_projection(y) + self.conditioning_projection(conditioning)
        y = self.depthwise(y)
        activation, gate = y.chunk(2, dim=1)
        y = torch.tanh(activation) * torch.sigmoid(gate)
        residual_delta, skip = self.output_projection(y).chunk(2, dim=1)
        x = (residual + residual_delta) * (2.0 ** -0.5)
        return x, skip


class LykenoxVocoderGeneratorV42(nn.Module):
    """Envelope-first pitch-conditioned source-filter for CPU-feasible speech synthesis."""

    architecture = VOCODER_GENERATOR_V4_2_ARCHITECTURE

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        hidden_channels: int = 64,
        conditioning_channels: int = 96,
        harmonics: int = 8,
        harmonic_log_range: float = 1.25,
        highpass_cutoff_hz: float = 30.0,
        highpass_kernel_size: int = 513,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if hidden_channels < 32:
            raise ValueError("hidden_channels must be >= 32")
        if conditioning_channels < hidden_channels:
            raise ValueError("conditioning_channels must be >= hidden_channels")
        if harmonics < 1 or harmonics > 16:
            raise ValueError("harmonics must be between 1 and 16")
        if harmonic_log_range <= 0.0:
            raise ValueError("harmonic_log_range must be positive")

        self.hidden_channels = int(hidden_channels)
        self.conditioning_channels = int(conditioning_channels)
        self.harmonics = int(harmonics)
        self.harmonic_log_range = float(harmonic_log_range)
        self.highpass_cutoff_hz = float(highpass_cutoff_hz)
        self.highpass_kernel_size = int(highpass_kernel_size)
        self.dilations = (1, 2, 4, 8, 16, 32, 64, 128)

        # Spectral envelope is encoded at mel rate first.  This avoids reducing the raw
        # 80-bin mel directly to a 32-channel sample-rate bottleneck as v4.1 did.
        self.frame_conditioner = nn.Sequential(
            nn.Conv1d(
                self.config.mel_bins,
                conditioning_channels,
                kernel_size=3,
                padding=1,
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
        self.mel_to_hidden = nn.Conv1d(conditioning_channels, hidden_channels, kernel_size=1)

        # Keep v4.1's bounded mel-conditioned harmonic envelope as a pitch source, but the
        # source is now transformed in its own temporal stem before entering the filter.
        self.harmonic_envelope = nn.Sequential(
            nn.Conv1d(self.config.mel_bins, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, self.harmonics, kernel_size=1),
        )
        final_envelope = self.harmonic_envelope[-1]
        assert isinstance(final_envelope, nn.Conv1d)
        nn.init.zeros_(final_envelope.weight)
        nn.init.zeros_(final_envelope.bias)

        source_channels = self.harmonics + 3  # harmonics + voiced + log-F0 + aperiodic
        self.source_stem = nn.Sequential(
            nn.Conv1d(source_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=15, padding=7),
            nn.GELU(),
        )
        self.source_gate = nn.Conv1d(
            conditioning_channels,
            hidden_channels,
            kernel_size=1,
        )
        # Start from a moderate source contribution.  It is learnable per channel and per
        # sample from mel conditioning; unlike a runtime gain tweak this is part of training.
        nn.init.zeros_(self.source_gate.weight)
        nn.init.constant_(self.source_gate.bias, -0.5)

        self.blocks = nn.ModuleList(
            [
                _EnvelopeFirstBlockV42(
                    hidden_channels,
                    conditioning_channels,
                    dilation,
                )
                for dilation in self.dilations
            ]
        )
        self.post = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=7, padding=3),
        )

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
            source = (
                source
                * voiced_samples
                * harmonic_weights_samples[:, harmonic_index - 1 : harmonic_index]
            )
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

        mel_frames_ch = mel.transpose(1, 2)
        conditioning_frames = self.frame_conditioner(mel_frames_ch)
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

        source = torch.cat(
            [harmonic, voiced_samples, log_f0, noise],
            dim=1,
        )
        source_features = self.source_stem(source)
        source_gate = torch.sigmoid(self.source_gate(conditioning_samples))
        x = self.mel_to_hidden(conditioning_samples) + source_gate * source_features

        skips: list[torch.Tensor] = []
        for block in self.blocks:
            x, skip = block(x, conditioning_samples)
            skips.append(skip)
        x = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(float(len(skips)))
        raw_waveform = self.post(F.leaky_relu(x, negative_slope=0.1))
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
                "LYKENOX v4.2 vocoder output length contract failed: "
                f"{tuple(waveform.shape)} != {(batch, samples)}"
            )
        return waveform

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def sample_receptive_field(self) -> int:
        # Source stem kernel 15 plus eight kernel-5 dilated blocks.
        return 15 + 4 * sum(self.dilations)
