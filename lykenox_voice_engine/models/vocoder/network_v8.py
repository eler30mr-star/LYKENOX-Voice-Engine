"""LYKENOX V8: frame-rate complex-spectral overlap-add vocoder candidate.

V4.2 forensics proved that the intelligible baseline depends on explicit harmonic and
aperiodic excitation yet still reconstructs the 300--4000 Hz envelope poorly. V7 proved
that learned waveform upsampling can lock catastrophically to the 256-sample frame grid.

V8 removes both failure mechanisms:

* no explicit sinusoidal, phase-accumulator, periodic-aperture, or noise source;
* no ConvTranspose1d and no learned sample-rate interpolation/upsampling;
* mel, log-F0, and voicing stay at frame rate and only condition a temporal network;
* the network predicts one-sided complex STFT coefficients directly;
* waveform synthesis is fixed Hann-window iSTFT overlap-add;
* output length is exactly ``mel_frames * hop_length``.

F0/voicing are conditioning features, not waveform carriers. This module is an
architecture candidate only; it does not load V4.2/V6/V7 weights and does not authorize
persistent training.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V8_ARCHITECTURE = "lykenox_complex_spectral_overlap_add_v8"


class _FrameResidualBlockV8(nn.Module):
    """Dilated frame-rate residual block; never operates at sample rate."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            dilation=dilation,
            padding=2 * dilation,
            groups=channels,
        )
        self.gate_projection = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.output_projection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(F.gelu(x))
        activation, gate = self.gate_projection(y).chunk(2, dim=1)
        y = torch.tanh(activation) * torch.sigmoid(gate)
        y = self.output_projection(y)
        return (residual + y) * (2.0 ** -0.5)


class LykenoxVocoderGeneratorV8(nn.Module):
    """Predict complex STFT frames and synthesize through fixed iSTFT overlap-add."""

    architecture = VOCODER_GENERATOR_V8_ARCHITECTURE
    source_free = True
    explicit_sample_rate_source = False
    learned_sample_rate_upsampling = False
    synthesis = "fixed_hann_istft_overlap_add"
    persistent_training_authorized = False

    def __init__(
        self,
        config: LykenoxVocoderConfig | None = None,
        *,
        hidden_channels: int = 128,
        n_fft: int = 1024,
        win_length: int = 1024,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxVocoderConfig()
        if hidden_channels < 64:
            raise ValueError("hidden_channels must be >= 64")
        if n_fft < self.config.hop_length * 2 or n_fft % 2:
            raise ValueError("n_fft must be even and at least 2 * hop_length")
        if win_length != n_fft:
            raise ValueError("V8 currently requires win_length == n_fft")
        self.hidden_channels = int(hidden_channels)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.frequency_bins = self.n_fft // 2 + 1
        self.dilations = (1, 2, 4, 8, 16)

        conditioning_channels = self.config.mel_bins + 2  # log-F0 + voicing
        self.input_conditioner = nn.Sequential(
            nn.Conv1d(conditioning_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_FrameResidualBlockV8(hidden_channels, dilation) for dilation in self.dilations]
        )
        self.spectrum_head = nn.Conv1d(
            hidden_channels,
            self.frequency_bins * 2,
            kernel_size=1,
        )
        nn.init.normal_(self.spectrum_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.spectrum_head.bias)

        window = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        self.register_buffer("synthesis_window", window, persistent=True)

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
        if mel.shape[1] < 2:
            raise ValueError("V8 requires at least two mel frames")

    def _frame_features(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        log_f0 = torch.log1p(f0_hz.clamp_min(0.0)) / math.log1p(500.0)
        conditioning = torch.cat(
            [mel, log_f0.unsqueeze(-1), voiced.clamp(0.0, 1.0).unsqueeze(-1)],
            dim=-1,
        )
        x = self.input_conditioner(conditioning.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return x

    def predict_complex_spectrum(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        """Return one-sided complex spectrum [B, F, T+1] for exact-length centered iSTFT."""
        self._validate_inputs(mel, f0_hz, voiced)
        hidden = self._frame_features(mel, f0_hz, voiced)
        # A centered STFT of a signal with T * hop samples has T+1 analysis frames.
        # Replicating only the final frame keeps all learned operations at mel rate while
        # matching that deterministic analysis/synthesis geometry.
        hidden = F.pad(hidden, (0, 1), mode="replicate")
        packed = self.spectrum_head(hidden)
        real, imag = packed.chunk(2, dim=1)
        spectrum = torch.complex(real, imag)
        expected = (mel.shape[0], self.frequency_bins, mel.shape[1] + 1)
        if tuple(spectrum.shape) != expected:
            raise RuntimeError(
                f"V8 spectrum shape contract failed: {tuple(spectrum.shape)} != {expected}"
            )
        return spectrum

    def synthesize_complex_spectrum(
        self,
        spectrum: torch.Tensor,
        *,
        samples: int,
    ) -> torch.Tensor:
        if spectrum.ndim != 3 or spectrum.shape[1] != self.frequency_bins:
            raise ValueError("spectrum must have shape [batch, frequency_bins, frames]")
        if not torch.is_complex(spectrum):
            raise ValueError("spectrum must be complex")
        if samples < self.config.hop_length:
            raise ValueError("samples is too short")
        waveform = torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.win_length,
            window=self.synthesis_window.to(
                device=spectrum.device,
                dtype=spectrum.real.dtype,
            ),
            center=True,
            normalized=False,
            onesided=True,
            length=int(samples),
            return_complex=False,
        )
        return waveform

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        spectrum = self.predict_complex_spectrum(mel, f0_hz, voiced)
        samples = int(mel.shape[1]) * self.config.hop_length
        waveform = self.synthesize_complex_spectrum(spectrum, samples=samples)
        expected = (mel.shape[0], samples)
        if tuple(waveform.shape) != expected:
            raise RuntimeError(
                f"V8 waveform length contract failed: {tuple(waveform.shape)} != {expected}"
            )
        return waveform

    def target_complex_spectrum(self, waveform: torch.Tensor) -> torch.Tensor:
        """Analysis transform matching V8 synthesis; useful for direct complex supervision."""
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.win_length,
            window=self.synthesis_window.to(
                device=waveform.device,
                dtype=waveform.dtype,
            ),
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def frame_receptive_field(self) -> int:
        return 1 + 4 * sum(self.dilations)
