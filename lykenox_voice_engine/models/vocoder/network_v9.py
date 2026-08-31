"""LYKENOX V9: phase-increment spectral overlap-add vocoder candidate.

V8 proved that predicting absolute complex STFT coefficients per frame can learn severe
hop-locked repetition even when the fixed STFT/iSTFT geometry is exact. V9 retains the
safe fixed Hann iSTFT renderer but removes the absolute-frame phase degree of freedom:

* the network predicts magnitude and a unit residual phase increment at frame rate;
* each residual phase increment is composed with the deterministic STFT-bin phase advance;
* phase is integrated across frames, so adjacent frames cannot independently reset phase;
* F0/voicing remain frame-rate conditioning only and never become a waveform carrier;
* there is no explicit sinusoidal/noise source, ConvTranspose, or learned sample-rate
  interpolation/upsampling.

The deterministic bin phase advance is analysis/synthesis coordinate geometry, not a
speech excitation source. This candidate does not authorize persistent training.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import LykenoxVocoderConfig


VOCODER_GENERATOR_V9_ARCHITECTURE = "lykenox_phase_increment_spectral_ola_v9"


class _FrameResidualBlockV9(nn.Module):
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
        return (residual + self.output_projection(y)) * (2.0 ** -0.5)


class LykenoxVocoderGeneratorV9(nn.Module):
    """Predict magnitude + inter-frame phase increments and synthesize by fixed iSTFT."""

    architecture = VOCODER_GENERATOR_V9_ARCHITECTURE
    source_free = True
    explicit_sample_rate_source = False
    learned_sample_rate_upsampling = False
    absolute_frame_phase_prediction = False
    phase_representation = "stft_bin_advance_plus_learned_residual_increment"
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
            raise ValueError("V9 currently requires win_length == n_fft")
        self.hidden_channels = int(hidden_channels)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.frequency_bins = self.n_fft // 2 + 1
        self.dilations = (1, 2, 4, 8, 16)

        conditioning_channels = self.config.mel_bins + 2
        self.input_conditioner = nn.Sequential(
            nn.Conv1d(conditioning_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_FrameResidualBlockV9(hidden_channels, dilation) for dilation in self.dilations]
        )
        self.magnitude_head = nn.Conv1d(hidden_channels, self.frequency_bins, kernel_size=1)
        self.phase_residual_head = nn.Conv1d(
            hidden_channels,
            self.frequency_bins * 2,
            kernel_size=1,
        )
        nn.init.normal_(self.magnitude_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.magnitude_head.bias, -2.0)
        nn.init.zeros_(self.phase_residual_head.weight)
        with torch.no_grad():
            self.phase_residual_head.bias[: self.frequency_bins].fill_(1.0)
            self.phase_residual_head.bias[self.frequency_bins :].zero_()

        window = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        self.register_buffer("synthesis_window", window, persistent=True)
        bin_indices = torch.arange(self.frequency_bins, dtype=torch.float32)
        phase_advance = (
            2.0
            * math.pi
            * bin_indices
            * float(self.config.hop_length)
            / float(self.n_fft)
        )
        self.register_buffer(
            "bin_phase_advance",
            torch.polar(torch.ones_like(phase_advance), phase_advance),
            persistent=True,
        )

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
        if mel.shape[1] < 2:
            raise ValueError("V9 requires at least two mel frames")

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
        hidden = self.input_conditioner(conditioning.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return F.pad(hidden, (0, 1), mode="replicate")

    @staticmethod
    def _unit_complex(cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(cosine.square() + sine.square()).clamp_min(1e-6)
        return torch.complex(cosine / norm, sine / norm)

    def predict_spectral_factors(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive magnitude and unit residual phase factors [B,F,T+1]."""
        self._validate_inputs(mel, f0_hz, voiced)
        hidden = self._frame_features(mel, f0_hz, voiced)
        magnitude = F.softplus(self.magnitude_head(hidden)) + 1e-5
        packed_phase = self.phase_residual_head(hidden)
        cosine, sine = packed_phase.chunk(2, dim=1)
        residual_phase = self._unit_complex(cosine, sine)
        expected = (mel.shape[0], self.frequency_bins, mel.shape[1] + 1)
        if tuple(magnitude.shape) != expected or tuple(residual_phase.shape) != expected:
            raise RuntimeError("V9 spectral-factor shape contract failed")
        return magnitude, residual_phase

    def integrate_phase_residual(self, residual_phase: torch.Tensor) -> torch.Tensor:
        """Integrate frame-rate residual phase factors into absolute unit phase."""
        if residual_phase.ndim != 3 or residual_phase.shape[1] != self.frequency_bins:
            raise ValueError("residual_phase must be [batch, frequency_bins, frames]")
        if not torch.is_complex(residual_phase):
            raise ValueError("residual_phase must be complex")
        anchor = residual_phase[..., :1]
        if residual_phase.shape[-1] == 1:
            return anchor
        base = self.bin_phase_advance.to(
            device=residual_phase.device,
            dtype=residual_phase.dtype,
        ).view(1, -1, 1)
        steps = residual_phase[..., 1:] * base
        accumulated = torch.cumprod(steps, dim=-1)
        return torch.cat([anchor, anchor * accumulated], dim=-1)

    def predict_complex_spectrum(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        magnitude, residual_phase = self.predict_spectral_factors(mel, f0_hz, voiced)
        phase = self.integrate_phase_residual(residual_phase)
        return magnitude.to(phase.dtype) * phase

    def target_complex_spectrum(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.win_length,
            window=self.synthesis_window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def factorize_target_spectrum(
        self,
        spectrum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a target STFT into V9 magnitude + residual phase representation."""
        if spectrum.ndim != 3 or spectrum.shape[1] != self.frequency_bins:
            raise ValueError("spectrum must be [batch, frequency_bins, frames]")
        if not torch.is_complex(spectrum):
            raise ValueError("spectrum must be complex")
        magnitude = spectrum.abs()
        unit = spectrum / magnitude.clamp_min(1e-7)
        anchor = unit[..., :1]
        if unit.shape[-1] == 1:
            return magnitude, anchor
        actual_steps = unit[..., 1:] * unit[..., :-1].conj()
        base = self.bin_phase_advance.to(
            device=spectrum.device,
            dtype=spectrum.dtype,
        ).view(1, -1, 1)
        residual_steps = actual_steps * base.conj()
        residual_phase = torch.cat([anchor, residual_steps], dim=-1)
        residual_phase = residual_phase / residual_phase.abs().clamp_min(1e-7)
        return magnitude, residual_phase

    def spectrum_from_factors(
        self,
        magnitude: torch.Tensor,
        residual_phase: torch.Tensor,
    ) -> torch.Tensor:
        if magnitude.shape != residual_phase.shape:
            raise ValueError("magnitude and residual_phase shapes must match")
        phase = self.integrate_phase_residual(residual_phase)
        return magnitude.to(phase.dtype) * phase

    def synthesize_complex_spectrum(
        self,
        spectrum: torch.Tensor,
        *,
        samples: int,
    ) -> torch.Tensor:
        if spectrum.ndim != 3 or spectrum.shape[1] != self.frequency_bins:
            raise ValueError("spectrum must be [batch, frequency_bins, frames]")
        if not torch.is_complex(spectrum):
            raise ValueError("spectrum must be complex")
        return torch.istft(
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

    def forward(
        self,
        mel: torch.Tensor,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
    ) -> torch.Tensor:
        spectrum = self.predict_complex_spectrum(mel, f0_hz, voiced)
        samples = int(mel.shape[1]) * self.config.hop_length
        waveform = self.synthesize_complex_spectrum(spectrum, samples=samples)
        if tuple(waveform.shape) != (mel.shape[0], samples):
            raise RuntimeError("V9 waveform length contract failed")
        return waveform

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def frame_receptive_field(self) -> int:
        return 1 + 4 * sum(self.dilations)
