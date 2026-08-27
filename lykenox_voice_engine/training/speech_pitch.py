"""Deterministic CPU pitch/voicing features for LYKENOX speech training.

This is not a runtime dependency on an external pitch tracker. It uses only PyTorch FFT
operations to obtain a bootstrap F0/voicing target from owned waveform data. The target
will later supervise the speech acoustic model and condition the LYKENOX source-filter
vocoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


PITCH_TARGET_VERSION = "lykenox-pitch-v1"


@dataclass(frozen=True)
class PitchFrames:
    f0_hz: torch.Tensor
    voiced: torch.Tensor
    periodicity: torch.Tensor


def _pitch_lag_bounds(
    *,
    sample_rate: int,
    frame_length: int,
    min_f0_hz: float,
    max_f0_hz: float,
) -> tuple[int, int]:
    """Return the exact integer-lag search interval used by pitch-v1.

    Pitch-v1 historically converts the requested F0 bounds to integer autocorrelation
    lags with ``int``. That quantization is part of the accepted v1 target semantics.
    For example, 24 kHz / 350 Hz becomes lag 68, whose realizable F0 is
    352.941... Hz. Keep this helper behaviorally identical to the original extractor.
    """

    if sample_rate < 1 or frame_length < 64:
        raise ValueError("invalid pitch analysis dimensions")
    if not 0.0 < min_f0_hz < max_f0_hz:
        raise ValueError("invalid pitch range")
    min_lag = max(1, int(sample_rate / max_f0_hz))
    max_lag = min(frame_length - 2, int(sample_rate / min_f0_hz))
    if max_lag < min_lag:
        raise ValueError("pitch range has no realizable integer-lag candidates")
    return min_lag, max_lag


def pitch_search_bounds_hz(
    *,
    sample_rate: int = 24000,
    frame_length: int = 1024,
    min_f0_hz: float = 60.0,
    max_f0_hz: float = 350.0,
) -> tuple[float, float]:
    """Return the effective F0 bounds realizable by the pitch-v1 integer-lag grid.

    These are validation bounds, not a new extraction algorithm. The nominal configured
    interval remains part of provenance; this function exposes the discrete frequencies
    that the already-versioned extractor can actually emit.
    """

    min_lag, max_lag = _pitch_lag_bounds(
        sample_rate=sample_rate,
        frame_length=frame_length,
        min_f0_hz=min_f0_hz,
        max_f0_hz=max_f0_hz,
    )
    return (
        float(sample_rate) / float(max_lag),
        float(sample_rate) / float(min_lag),
    )


def extract_pitch_frames(
    waveform: torch.Tensor,
    *,
    frame_count: int,
    sample_rate: int = 24000,
    hop_length: int = 256,
    frame_length: int = 1024,
    min_f0_hz: float = 60.0,
    max_f0_hz: float = 350.0,
    voiced_periodicity_threshold: float = 0.30,
    voiced_rms_fraction: float = 0.08,
) -> PitchFrames:
    """Extract one F0/voicing target per mel frame using FFT autocorrelation.

    ``waveform`` is a mono tensor ``[samples]``. The returned tensors have exactly
    ``frame_count`` entries and use the same centered-frame convention as the speech
    mel frontend. Unvoiced F0 values are zero.

    Full utterances do not normally contain exactly ``frame_count * hop_length`` samples
    because the mel frontend uses centered STFT frames. Earlier vocoder probes happened
    to use exact hop-sized crops. The pitch algorithm itself is unchanged; this function
    now accepts both contracts and pads only when a caller requests a final centered frame
    beyond the available waveform edge.
    """

    if waveform.ndim != 1:
        raise ValueError("waveform must be mono [samples]")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if sample_rate < 1 or hop_length < 1 or frame_length < 64:
        raise ValueError("invalid pitch analysis dimensions")
    if not 0.0 < min_f0_hz < max_f0_hz:
        raise ValueError("invalid pitch range")
    if waveform.numel() < 2:
        raise ValueError("waveform is too short for pitch analysis")

    half = frame_length // 2
    # Reflect padding mirrors torchaudio's centered mel framing convention. For ordinary
    # full utterances this already yields floor(samples / hop) + 1 frames, matching the
    # active center=True mel frontend. Exact-hop vocoder crops yield one extra trailing
    # centered frame; slicing below preserves their existing v1 target values exactly.
    padded = F.pad(waveform.unsqueeze(0).unsqueeze(0), (half, half), mode="reflect")
    frames = padded.squeeze(0).squeeze(0).unfold(0, frame_length, hop_length)
    if int(frames.shape[0]) < frame_count:
        missing = frame_count - int(frames.shape[0])
        right_pad = missing * hop_length
        extended = F.pad(
            waveform.unsqueeze(0).unsqueeze(0),
            (0, right_pad),
            mode="reflect",
        )
        padded = F.pad(extended, (half, half), mode="reflect")
        frames = padded.squeeze(0).squeeze(0).unfold(0, frame_length, hop_length)
    if int(frames.shape[0]) < frame_count:
        raise RuntimeError("pitch framing produced too few frames")
    frames = frames[:frame_count].to(torch.float32)

    window = torch.hann_window(frame_length, device=frames.device, dtype=frames.dtype)
    frames = frames * window.unsqueeze(0)
    frames = frames - frames.mean(dim=1, keepdim=True)
    rms = torch.sqrt(frames.square().mean(dim=1) + 1e-12)

    spectrum = torch.fft.rfft(frames, n=frame_length * 2, dim=1)
    autocorrelation = torch.fft.irfft(
        spectrum * spectrum.conj(), n=frame_length * 2, dim=1
    ).real[:, :frame_length]
    zero_lag = autocorrelation[:, :1].clamp_min(1e-8)
    normalized = autocorrelation / zero_lag

    min_lag, max_lag = _pitch_lag_bounds(
        sample_rate=sample_rate,
        frame_length=frame_length,
        min_f0_hz=min_f0_hz,
        max_f0_hz=max_f0_hz,
    )
    candidates = normalized[:, min_lag : max_lag + 1]
    periodicity, relative_lag = torch.max(candidates, dim=1)
    lag = relative_lag + min_lag
    f0 = sample_rate / lag.to(torch.float32)

    rms_threshold = rms.max().clamp_min(1e-8) * float(voiced_rms_fraction)
    voiced = (periodicity >= voiced_periodicity_threshold) & (rms >= rms_threshold)
    f0 = torch.where(voiced, f0, torch.zeros_like(f0))
    return PitchFrames(
        f0_hz=f0.contiguous(),
        voiced=voiced.to(torch.float32).contiguous(),
        periodicity=periodicity.contiguous(),
    )
