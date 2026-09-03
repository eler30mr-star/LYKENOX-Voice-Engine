"""Coherent owned pitch-conditioning contract for LYKENOX source models.

Pitch-v1 exposes three values with incompatible transition semantics: F0 is forced to zero when a
binary voiced decision is false, while raw autocorrelation periodicity remains non-zero. Learned
sources therefore receive combinations such as F0=0, voiced=0, periodicity=0.4 at low-energy speech
boundaries. This module preserves the trusted v1 anchor decisions but exposes a coherent continuous
contract for future source training:

- ``f0_track_hz`` keeps every trusted v1 voiced-anchor F0 exactly and fills non-anchor gaps by
  log-frequency interpolation/edge hold, so phase coordinates never reset merely because confidence
  becomes low;
- ``energy_confidence`` is a smooth form of the existing v1 RMS criterion using the same
  ``voiced_rms_fraction`` parameter;
- ``periodic_strength`` is raw normalized autocorrelation strength multiplied by energy confidence,
  so low-energy noise cannot carry strong periodic authority;
- no binary voiced flag is part of the product conditioning contract.

This is deterministic LYKENOX-owned DSP. It trains no model, uses no external weights/services, and
performs no waveform post-processing. Policy: LYX-POL-001.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from lykenox_voice_engine.training.speech_pitch import _pitch_lag_bounds


PITCH_CONDITIONING_V2 = "lykenox-pitch-conditioning-v2-continuous-strength"
EPSILON = 1.0e-8


@dataclass(frozen=True)
class PitchConditioningV2:
    f0_track_hz: torch.Tensor
    periodic_strength: torch.Tensor
    energy_confidence: torch.Tensor
    raw_periodicity: torch.Tensor
    anchor_voiced: torch.Tensor
    frame_rms: torch.Tensor


def _centered_windowed_frames(
    waveform: torch.Tensor,
    *,
    frame_count: int,
    frame_length: int,
    hop_length: int,
) -> torch.Tensor:
    if waveform.ndim != 1:
        raise ValueError("waveform must be mono [samples]")
    if frame_count < 1 or frame_length < 64 or hop_length < 1:
        raise ValueError("invalid pitch-conditioning geometry")
    half = frame_length // 2
    padded = F.pad(waveform.unsqueeze(0).unsqueeze(0), (half, half), mode="reflect")
    frames = padded.squeeze(0).squeeze(0).unfold(0, frame_length, hop_length)
    if int(frames.shape[0]) < frame_count:
        missing = frame_count - int(frames.shape[0])
        extended = F.pad(
            waveform.unsqueeze(0).unsqueeze(0),
            (0, missing * hop_length),
            mode="reflect",
        )
        padded = F.pad(extended, (half, half), mode="reflect")
        frames = padded.squeeze(0).squeeze(0).unfold(0, frame_length, hop_length)
    if int(frames.shape[0]) < frame_count:
        raise RuntimeError("pitch-conditioning framing produced too few frames")
    frames = frames[:frame_count].to(torch.float32)
    window = torch.hann_window(frame_length, dtype=frames.dtype, device=frames.device)
    frames = frames * window.unsqueeze(0)
    return frames - frames.mean(dim=1, keepdim=True)


def _fill_continuous_f0(anchor_f0: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Preserve anchor F0 exactly and fill all other frames continuously in log-frequency."""
    if anchor_f0.ndim != 1 or anchors.shape != anchor_f0.shape:
        raise ValueError("anchor F0 geometry mismatch")
    count = int(anchor_f0.numel())
    indices = torch.nonzero(anchors, as_tuple=False).flatten()
    if not int(indices.numel()):
        return torch.zeros_like(anchor_f0)

    result = torch.empty_like(anchor_f0)
    first = int(indices[0])
    last = int(indices[-1])
    result[: first + 1] = anchor_f0[first]
    result[last:] = anchor_f0[last]

    for left_tensor, right_tensor in zip(indices[:-1], indices[1:]):
        left = int(left_tensor)
        right = int(right_tensor)
        left_f0 = anchor_f0[left].clamp_min(EPSILON)
        right_f0 = anchor_f0[right].clamp_min(EPSILON)
        span = right - left
        if span < 1:
            continue
        fraction = torch.arange(span + 1, dtype=anchor_f0.dtype, device=anchor_f0.device) / float(span)
        log_left = torch.log(left_f0)
        log_right = torch.log(right_f0)
        result[left : right + 1] = torch.exp(log_left + (log_right - log_left) * fraction)

    result[indices] = anchor_f0[indices]
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("continuous F0 interpolation produced non-finite values")
    return result.contiguous()


def extract_pitch_conditioning_v2(
    waveform: torch.Tensor,
    *,
    frame_count: int,
    sample_rate: int = 24000,
    hop_length: int = 256,
    frame_length: int = 1024,
    min_f0_hz: float = 60.0,
    max_f0_hz: float = 350.0,
    anchor_periodicity_threshold: float = 0.30,
    anchor_rms_fraction: float = 0.08,
) -> PitchConditioningV2:
    """Extract the continuous F0/periodic-strength conditioning contract from owned waveform data."""
    if sample_rate < 1 or not 0.0 < anchor_rms_fraction < 1.0:
        raise ValueError("invalid conditioning configuration")
    frames = _centered_windowed_frames(
        waveform,
        frame_count=frame_count,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    frame_rms = torch.sqrt(frames.square().mean(dim=1).clamp_min(EPSILON * EPSILON))
    spectrum = torch.fft.rfft(frames, n=frame_length * 2, dim=1)
    autocorrelation = torch.fft.irfft(
        spectrum * spectrum.conj(), n=frame_length * 2, dim=1
    ).real[:, :frame_length]
    normalized = autocorrelation / autocorrelation[:, :1].clamp_min(EPSILON)

    min_lag, max_lag = _pitch_lag_bounds(
        sample_rate=sample_rate,
        frame_length=frame_length,
        min_f0_hz=min_f0_hz,
        max_f0_hz=max_f0_hz,
    )
    candidates = normalized[:, min_lag : max_lag + 1]
    raw_periodicity, relative_lag = torch.max(candidates, dim=1)
    lag = relative_lag + min_lag
    candidate_f0 = float(sample_rate) / lag.to(torch.float32)

    rms_reference = frame_rms.max().clamp_min(EPSILON)
    rms_threshold = rms_reference * float(anchor_rms_fraction)
    anchor_voiced = (raw_periodicity >= float(anchor_periodicity_threshold)) & (frame_rms >= rms_threshold)
    anchor_f0 = torch.where(anchor_voiced, candidate_f0, torch.zeros_like(candidate_f0))
    f0_track = _fill_continuous_f0(anchor_f0, anchor_voiced)

    # Smooth counterpart of the v1 energy threshold: confidence is 0.5 exactly at the historical
    # RMS threshold, approaches zero with vanishing energy and one for strong frames.
    energy_confidence = frame_rms / (frame_rms + rms_threshold + EPSILON)
    periodic_strength = raw_periodicity.clamp(0.0, 1.0) * energy_confidence.clamp(0.0, 1.0)

    for name, value in (
        ("f0_track_hz", f0_track),
        ("periodic_strength", periodic_strength),
        ("energy_confidence", energy_confidence),
        ("raw_periodicity", raw_periodicity),
        ("frame_rms", frame_rms),
    ):
        if value.shape != (frame_count,) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"invalid {name} in pitch-conditioning v2")
    if not bool(((periodic_strength >= 0.0) & (periodic_strength <= 1.0)).all()):
        raise RuntimeError("periodic_strength left [0,1]")
    if not bool(((energy_confidence >= 0.0) & (energy_confidence <= 1.0)).all()):
        raise RuntimeError("energy_confidence left [0,1]")
    if int(anchor_voiced.sum()) and not torch.equal(f0_track[anchor_voiced], anchor_f0[anchor_voiced]):
        raise RuntimeError("trusted v1 anchor F0 values were modified")

    return PitchConditioningV2(
        f0_track_hz=f0_track,
        periodic_strength=periodic_strength.contiguous(),
        energy_confidence=energy_confidence.contiguous(),
        raw_periodicity=raw_periodicity.contiguous(),
        anchor_voiced=anchor_voiced.to(torch.float32).contiguous(),
        frame_rms=frame_rms.contiguous(),
    )


__all__ = [
    "PITCH_CONDITIONING_V2",
    "PitchConditioningV2",
    "extract_pitch_conditioning_v2",
]
