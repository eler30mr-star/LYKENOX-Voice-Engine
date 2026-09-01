"""Fixed minimum-phase renderer for the next LYKENOX-owned vocoder gate.

This module contains no neural model, learned parameter, optimizer, checkpoint IO, trainer,
or product-quality acceptance logic.  It implements only the deterministic signal path that
must be proven safe before a trainable envelope predictor may exist.

The trainable representation selected by the architecture contract is a frame-rate one-sided
real cepstrum.  The fixed transform doubles positive-quefrency coefficients, exponentiates
the resulting causal complex log spectrum, and converts it to a minimum-phase impulse
response.  A neutral voiced/aperiodic excitation is then filtered with linearly interpolated
frame filters.  The excitation has no direct output bypass.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


RENDERER_VERSION = "owned-minimum-phase-time-varying-renderer-v1"
SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_FFT = 1024
CEPSTRAL_ORDER = 64
LOWPASS_TAPS = 63
LOWPASS_CUTOFF_HZ = 10800.0


def _require_real_floating(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.is_complex() or not tensor.is_floating_point():
        raise ValueError(f"{name} must be a real floating tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def cepstral_log_magnitude(
    cepstrum: torch.Tensor,
    *,
    n_fft: int = N_FFT,
) -> torch.Tensor:
    """Return log magnitude represented by one-sided real cepstral coefficients.

    ``cepstrum[..., 0]`` is the DC quefrency coefficient and positive coefficients describe
    the even log-magnitude series as ``c0 + 2*sum(c[k]*cos(k*w))``.
    """

    _require_real_floating(cepstrum, name="cepstrum")
    if cepstrum.ndim < 1:
        raise ValueError("cepstrum must have at least one dimension")
    order = int(cepstrum.shape[-1])
    if order < 1 or order > n_fft // 2:
        raise ValueError("cepstral order must be in [1, n_fft/2]")
    causal = torch.zeros(
        *cepstrum.shape[:-1],
        n_fft,
        device=cepstrum.device,
        dtype=cepstrum.dtype,
    )
    causal[..., 0] = cepstrum[..., 0]
    if order > 1:
        causal[..., 1:order] = 2.0 * cepstrum[..., 1:]
    return torch.fft.rfft(causal, n=n_fft, dim=-1).real


def one_sided_real_cepstrum_to_minimum_phase_fir(
    cepstrum: torch.Tensor,
    *,
    n_fft: int = N_FFT,
) -> torch.Tensor:
    """Convert one-sided real cepstrum to a fixed-length minimum-phase FIR realization."""

    _require_real_floating(cepstrum, name="cepstrum")
    if cepstrum.ndim < 1:
        raise ValueError("cepstrum must have at least one dimension")
    order = int(cepstrum.shape[-1])
    if order < 1 or order > n_fft // 2:
        raise ValueError("cepstral order must be in [1, n_fft/2]")

    causal = torch.zeros(
        *cepstrum.shape[:-1],
        n_fft,
        device=cepstrum.device,
        dtype=cepstrum.dtype,
    )
    causal[..., 0] = cepstrum[..., 0]
    if order > 1:
        causal[..., 1:order] = 2.0 * cepstrum[..., 1:]

    complex_log_transfer = torch.fft.rfft(causal, n=n_fft, dim=-1)
    transfer = torch.exp(complex_log_transfer)
    impulse = torch.fft.irfft(transfer, n=n_fft, dim=-1)
    if not torch.isfinite(impulse).all():
        raise ValueError("minimum-phase transform produced non-finite impulse response")
    return impulse


def reference_log_magnitude_to_one_sided_cepstrum(
    log_magnitude: torch.Tensor,
    *,
    cepstral_order: int = CEPSTRAL_ORDER,
    n_fft: int = N_FFT,
) -> torch.Tensor:
    """Diagnostic-only magnitude oracle transform used to test renderer expressivity.

    This is not an inference path.  It converts an owned reference log magnitude into the
    exact representation selected by the architecture contract, truncated to the contracted
    cepstral order.
    """

    _require_real_floating(log_magnitude, name="log_magnitude")
    if log_magnitude.shape[-1] != n_fft // 2 + 1:
        raise ValueError("log_magnitude has the wrong rFFT bin count")
    if cepstral_order < 1 or cepstral_order > n_fft // 2:
        raise ValueError("invalid cepstral_order")
    full_real_cepstrum = torch.fft.irfft(log_magnitude, n=n_fft, dim=-1)
    return full_real_cepstrum[..., :cepstral_order]


def fixed_linear_frame_to_sample(
    frames: torch.Tensor,
    *,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """Linearly interpolate frame values onto an exact ``frames*hop`` sample grid."""

    _require_real_floating(frames, name="frames")
    if frames.ndim != 2:
        raise ValueError("frames must have shape [batch, frame_count]")
    if frames.shape[-1] < 1 or hop_length < 2:
        raise ValueError("invalid frame_count or hop_length")

    batch, frame_count = frames.shape
    sample_count = frame_count * hop_length
    sample_index = torch.arange(sample_count, device=frames.device)
    left = torch.div(sample_index, hop_length, rounding_mode="floor")
    right = torch.clamp(left + 1, max=frame_count - 1)
    fraction = (sample_index.remainder(hop_length)).to(frames.dtype) / float(hop_length)
    left_value = frames[:, left]
    right_value = frames[:, right]
    return left_value + (right_value - left_value) * fraction.unsqueeze(0)


def _fixed_lowpass_kernel(
    *,
    device: torch.device,
    dtype: torch.dtype,
    taps: int = LOWPASS_TAPS,
    cutoff_hz: float = LOWPASS_CUTOFF_HZ,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    if taps < 3 or taps % 2 == 0:
        raise ValueError("lowpass taps must be odd and >=3")
    if not 0.0 < cutoff_hz < sample_rate / 2.0:
        raise ValueError("lowpass cutoff must be below Nyquist")
    center = (taps - 1) / 2.0
    n = torch.arange(taps, device=device, dtype=dtype) - center
    normalized_cutoff = cutoff_hz / float(sample_rate)
    ideal = 2.0 * normalized_cutoff * torch.sinc(2.0 * normalized_cutoff * n)
    window = torch.hann_window(taps, periodic=False, device=device, dtype=dtype)
    kernel = ideal * window
    kernel = kernel / kernel.sum()
    return kernel


def _deterministic_aperiodic_noise(
    sample_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Stateless deterministic pseudo-noise with no learned identity parameters."""

    index = torch.arange(sample_count, device=device, dtype=dtype)
    phase = (index + float(seed) * 131.0) * 12.9898 + 78.233
    hashed = torch.sin(phase) * 43758.5453123
    unit = hashed - torch.floor(hashed)
    return unit.mul(2.0).sub(1.0)


def build_neutral_excitation(
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    noise_seed: int = 0,
) -> torch.Tensor:
    """Build fixed band-limited pulse plus deterministic aperiodic excitation.

    Inputs are full-utterance-cache frame values supplied by the V2 data contract.  They are
    interpolated by a fixed linear rule; pitch is never re-estimated from waveform crops.
    """

    for name, value in (("f0_hz", f0_hz), ("voiced", voiced), ("periodicity", periodicity)):
        _require_real_floating(value, name=name)
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, frame_count]")
    if f0_hz.shape != voiced.shape or f0_hz.shape != periodicity.shape:
        raise ValueError("f0_hz, voiced and periodicity must share shape")
    if sample_rate < 1 or hop_length < 2:
        raise ValueError("invalid sample_rate or hop_length")

    f0 = fixed_linear_frame_to_sample(f0_hz.clamp_min(0.0), hop_length=hop_length)
    voiced_sample = fixed_linear_frame_to_sample(voiced.clamp(0.0, 1.0), hop_length=hop_length)
    periodicity_sample = fixed_linear_frame_to_sample(
        periodicity.clamp(0.0, 1.0), hop_length=hop_length
    )
    periodic_strength = (voiced_sample * periodicity_sample).clamp(0.0, 1.0)

    phase_increment = torch.where(
        f0 > 0.0,
        f0 / float(sample_rate),
        torch.zeros_like(f0),
    )
    accumulated = torch.cumsum(phase_increment, dim=-1)
    previous = F.pad(accumulated[..., :-1], (1, 0), value=0.0)
    pulse = (torch.floor(accumulated) > torch.floor(previous)).to(f0.dtype)
    pulse_scale = torch.where(
        f0 > 1.0,
        torch.sqrt(float(sample_rate) / f0.clamp_min(1.0)),
        torch.zeros_like(f0),
    )
    pulse = pulse * pulse_scale

    kernel = _fixed_lowpass_kernel(device=f0.device, dtype=f0.dtype, sample_rate=sample_rate)
    padding = (kernel.numel() - 1) // 2
    bandlimited_pulse = F.conv1d(
        pulse.unsqueeze(1),
        kernel.view(1, 1, -1),
        padding=padding,
    ).squeeze(1)

    base_noise = _deterministic_aperiodic_noise(
        f0.shape[-1],
        device=f0.device,
        dtype=f0.dtype,
        seed=int(noise_seed),
    ).unsqueeze(0).expand(f0.shape[0], -1)
    aperiodic_strength = torch.sqrt((1.0 - periodic_strength.square()).clamp_min(0.0))
    excitation = periodic_strength * bandlimited_pulse + aperiodic_strength * base_noise
    if not torch.isfinite(excitation).all():
        raise ValueError("neutral excitation produced non-finite values")
    return excitation


def render_time_varying_minimum_phase(
    excitation: torch.Tensor,
    cepstrum: torch.Tensor,
    *,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> torch.Tensor:
    """Filter excitation with fixed linear interpolation between frame minimum-phase FIRs."""

    _require_real_floating(excitation, name="excitation")
    _require_real_floating(cepstrum, name="cepstrum")
    if excitation.ndim != 2:
        raise ValueError("excitation must have shape [batch, samples]")
    if cepstrum.ndim != 3:
        raise ValueError("cepstrum must have shape [batch, frame_count, order]")
    if excitation.shape[0] != cepstrum.shape[0]:
        raise ValueError("excitation and cepstrum batch sizes must match")
    if excitation.shape[-1] != cepstrum.shape[1] * hop_length:
        raise ValueError("excitation length must equal frame_count*hop_length")

    filters = one_sided_real_cepstrum_to_minimum_phase_fir(cepstrum, n_fft=n_fft)
    batch, frame_count, _ = filters.shape
    padded = F.pad(excitation, (n_fft - 1, 0))
    output_blocks: list[torch.Tensor] = []
    alpha = torch.linspace(
        0.0,
        1.0,
        hop_length,
        device=excitation.device,
        dtype=excitation.dtype,
    ).view(1, hop_length)

    for frame_index in range(frame_count):
        start = frame_index * hop_length
        local = padded[:, start : start + hop_length + n_fft - 1]
        windows = local.unfold(-1, n_fft, 1)
        current_filter = filters[:, frame_index, :]
        current = (windows * current_filter.flip(-1).unsqueeze(1)).sum(dim=-1)
        if frame_index == 0:
            block = current
        else:
            previous_filter = filters[:, frame_index - 1, :]
            previous_output = (windows * previous_filter.flip(-1).unsqueeze(1)).sum(dim=-1)
            block = previous_output + (current - previous_output) * alpha
        output_blocks.append(block)

    waveform = torch.cat(output_blocks, dim=-1)
    expected = frame_count * hop_length
    if waveform.shape[-1] != expected:
        raise RuntimeError("renderer violated exact output-length contract")
    if not torch.isfinite(waveform).all():
        raise ValueError("renderer produced non-finite waveform")
    return waveform


def render_owned_minimum_phase_vocoder_path(
    cepstrum: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
    *,
    noise_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render through the only authorized fixed source->filter path.

    Returns ``(waveform, excitation)`` so structural audits can prove that a flat filter is
    identity and a strongly attenuating filter suppresses the source.  Product code must use
    the waveform; the excitation return exists solely for renderer verification.
    """

    if cepstrum.ndim != 3:
        raise ValueError("cepstrum must have shape [batch, frame_count, order]")
    frame_shape = cepstrum.shape[:2]
    if f0_hz.shape != frame_shape or voiced.shape != frame_shape or periodicity.shape != frame_shape:
        raise ValueError("conditioning tensors must match cepstrum batch/frame dimensions")
    excitation = build_neutral_excitation(
        f0_hz,
        voiced,
        periodicity,
        noise_seed=noise_seed,
    )
    waveform = render_time_varying_minimum_phase(excitation, cepstrum)
    return waveform, excitation
