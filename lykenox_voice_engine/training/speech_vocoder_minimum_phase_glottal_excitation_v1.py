"""LYKENOX-owned calibrated Rosenberg glottal excitation candidate.

This module is an alternative source generator for oracle A/B evaluation.  It does not modify
or replace the production renderer.  All identity-bearing DSP parameters are loaded from two
owned calibration artifacts produced from LYKENOX recordings:

* ``glottal_pulse_v1.json``: open quotient, pulse asymmetry, spectral tilt, residual RMS vs F0.
* ``band_aperiodicity_v1.json``: residual aperiodicity vs F0 in four frequency bands.

There are no learned weights, third-party voice models, remote calls, or gradient updates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from lykenox_voice_engine.training.speech_band_aperiodicity_calibration import (
    BAND_APERIODICITY_CALIBRATION_VERSION,
    BANDS_HZ,
)
from lykenox_voice_engine.training.speech_glottal_calibration import GLOTTAL_CALIBRATION_VERSION
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    SAMPLE_RATE,
    _fixed_lowpass_kernel,
    fixed_linear_frame_to_sample,
)


GLOTTAL_EXCITATION_VERSION = "owned-calibrated-rosenberg-band-aperiodic-excitation-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_GLOTTAL_ARTIFACT = Path("models/lykenox_identity/calibration/glottal_pulse_v1.json")
DEFAULT_APERIODICITY_ARTIFACT = Path(
    "models/lykenox_identity/calibration/band_aperiodicity_v1.json"
)
FILTER_TAPS = 63
BAND_EDGES_HZ = (1000.0, 2000.0, 4000.0, 8000.0)
TILT_REFERENCE_HZ = 1000.0
MAX_TILT_CORRECTION_DB_PER_OCTAVE = 12.0
MAX_TILT_GAIN_DB = 18.0


@dataclass(frozen=True)
class _ScalarCalibrationTable:
    centers_hz: torch.Tensor
    values: torch.Tensor
    global_value: float


@dataclass(frozen=True)
class OwnedCalibratedGlottalExcitationV1:
    glottal_path: Path
    aperiodicity_path: Path
    open_quotient: _ScalarCalibrationTable
    asymmetry: _ScalarCalibrationTable
    tilt_db_per_octave: _ScalarCalibrationTable
    residual_rms: _ScalarCalibrationTable
    band_aperiodicity: tuple[_ScalarCalibrationTable, ...]

    @classmethod
    def from_root(cls, root: Path) -> "OwnedCalibratedGlottalExcitationV1":
        root = Path(root).resolve()
        glottal_path = root / DEFAULT_GLOTTAL_ARTIFACT
        aperiodicity_path = root / DEFAULT_APERIODICITY_ARTIFACT
        glottal = _load_json(glottal_path)
        aperiodicity = _load_json(aperiodicity_path)
        _validate_glottal_artifact(glottal)
        _validate_aperiodicity_artifact(aperiodicity)

        return cls(
            glottal_path=glottal_path,
            aperiodicity_path=aperiodicity_path,
            open_quotient=_metric_table(glottal, "open_quotient"),
            asymmetry=_metric_table(glottal, "asymmetry_peak_position"),
            tilt_db_per_octave=_metric_table(glottal, "spectral_tilt_db_per_octave"),
            residual_rms=_metric_table(glottal, "residual_rms"),
            band_aperiodicity=tuple(
                _band_table(aperiodicity, _band_key(low_hz, high_hz))
                for low_hz, high_hz in BANDS_HZ
            ),
        )

    def build(
        self,
        f0_hz: torch.Tensor,
        voiced: torch.Tensor,
        periodicity: torch.Tensor,
        *,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        noise_seed: int = 0,
    ) -> torch.Tensor:
        """Build calibrated excitation with the same conditioning contract as production."""

        if sample_rate != SAMPLE_RATE or hop_length != HOP_LENGTH:
            raise ValueError("calibrated excitation v1 is bound to the owned 24k/256 geometry")
        for name, value in (("f0_hz", f0_hz), ("voiced", voiced), ("periodicity", periodicity)):
            if value.ndim != 2 or value.is_complex() or not value.is_floating_point():
                raise ValueError(f"{name} must be real floating [batch, frames]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if f0_hz.shape != voiced.shape or f0_hz.shape != periodicity.shape:
            raise ValueError("f0_hz, voiced and periodicity must share shape")

        frame_f0 = f0_hz.clamp_min(0.0)
        frame_voiced = voiced.clamp(0.0, 1.0)
        frame_periodicity = periodicity.clamp(0.0, 1.0)
        frame_oq = _interpolate_table(self.open_quotient, frame_f0).clamp(0.20, 0.95)
        frame_asym = _interpolate_table(self.asymmetry, frame_f0).clamp(0.03, 0.92)
        frame_tilt = _interpolate_table(self.tilt_db_per_octave, frame_f0).clamp(-36.0, 6.0)
        frame_rms = _interpolate_table(self.residual_rms, frame_f0).clamp_min(1.0e-6)
        frame_band_a = [
            _interpolate_table(table, frame_f0).clamp(0.0, 1.0)
            for table in self.band_aperiodicity
        ]

        sample_f0 = fixed_linear_frame_to_sample(frame_f0, hop_length=hop_length)
        sample_voiced = fixed_linear_frame_to_sample(frame_voiced, hop_length=hop_length)
        sample_periodicity = fixed_linear_frame_to_sample(frame_periodicity, hop_length=hop_length)
        sample_oq = fixed_linear_frame_to_sample(frame_oq, hop_length=hop_length)
        sample_asym = fixed_linear_frame_to_sample(frame_asym, hop_length=hop_length)
        sample_tilt = fixed_linear_frame_to_sample(frame_tilt, hop_length=hop_length)
        sample_rms = fixed_linear_frame_to_sample(frame_rms, hop_length=hop_length)
        sample_band_a = [
            fixed_linear_frame_to_sample(value, hop_length=hop_length)
            for value in frame_band_a
        ]

        periodic_source = _build_rosenberg_periodic_source(
            sample_f0,
            sample_voiced,
            sample_oq,
            sample_asym,
            sample_tilt,
            sample_rms,
            sample_rate=sample_rate,
        )
        noise = _deterministic_gaussian_noise_batch(
            periodic_source.shape,
            dtype=periodic_source.dtype,
            device=periodic_source.device,
            seed=int(noise_seed),
        ) * sample_rms

        voiced_periodicity = (sample_voiced * sample_periodicity).clamp(0.0, 1.0)
        kernels = _complementary_band_kernels(
            device=periodic_source.device,
            dtype=periodic_source.dtype,
            sample_rate=sample_rate,
        )
        excitation = torch.zeros_like(periodic_source)
        aperiodicity_for_kernels = [*sample_band_a, sample_band_a[-1]]
        for kernel, calibrated_a in zip(kernels, aperiodicity_for_kernels):
            padding = (int(kernel.numel()) - 1) // 2
            periodic_band = F.conv1d(
                periodic_source.unsqueeze(1),
                kernel.view(1, 1, -1),
                padding=padding,
            ).squeeze(1)
            noise_band = F.conv1d(
                noise.unsqueeze(1),
                kernel.view(1, 1, -1),
                padding=padding,
            ).squeeze(1)
            periodic_gain = voiced_periodicity * torch.sqrt((1.0 - calibrated_a).clamp_min(0.0))
            noise_gain = torch.sqrt((1.0 - periodic_gain.square()).clamp_min(0.0))
            excitation = excitation + periodic_gain * periodic_band + noise_gain * noise_band

        if not bool(torch.isfinite(excitation).all()):
            raise RuntimeError("calibrated glottal excitation produced non-finite values")
        return excitation


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required owned calibration artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"calibration artifact must contain an object: {path}")
    return payload


def _validate_common(payload: dict[str, object], *, expected_version: str) -> None:
    if payload.get("calibration_version") != expected_version:
        raise ValueError("calibration artifact version mismatch")
    if payload.get("policy_id") != POLICY_ID:
        raise ValueError("calibration artifact policy mismatch")
    if payload.get("split") != "train" or payload.get("owned_data_only") is not True:
        raise ValueError("calibration artifact is not owned train-data evidence")
    if payload.get("third_party_model_or_checkpoint_used") is not False:
        raise ValueError("third-party calibration evidence is forbidden")
    if int(payload.get("sample_rate", -1)) != SAMPLE_RATE:
        raise ValueError("calibration sample rate mismatch")


def _validate_glottal_artifact(payload: dict[str, object]) -> None:
    _validate_common(payload, expected_version=GLOTTAL_CALIBRATION_VERSION)
    if not payload.get("f0_bins"):
        raise ValueError("glottal calibration contains no F0 bins")


def _validate_aperiodicity_artifact(payload: dict[str, object]) -> None:
    _validate_common(payload, expected_version=BAND_APERIODICITY_CALIBRATION_VERSION)
    bands = payload.get("bands_hz")
    if not isinstance(bands, list) or len(bands) < 4:
        raise ValueError("aperiodicity calibration must contain at least four bands")


def _metric_table(payload: dict[str, object], metric: str) -> _ScalarCalibrationTable:
    bins = payload["f0_bins"]
    assert isinstance(bins, list)
    centers: list[float] = []
    values: list[float] = []
    for item in bins:
        if not isinstance(item, dict):
            continue
        metric_payload = item.get(metric)
        if not isinstance(metric_payload, dict):
            continue
        centers.append(float(item["f0_center_hz"]))
        values.append(float(metric_payload["median"]))
    global_payload = payload["global"]
    assert isinstance(global_payload, dict)
    global_metric = global_payload[metric]
    assert isinstance(global_metric, dict)
    return _table(centers, values, float(global_metric["median"]))


def _band_key(low_hz: float, high_hz: float) -> str:
    return f"{int(low_hz)}_{int(high_hz)}_hz"


def _band_table(payload: dict[str, object], key: str) -> _ScalarCalibrationTable:
    bins = payload["f0_bins"]
    assert isinstance(bins, list)
    centers: list[float] = []
    values: list[float] = []
    for item in bins:
        if not isinstance(item, dict):
            continue
        bands = item.get("bands")
        if not isinstance(bands, dict) or key not in bands:
            continue
        summary = bands[key]
        if not isinstance(summary, dict):
            continue
        centers.append(float(item["f0_center_hz"]))
        values.append(float(summary["median"]))
    global_payload = payload["global"]
    assert isinstance(global_payload, dict)
    global_bands = global_payload["bands"]
    assert isinstance(global_bands, dict)
    global_summary = global_bands[key]
    assert isinstance(global_summary, dict)
    return _table(centers, values, float(global_summary["median"]))


def _table(centers: list[float], values: list[float], global_value: float) -> _ScalarCalibrationTable:
    if not centers or len(centers) != len(values):
        raise ValueError("calibration table is empty or malformed")
    order = sorted(range(len(centers)), key=centers.__getitem__)
    return _ScalarCalibrationTable(
        centers_hz=torch.tensor([centers[index] for index in order], dtype=torch.float32),
        values=torch.tensor([values[index] for index in order], dtype=torch.float32),
        global_value=float(global_value),
    )


def _interpolate_table(table: _ScalarCalibrationTable, f0_hz: torch.Tensor) -> torch.Tensor:
    centers = table.centers_hz.to(device=f0_hz.device, dtype=f0_hz.dtype)
    values = table.values.to(device=f0_hz.device, dtype=f0_hz.dtype)
    if int(centers.numel()) == 1:
        return torch.full_like(f0_hz, float(values[0]))
    flat = f0_hz.reshape(-1)
    index_right = torch.searchsorted(centers, flat).clamp(1, int(centers.numel()) - 1)
    index_left = index_right - 1
    left_x = centers[index_left]
    right_x = centers[index_right]
    fraction = ((flat - left_x) / (right_x - left_x).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    interpolated = values[index_left] + fraction * (values[index_right] - values[index_left])
    interpolated = torch.where(flat <= centers[0], values[0], interpolated)
    interpolated = torch.where(flat >= centers[-1], values[-1], interpolated)
    interpolated = torch.where(
        flat > 0.0,
        interpolated,
        torch.full_like(interpolated, table.global_value),
    )
    return interpolated.reshape_as(f0_hz)


def _cycle_tilt_db_per_octave(cycle: torch.Tensor, *, sample_rate: int) -> float:
    signal = cycle.to(torch.float64) - cycle.to(torch.float64).mean()
    n_fft = max(256, 1 << int(math.ceil(math.log2(max(2, int(signal.numel()) * 4)))))
    spectrum = torch.fft.rfft(signal * torch.hann_window(int(signal.numel()), periodic=False, dtype=torch.float64), n=n_fft)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).to(torch.float64)
    magnitude_db = 20.0 * torch.log10(spectrum.abs().clamp_min(1.0e-10))
    mask = (freqs >= 300.0) & (freqs <= 8000.0)
    x = torch.log2(freqs[mask] / TILT_REFERENCE_HZ)
    y = magnitude_db[mask]
    if int(x.numel()) < 4:
        return 0.0
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / xc.square().sum().clamp_min(1.0e-12))


def _rosenberg_cycle(
    period_samples: int,
    *,
    open_quotient: float,
    asymmetry_peak_position: float,
    target_tilt_db_per_octave: float,
    target_rms: float,
    sample_rate: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    oq = min(0.95, max(0.20, float(open_quotient)))
    peak_position = min(oq - 0.03, max(0.03, float(asymmetry_peak_position)))
    phase = torch.arange(period_samples, dtype=dtype, device=device) / float(period_samples)
    flow = torch.zeros_like(phase)
    opening = phase < peak_position
    closing = (phase >= peak_position) & (phase < oq)
    flow[opening] = 0.5 * (
        1.0 - torch.cos(math.pi * phase[opening] / max(peak_position, 1.0e-4))
    )
    close_fraction = (phase[closing] - peak_position) / max(oq - peak_position, 1.0e-4)
    flow[closing] = torch.cos(0.5 * math.pi * close_fraction)

    source = flow - torch.roll(flow, shifts=1, dims=0)
    source[0] = flow[0]
    source = source - source.mean()
    native_tilt = _cycle_tilt_db_per_octave(source.detach().cpu(), sample_rate=sample_rate)
    correction = max(
        -MAX_TILT_CORRECTION_DB_PER_OCTAVE,
        min(MAX_TILT_CORRECTION_DB_PER_OCTAVE, float(target_tilt_db_per_octave) - native_tilt),
    )
    spectrum = torch.fft.rfft(source)
    freqs = torch.fft.rfftfreq(period_samples, d=1.0 / float(sample_rate)).to(
        device=device, dtype=dtype
    )
    octave = torch.log2(freqs.clamp_min(float(sample_rate) / period_samples) / TILT_REFERENCE_HZ)
    gain_db = (correction * octave).clamp(-MAX_TILT_GAIN_DB, MAX_TILT_GAIN_DB)
    gain = torch.pow(torch.tensor(10.0, dtype=dtype, device=device), gain_db / 20.0)
    if gain.numel():
        gain[0] = 0.0
    shaped = torch.fft.irfft(spectrum * gain.to(spectrum.dtype), n=period_samples)
    shaped = shaped - shaped.mean()
    rms = torch.sqrt(shaped.square().mean().clamp_min(1.0e-12))
    return shaped * (float(target_rms) / rms)


def _build_rosenberg_periodic_source(
    sample_f0: torch.Tensor,
    sample_voiced: torch.Tensor,
    sample_oq: torch.Tensor,
    sample_asym: torch.Tensor,
    sample_tilt: torch.Tensor,
    sample_rms: torch.Tensor,
    *,
    sample_rate: int,
) -> torch.Tensor:
    batch, sample_count = sample_f0.shape
    output = torch.zeros_like(sample_f0)
    for batch_index in range(batch):
        f0 = sample_f0[batch_index]
        voiced = sample_voiced[batch_index]
        phase_increment = torch.where(
            (f0 > 0.0) & (voiced >= 0.5),
            f0 / float(sample_rate),
            torch.zeros_like(f0),
        )
        phase = torch.cumsum(phase_increment, dim=0)
        previous = torch.cat((torch.zeros(1, dtype=phase.dtype, device=phase.device), phase[:-1]))
        crossings = torch.nonzero(torch.floor(phase) > torch.floor(previous), as_tuple=False).flatten()
        if int(crossings.numel()) < 2:
            continue
        for left_tensor, right_tensor in zip(crossings[:-1], crossings[1:]):
            left = int(left_tensor)
            right = int(right_tensor)
            period = right - left
            if period < 8 or period > sample_rate // 20:
                continue
            middle = min(sample_count - 1, left + period // 2)
            if float(voiced[left:right].mean()) < 0.5:
                continue
            cycle = _rosenberg_cycle(
                period,
                open_quotient=float(sample_oq[batch_index, middle]),
                asymmetry_peak_position=float(sample_asym[batch_index, middle]),
                target_tilt_db_per_octave=float(sample_tilt[batch_index, middle]),
                target_rms=float(sample_rms[batch_index, middle]),
                sample_rate=sample_rate,
                dtype=output.dtype,
                device=output.device,
            )
            output[batch_index, left:right] = cycle
    return output


def _deterministic_gaussian_noise_batch(
    shape: torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    rows = []
    for batch_index in range(int(shape[0])):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1009 * batch_index)
        row = torch.randn(int(shape[1]), generator=generator, dtype=dtype, device="cpu")
        rows.append(row)
    return torch.stack(rows, dim=0).to(device=device)


def _complementary_band_kernels(
    *,
    device: torch.device,
    dtype: torch.dtype,
    sample_rate: int,
) -> tuple[torch.Tensor, ...]:
    lowpasses = [
        _fixed_lowpass_kernel(
            device=device,
            dtype=dtype,
            taps=FILTER_TAPS,
            cutoff_hz=edge,
            sample_rate=sample_rate,
        )
        for edge in BAND_EDGES_HZ
    ]
    delta = torch.zeros_like(lowpasses[0])
    delta[(FILTER_TAPS - 1) // 2] = 1.0
    return (
        lowpasses[0],
        lowpasses[1] - lowpasses[0],
        lowpasses[2] - lowpasses[1],
        lowpasses[3] - lowpasses[2],
        delta - lowpasses[3],
    )


__all__ = [
    "GLOTTAL_EXCITATION_VERSION",
    "OwnedCalibratedGlottalExcitationV1",
]
