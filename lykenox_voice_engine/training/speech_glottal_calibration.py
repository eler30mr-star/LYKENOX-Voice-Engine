"""Owned deterministic glottal-source calibration from LYKENOX recordings.

This module measures pitch-synchronous source statistics from the real residual demonstrated by
``scripts/diagnostic_real_residual_resynthesis_v1.py``.  It is calibration, not model training:
no learned model, optimizer, checkpoint, external voice component, or remote service is used.

The calibration artifact is derived only from the repository's owned/authorized training split
and records source WAV hashes plus algorithm/data-contract provenance under LYX-POL-001.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch

from lykenox_voice_engine.training.speech_vocoder_loss_v2 import _centered_stft_magnitude
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    FULL_UTTERANCE_DATA_VERSION,
    OwnedVocoderUtterance,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    fixed_linear_frame_to_sample,
    reference_log_magnitude_to_one_sided_cepstrum,
)


GLOTTAL_CALIBRATION_VERSION = "owned-glottal-pulse-calibration-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_ITEMS = 1_000_000
F0_BIN_HZ = 20.0
TRANSFER_EPSILON = 1.0e-6
OPEN_ENERGY_FRACTION = 0.25
MIN_CYCLE_SAMPLES = 24
MAX_CYCLE_SAMPLES = 800
TILT_FMIN_HZ = 300.0
TILT_FMAX_HZ = 8000.0
TILT_N_FFT = 1024


@dataclass(frozen=True)
class ResidualCycle:
    utterance_id: str
    f0_hz: float
    cycle: torch.Tensor


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_log_magnitude(waveform: torch.Tensor, *, frame_count: int) -> torch.Tensor:
    magnitude = _centered_stft_magnitude(
        waveform.unsqueeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
    )
    if int(magnitude.shape[1]) != N_FFT // 2 + 1:
        raise RuntimeError("reference STFT bin geometry changed")
    if int(magnitude.shape[-1]) < frame_count:
        raise RuntimeError("reference STFT has fewer frames than conditioning")
    return torch.log(magnitude[0, :, :frame_count].transpose(0, 1).clamp_min(1.0e-5))


def _minimum_phase_transfer_from_cepstrum(cepstrum: torch.Tensor) -> torch.Tensor:
    if cepstrum.ndim != 2:
        raise ValueError("cepstrum must have shape [frames, order]")
    order = int(cepstrum.shape[-1])
    causal = torch.zeros(
        int(cepstrum.shape[0]),
        N_FFT,
        dtype=cepstrum.dtype,
        device=cepstrum.device,
    )
    causal[:, 0] = cepstrum[:, 0]
    if order > 1:
        causal[:, 1:order] = 2.0 * cepstrum[:, 1:]
    return torch.exp(torch.fft.rfft(causal, n=N_FFT, dim=-1))


def extract_owned_real_residual(
    waveform: torch.Tensor,
    *,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (real_residual, oracle_cepstrum, terminal_transfer_extension_frames)."""

    waveform = waveform.to(torch.float32).contiguous()
    expected_samples = int(frame_count) * HOP_LENGTH
    if int(waveform.numel()) != expected_samples:
        raise ValueError("waveform must have frame_count * hop_length samples")

    log_magnitude = _reference_log_magnitude(waveform, frame_count=frame_count)
    cepstrum = reference_log_magnitude_to_one_sided_cepstrum(
        log_magnitude,
        cepstral_order=CEPSTRAL_ORDER,
        n_fft=N_FFT,
    )
    transfer = _minimum_phase_transfer_from_cepstrum(cepstrum)

    window = torch.hann_window(N_FFT, dtype=waveform.dtype, device=waveform.device)
    spectrum = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )
    analysis_frames = int(spectrum.shape[-1])
    extension_frames = analysis_frames - frame_count
    if extension_frames < 0:
        raise RuntimeError("complex STFT has fewer frames than conditioning")
    if extension_frames:
        transfer = torch.cat(
            (transfer, transfer[-1:, :].expand(extension_frames, -1)),
            dim=0,
        )
    residual_spectrum = spectrum / (transfer.transpose(0, 1).to(spectrum.dtype) + TRANSFER_EPSILON)
    residual = torch.istft(
        residual_spectrum,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        length=expected_samples,
    )
    if not bool(torch.isfinite(residual).all()):
        raise RuntimeError("real residual contains non-finite values")
    return residual.to(torch.float32).contiguous(), cepstrum, extension_frames


def _pitch_synchronous_cycles(
    utterance: OwnedVocoderUtterance,
    residual: torch.Tensor,
) -> Iterator[ResidualCycle]:
    frame_f0 = utterance.f0_hz.unsqueeze(0).to(torch.float32)
    frame_voiced = utterance.voiced.unsqueeze(0).to(torch.float32)
    sample_f0 = fixed_linear_frame_to_sample(frame_f0, hop_length=HOP_LENGTH).squeeze(0)
    sample_voiced = fixed_linear_frame_to_sample(frame_voiced, hop_length=HOP_LENGTH).squeeze(0)
    if int(sample_f0.numel()) != int(residual.numel()):
        raise RuntimeError("pitch interpolation and residual lengths differ")

    phase_increment = torch.where(
        (sample_f0 > 0.0) & (sample_voiced >= 0.5),
        sample_f0 / float(SAMPLE_RATE),
        torch.zeros_like(sample_f0),
    )
    phase = torch.cumsum(phase_increment, dim=0)
    previous = torch.cat((torch.zeros(1, dtype=phase.dtype), phase[:-1]), dim=0)
    crossings = torch.nonzero(torch.floor(phase) > torch.floor(previous), as_tuple=False).flatten()
    if crossings.numel() < 2:
        return

    for left_tensor, right_tensor in zip(crossings[:-1], crossings[1:]):
        left = int(left_tensor)
        right = int(right_tensor)
        period = right - left
        if period < MIN_CYCLE_SAMPLES or period > MAX_CYCLE_SAMPLES:
            continue
        voiced_fraction = float(sample_voiced[left:right].mean())
        if voiced_fraction < 0.75:
            continue
        local_f0 = sample_f0[left:right]
        positive = local_f0[local_f0 > 0.0]
        if positive.numel() < max(4, period // 2):
            continue
        f0_hz = float(torch.median(positive))
        if not math.isfinite(f0_hz) or f0_hz <= 0.0:
            continue
        cycle = residual[left:right].clone().contiguous()
        if float(cycle.abs().max()) <= 1.0e-7:
            continue
        yield ResidualCycle(
            utterance_id=utterance.utterance_id,
            f0_hz=f0_hz,
            cycle=cycle,
        )


def iter_owned_residual_cycles(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> Iterator[tuple[OwnedVocoderUtterance, list[ResidualCycle], int]]:
    """Yield one utterance at a time so calibration does not retain the full corpus residual."""

    utterances = collect_owned_vocoder_utterances(
        Path(root).resolve(),
        split=split,
        max_items=max_items,
    )
    for utterance in utterances:
        residual, _, extension_frames = extract_owned_real_residual(
            utterance.waveform.cpu(),
            frame_count=int(utterance.mel_frames),
        )
        cycles = list(_pitch_synchronous_cycles(utterance, residual))
        yield utterance, cycles, extension_frames


def _cycle_tilt_db_per_octave(cycle: torch.Tensor) -> float | None:
    centered = cycle.to(torch.float64) - cycle.to(torch.float64).mean()
    if int(centered.numel()) < MIN_CYCLE_SAMPLES:
        return None
    window = torch.hann_window(int(centered.numel()), periodic=False, dtype=torch.float64)
    spectrum = torch.fft.rfft(centered * window, n=TILT_N_FFT)
    magnitude_db = 20.0 * torch.log10(spectrum.abs().clamp_min(1.0e-10))
    freqs = torch.fft.rfftfreq(TILT_N_FFT, d=1.0 / float(SAMPLE_RATE)).to(torch.float64)
    mask = (freqs >= TILT_FMIN_HZ) & (freqs <= TILT_FMAX_HZ)
    x = torch.log2(freqs[mask] / 1000.0)
    y = magnitude_db[mask]
    if x.numel() < 8:
        return None
    x_centered = x - x.mean()
    denominator = (x_centered.square()).sum()
    if float(denominator) <= 0.0:
        return None
    slope = ((x_centered * (y - y.mean())).sum() / denominator).item()
    return float(slope) if math.isfinite(float(slope)) else None


def _measure_cycle(cycle: ResidualCycle) -> dict[str, float] | None:
    signal = cycle.cycle.to(torch.float64)
    abs_signal = signal.abs()
    peak = float(abs_signal.max())
    if peak <= 1.0e-10:
        return None
    power = signal.square()
    threshold = float(power.max()) * OPEN_ENERGY_FRACTION
    open_quotient = float((power >= threshold).to(torch.float64).mean())
    peak_index = int(torch.argmax(abs_signal))
    asymmetry = peak_index / float(max(1, int(signal.numel()) - 1))
    tilt = _cycle_tilt_db_per_octave(signal)
    if tilt is None:
        return None
    rms = float(torch.sqrt(power.mean().clamp_min(1.0e-20)))
    return {
        "f0_hz": cycle.f0_hz,
        "open_quotient": open_quotient,
        "asymmetry_peak_position": asymmetry,
        "spectral_tilt_db_per_octave": tilt,
        "residual_rms": rms,
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "median": float(torch.median(tensor)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p90": float(torch.quantile(tensor, 0.90)),
    }


def _aggregate_records(records: list[dict[str, float]]) -> dict[str, object]:
    return {
        "cycle_count": len(records),
        "open_quotient": _summary([item["open_quotient"] for item in records]),
        "asymmetry_peak_position": _summary(
            [item["asymmetry_peak_position"] for item in records]
        ),
        "spectral_tilt_db_per_octave": _summary(
            [item["spectral_tilt_db_per_octave"] for item in records]
        ),
        "residual_rms": _summary([item["residual_rms"] for item in records]),
    }


def run_glottal_calibration(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_MAX_ITEMS,
    output_path: Path | None = None,
) -> dict[str, object]:
    if split != "train":
        raise ValueError("glottal identity calibration must use the owned train split")
    root = Path(root).resolve()
    output_path = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "models" / "lykenox_identity" / "calibration" / "glottal_pulse_v1.json"
    )

    records: list[dict[str, float]] = []
    provenance: list[dict[str, object]] = []
    terminal_extension_counts: list[int] = []
    for utterance, cycles, extension_frames in iter_owned_residual_cycles(
        root,
        split=split,
        max_items=max_items,
    ):
        measured = [value for cycle in cycles if (value := _measure_cycle(cycle)) is not None]
        if measured:
            records.extend(measured)
            wav_path = Path(utterance.wav_path)
            provenance.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "wav_path": str(wav_path),
                    "wav_sha256": _sha256_file(wav_path),
                    "accepted_cycle_count": len(measured),
                    "pitch_cache_version": utterance.pitch_cache_version,
                    "conditioning_contract_version": utterance.conditioning_contract_version,
                }
            )
            terminal_extension_counts.append(extension_frames)

    if not records:
        raise RuntimeError("no valid voiced residual cycles were available for calibration")

    bins: dict[int, list[dict[str, float]]] = {}
    for record in records:
        lower = int(math.floor(record["f0_hz"] / F0_BIN_HZ) * F0_BIN_HZ)
        bins.setdefault(lower, []).append(record)
    f0_bins = []
    for lower in sorted(bins):
        group = bins[lower]
        f0_bins.append(
            {
                "f0_min_hz": float(lower),
                "f0_max_hz": float(lower + F0_BIN_HZ),
                "f0_center_hz": float(lower + F0_BIN_HZ / 2.0),
                **_aggregate_records(group),
            }
        )

    artifact: dict[str, object] = {
        "status": "calibrated_from_owned_train_residual",
        "calibration_version": GLOTTAL_CALIBRATION_VERSION,
        "policy_id": POLICY_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "split": split,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "cepstral_order": CEPSTRAL_ORDER,
        "renderer_version": RENDERER_VERSION,
        "full_utterance_data_version": FULL_UTTERANCE_DATA_VERSION,
        "source_residual_method": "owned_reference_stft_divided_by_order64_oracle_minimum_phase_transfer",
        "terminal_transfer_extension_rule": "repeat_last_conditioning_transfer_for_centered_stft_terminal_frames",
        "f0_bin_width_hz": F0_BIN_HZ,
        "open_quotient_proxy": {
            "definition": "fraction_of_cycle_samples_with_squared_residual_at_least_fraction_of_cycle_peak_power",
            "peak_power_fraction": OPEN_ENERGY_FRACTION,
        },
        "asymmetry_proxy": "absolute_residual_peak_position_divided_by_cycle_length_minus_one",
        "tilt_measurement": {
            "definition": "linear_regression_of_log_magnitude_db_against_log2_frequency",
            "fmin_hz": TILT_FMIN_HZ,
            "fmax_hz": TILT_FMAX_HZ,
            "analysis_n_fft": TILT_N_FFT,
        },
        "global": _aggregate_records(records),
        "f0_bins": f0_bins,
        "utterance_count": len(provenance),
        "cycle_count": len(records),
        "terminal_transfer_extension_frames_seen": sorted(set(terminal_extension_counts)),
        "provenance": provenance,
        "owned_data_only": True,
        "third_party_model_or_checkpoint_used": False,
        "training_executed": False,
        "optimizer_created": False,
        "posthoc_output_processing_used": False,
    }
    _atomic_json(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_glottal_calibration(
                args.root,
                split=args.split,
                max_items=args.max_items,
                output_path=args.output_path,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "GLOTTAL_CALIBRATION_VERSION",
    "ResidualCycle",
    "extract_owned_real_residual",
    "iter_owned_residual_cycles",
    "run_glottal_calibration",
]
