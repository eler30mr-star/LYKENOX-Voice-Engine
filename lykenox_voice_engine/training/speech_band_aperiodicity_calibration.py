"""Owned deterministic band-aperiodicity calibration from LYKENOX real residual cycles.

The calibration reuses the same real-residual extraction and pitch-synchronous cycle geometry
as ``speech_glottal_calibration``.  No learned model, optimizer, checkpoint, external voice
component, or remote service participates.  The artifact is derived from owned train data only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from lykenox_voice_engine.training.speech_glottal_calibration import (
    DEFAULT_MAX_ITEMS,
    F0_BIN_HZ,
    GLOTTAL_CALIBRATION_VERSION,
    ResidualCycle,
    _sha256_file,
    iter_owned_residual_cycles,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    FULL_UTTERANCE_DATA_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
)


BAND_APERIODICITY_CALIBRATION_VERSION = "owned-band-aperiodicity-calibration-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_SPLIT = "train"
ANALYSIS_N_FFT = 2048
HARMONIC_NEIGHBORHOOD_F0_FRACTION = 0.12
BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _band_key(low_hz: float, high_hz: float) -> str:
    return f"{int(low_hz)}_{int(high_hz)}_hz"


def _measure_cycle_band_aperiodicity(cycle: ResidualCycle) -> dict[str, float] | None:
    signal = cycle.cycle.to(torch.float64)
    if int(signal.numel()) < 24 or cycle.f0_hz <= 0.0:
        return None
    signal = signal - signal.mean()
    window = torch.hann_window(int(signal.numel()), periodic=False, dtype=torch.float64)
    spectrum = torch.fft.rfft(signal * window, n=ANALYSIS_N_FFT)
    power = spectrum.abs().square().clamp_min(1.0e-20)
    freqs = torch.fft.rfftfreq(ANALYSIS_N_FFT, d=1.0 / float(SAMPLE_RATE)).to(torch.float64)
    bin_hz = float(SAMPLE_RATE) / float(ANALYSIS_N_FFT)
    harmonic_half_width_hz = max(1.5 * bin_hz, cycle.f0_hz * HARMONIC_NEIGHBORHOOD_F0_FRACTION)

    values: dict[str, float] = {"f0_hz": cycle.f0_hz}
    for low_hz, high_hz in BANDS_HZ:
        band_mask = (freqs >= low_hz) & (freqs < high_hz)
        if low_hz == 0.0:
            band_mask &= freqs > 0.0
        if int(band_mask.sum()) < 8:
            return None

        harmonic_mask = torch.zeros_like(band_mask)
        maximum_harmonic = int(math.floor((high_hz - 1.0e-9) / cycle.f0_hz))
        minimum_harmonic = max(1, int(math.ceil(max(low_hz, 1.0) / cycle.f0_hz)))
        for harmonic_index in range(minimum_harmonic, maximum_harmonic + 1):
            center_hz = harmonic_index * cycle.f0_hz
            harmonic_mask |= (freqs >= center_hz - harmonic_half_width_hz) & (
                freqs <= center_hz + harmonic_half_width_hz
            )
        harmonic_mask &= band_mask
        noise_mask = band_mask & ~harmonic_mask
        if int(harmonic_mask.sum()) < 1 or int(noise_mask.sum()) < 3:
            return None

        harmonic_power = power[harmonic_mask].mean()
        noise_power = power[noise_mask].mean()
        aperiodicity = noise_power / (harmonic_power + noise_power).clamp_min(1.0e-20)
        values[_band_key(low_hz, high_hz)] = float(aperiodicity.clamp(0.0, 1.0))
    return values


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


def _aggregate(records: list[dict[str, float]]) -> dict[str, object]:
    bands: dict[str, object] = {}
    for low_hz, high_hz in BANDS_HZ:
        key = _band_key(low_hz, high_hz)
        bands[key] = _summary([record[key] for record in records])
    return {"cycle_count": len(records), "bands": bands}


def run_band_aperiodicity_calibration(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_MAX_ITEMS,
    output_path: Path | None = None,
) -> dict[str, object]:
    if split != "train":
        raise ValueError("band aperiodicity identity calibration must use the owned train split")
    root = Path(root).resolve()
    output_path = (
        Path(output_path).resolve()
        if output_path is not None
        else root
        / "models"
        / "lykenox_identity"
        / "calibration"
        / "band_aperiodicity_v1.json"
    )

    records: list[dict[str, float]] = []
    provenance: list[dict[str, object]] = []
    for utterance, cycles, _ in iter_owned_residual_cycles(
        root,
        split=split,
        max_items=max_items,
    ):
        measured = [
            value
            for cycle in cycles
            if (value := _measure_cycle_band_aperiodicity(cycle)) is not None
        ]
        if not measured:
            continue
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

    if not records:
        raise RuntimeError("no valid voiced residual cycles were available for aperiodicity calibration")

    grouped: dict[int, list[dict[str, float]]] = {}
    for record in records:
        lower = int(math.floor(record["f0_hz"] / F0_BIN_HZ) * F0_BIN_HZ)
        grouped.setdefault(lower, []).append(record)
    f0_bins = []
    for lower in sorted(grouped):
        f0_bins.append(
            {
                "f0_min_hz": float(lower),
                "f0_max_hz": float(lower + F0_BIN_HZ),
                "f0_center_hz": float(lower + F0_BIN_HZ / 2.0),
                **_aggregate(grouped[lower]),
            }
        )

    artifact: dict[str, object] = {
        "status": "calibrated_from_owned_train_residual",
        "calibration_version": BAND_APERIODICITY_CALIBRATION_VERSION,
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
        "residual_extraction_version": GLOTTAL_CALIBRATION_VERSION,
        "analysis_n_fft": ANALYSIS_N_FFT,
        "harmonic_neighborhood_f0_fraction": HARMONIC_NEIGHBORHOOD_F0_FRACTION,
        "aperiodicity_definition": "mean_interharmonic_power_divided_by_mean_harmonic_plus_interharmonic_power",
        "bands_hz": [
            {"low_hz": low_hz, "high_hz": high_hz, "key": _band_key(low_hz, high_hz)}
            for low_hz, high_hz in BANDS_HZ
        ],
        "global": _aggregate(records),
        "f0_bin_width_hz": F0_BIN_HZ,
        "f0_bins": f0_bins,
        "utterance_count": len(provenance),
        "cycle_count": len(records),
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
            run_band_aperiodicity_calibration(
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
    "BAND_APERIODICITY_CALIBRATION_VERSION",
    "BANDS_HZ",
    "run_band_aperiodicity_calibration",
]
