"""Direct, non-generative comparison of held-out vocoder WAVs against their references.

This diagnostic exists because repeated listening gates showed that source variants could change
robotization, nasal coloration, wind-like noise and terminal chirps while aggregate training losses
were not sufficiently explanatory.  It does NOT synthesize audio and cannot accept product quality.

For every held-out final-output WAV found under ``models/lykenox_identity/evaluation`` it compares the
already-written waveform against the matching ``__reference.wav`` and uses the matching
``__identity_roundtrip_ceiling.wav`` as a calibration floor.  The ceiling is important: a metric that
also reports a large difference between the known-clean ceiling and reference is not trusted as a
strong locator.

Outputs:
- JSON with per-file metrics, top anomaly timestamps, terminal-region diagnostics and aggregate ranks;
- CSV with one row per compared candidate;
- no WAVs, no checkpoint writes, no model inference, no post-hoc processing.

Policy: LYX-POL-001. Metrics may reject or localize defects; they may not accept product quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


DIAGNOSTIC_VERSION = "owned-vocoder-generated-vs-reference-diagnostic-v1"
POLICY_ID = "LYX-POL-001"
EXPECTED_SAMPLE_RATE = 24000
N_FFT = 1024
HOP = 256
EPSILON = 1.0e-8

# These are internal/residual artifacts, not final speech candidates.
EXCLUDED_SUFFIX_MARKERS = (
    "__reference.wav",
    "__identity_roundtrip_ceiling.wav",
    "__continuous_predicted_residual",
    "__predicted_residual",
    "__hybrid_residual",
    "__unified_residual.wav",
    "__periodic_coordinate.wav",
    "__aperiodic_coordinate.wav",
    "__voiced_cycles.wav",
)

BANDS_HZ: tuple[tuple[str, float, float], ...] = (
    ("low", 80.0, 300.0),
    ("body", 300.0, 1200.0),
    ("presence", 1200.0, 4000.0),
    ("high", 4000.0, 8000.0),
    ("air", 8000.0, 11900.0),
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _load_mono(path: Path) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data).mean(dim=1).to(torch.float32).contiguous()
    if waveform.numel() < N_FFT:
        raise RuntimeError(f"audio too short for comparison: {path}")
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError(f"non-finite audio: {path}")
    return waveform, int(sample_rate)


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.to(torch.float64).square().mean().clamp_min(1.0e-30)))


def _db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1.0e-12) / max(denominator, 1.0e-12))


def _stft_magnitude(waveform: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(N_FFT, dtype=waveform.dtype)
    return torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    ).abs().transpose(0, 1).contiguous()  # [frames, bins]


def _band_power(magnitude: torch.Tensor, sample_rate: int, low: float, high: float) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(N_FFT, d=1.0 / float(sample_rate))
    mask = (freqs >= float(low)) & (freqs < float(high))
    if int(mask.sum()) < 1:
        raise RuntimeError(f"empty diagnostic band {low}-{high} Hz")
    return magnitude[:, mask].square().mean(dim=-1).clamp_min(EPSILON * EPSILON)


def _frame_rms_db(magnitude: torch.Tensor) -> torch.Tensor:
    power = magnitude.square().mean(dim=-1).clamp_min(EPSILON * EPSILON)
    return 10.0 * torch.log10(power)


def _spectral_flatness(magnitude: torch.Tensor, sample_rate: int, low: float, high: float) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(N_FFT, d=1.0 / float(sample_rate))
    mask = (freqs >= float(low)) & (freqs < float(high))
    values = magnitude[:, mask].clamp_min(EPSILON)
    geometric = torch.exp(torch.log(values).mean(dim=-1))
    arithmetic = values.mean(dim=-1).clamp_min(EPSILON)
    return (geometric / arithmetic).clamp(0.0, 1.0)


def _tonal_prominence_db(magnitude: torch.Tensor, sample_rate: int, low: float = 2500.0) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(N_FFT, d=1.0 / float(sample_rate))
    mask = (freqs >= float(low)) & (freqs < sample_rate / 2.0 - 50.0)
    values = magnitude[:, mask].clamp_min(EPSILON)
    peak = values.amax(dim=-1)
    median = values.median(dim=-1).values.clamp_min(EPSILON)
    return 20.0 * torch.log10(peak / median)


def _terminal_frame_indices(reference_rms_db: torch.Tensor) -> list[int]:
    """Return speech-energy falling edges followed by a real low-energy interval."""
    peak = float(reference_rms_db.max())
    threshold = max(peak - 34.0, -58.0)
    active = reference_rms_db >= threshold
    minimum_quiet = 6  # about 64 ms at 24 kHz / hop 256
    terminals: list[int] = []
    for index in range(1, int(active.numel()) - minimum_quiet):
        if bool(active[index - 1]) and not bool(active[index]):
            if not bool(active[index : index + minimum_quiet].any()):
                terminals.append(index)
    # Collapse very close edges so one phrase ending is not counted multiple times.
    result: list[int] = []
    for index in terminals:
        if not result or index - result[-1] >= 8:
            result.append(index)
    return result


def _pair_metrics(candidate: torch.Tensor, reference: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    samples = min(int(candidate.numel()), int(reference.numel()))
    cand = candidate[:samples]
    ref = reference[:samples]
    cand_mag = _stft_magnitude(cand)
    ref_mag = _stft_magnitude(ref)
    frames = min(int(cand_mag.shape[0]), int(ref_mag.shape[0]))
    cand_mag = cand_mag[:frames]
    ref_mag = ref_mag[:frames]

    cand_rms = _rms(cand)
    ref_rms = _rms(ref)
    log_cand = 20.0 * torch.log10(cand_mag.clamp_min(EPSILON))
    log_ref = 20.0 * torch.log10(ref_mag.clamp_min(EPSILON))
    frame_log_mae = (log_cand - log_ref).abs().mean(dim=-1)
    spectral_convergence = float(
        torch.linalg.vector_norm(cand_mag - ref_mag)
        / torch.linalg.vector_norm(ref_mag).clamp_min(EPSILON)
    )

    cand_frame_rms = _frame_rms_db(cand_mag)
    ref_frame_rms = _frame_rms_db(ref_mag)
    frame_rms_delta = cand_frame_rms - ref_frame_rms

    band_delta_db: dict[str, float] = {}
    band_frame_delta: dict[str, torch.Tensor] = {}
    for name, low, high in BANDS_HZ:
        cp = _band_power(cand_mag, sample_rate, low, high)
        rp = _band_power(ref_mag, sample_rate, low, high)
        delta = 10.0 * torch.log10(cp / rp)
        band_delta_db[name] = float(delta.mean())
        band_frame_delta[name] = delta

    cand_flatness = _spectral_flatness(cand_mag, sample_rate, 4000.0, 11900.0)
    ref_flatness = _spectral_flatness(ref_mag, sample_rate, 4000.0, 11900.0)
    flatness_delta = cand_flatness - ref_flatness

    cand_tonal = _tonal_prominence_db(cand_mag, sample_rate)
    ref_tonal = _tonal_prominence_db(ref_mag, sample_rate)
    tonal_delta = cand_tonal - ref_tonal

    # Locator only. It combines independent deviations; it is not an acceptance score.
    anomaly_score = (
        frame_log_mae
        + 0.35 * frame_rms_delta.abs()
        + 0.40 * tonal_delta.clamp_min(0.0)
        + 0.30 * band_frame_delta["high"].clamp_min(0.0)
        + 0.20 * band_frame_delta["air"].clamp_min(0.0)
        + 8.0 * flatness_delta.clamp_min(0.0)
    )
    top_count = min(10, int(anomaly_score.numel()))
    top_values, top_indices = torch.topk(anomaly_score, k=top_count)
    top_anomalies: list[dict[str, float]] = []
    for value, index_tensor in zip(top_values.tolist(), top_indices.tolist()):
        index = int(index_tensor)
        top_anomalies.append(
            {
                "time_seconds": index * HOP / float(sample_rate),
                "locator_score": float(value),
                "log_spectral_mae_db": float(frame_log_mae[index]),
                "rms_delta_db": float(frame_rms_delta[index]),
                "tonal_prominence_excess_db": float(tonal_delta[index]),
                "high_band_excess_db": float(band_frame_delta["high"][index]),
                "air_band_excess_db": float(band_frame_delta["air"][index]),
                "high_band_flatness_delta": float(flatness_delta[index]),
            }
        )

    terminal_regions: list[dict[str, float]] = []
    for edge in _terminal_frame_indices(ref_frame_rms):
        start = max(0, edge - 3)
        stop = min(frames, edge + 10)
        terminal_regions.append(
            {
                "time_seconds": edge * HOP / float(sample_rate),
                "window_start_seconds": start * HOP / float(sample_rate),
                "window_end_seconds": stop * HOP / float(sample_rate),
                "mean_log_spectral_mae_db": float(frame_log_mae[start:stop].mean()),
                "mean_rms_delta_db": float(frame_rms_delta[start:stop].mean()),
                "mean_tonal_prominence_excess_db": float(tonal_delta[start:stop].mean()),
                "mean_high_band_excess_db": float(band_frame_delta["high"][start:stop].mean()),
                "mean_air_band_excess_db": float(band_frame_delta["air"][start:stop].mean()),
                "mean_high_band_flatness_delta": float(flatness_delta[start:stop].mean()),
            }
        )

    return {
        "compared_samples": samples,
        "duration_seconds": samples / float(sample_rate),
        "candidate_rms": cand_rms,
        "reference_rms": ref_rms,
        "rms_ratio": cand_rms / max(ref_rms, 1.0e-12),
        "rms_delta_db": _db_ratio(cand_rms, ref_rms),
        "log_spectral_mae_db": float(frame_log_mae.mean()),
        "log_spectral_mae_db_p95": float(torch.quantile(frame_log_mae, 0.95)),
        "spectral_convergence": spectral_convergence,
        "band_energy_delta_db": band_delta_db,
        "high_band_flatness_delta_mean": float(flatness_delta.mean()),
        "high_band_flatness_delta_p95": float(torch.quantile(flatness_delta, 0.95)),
        "tonal_prominence_excess_db_mean": float(tonal_delta.mean()),
        "tonal_prominence_excess_db_p95": float(torch.quantile(tonal_delta, 0.95)),
        "terminal_regions": terminal_regions,
        "top_anomaly_timestamps": top_anomalies,
    }


def _utterance_stem(path: Path) -> str | None:
    name = path.name
    if "__" not in name or not name.endswith(".wav"):
        return None
    return name.split("__", 1)[0]


def _candidate_wavs(directory: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(directory.glob("*.wav")):
        lowered = path.name.lower()
        if any(marker in lowered for marker in EXCLUDED_SUFFIX_MARKERS):
            continue
        if "__" not in path.name:
            continue
        result.append(path)
    return result


def _comparison_key(path: Path) -> str:
    suffix = path.name.split("__", 1)[1].removesuffix(".wav")
    return f"{path.parent.name}/{suffix}"


def diagnose(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    evaluation_root = root / "models" / "lykenox_identity" / "evaluation"
    if not evaluation_root.exists():
        raise FileNotFoundError(str(evaluation_root))

    output_dir = evaluation_root / "generated_vs_reference_diagnostic_v1"
    comparisons: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for directory in sorted(path for path in evaluation_root.iterdir() if path.is_dir()):
        if directory == output_dir:
            continue
        by_stem_reference = {
            _utterance_stem(path): path
            for path in directory.glob("*__reference.wav")
            if _utterance_stem(path) is not None
        }
        by_stem_ceiling = {
            _utterance_stem(path): path
            for path in directory.glob("*__identity_roundtrip_ceiling.wav")
            if _utterance_stem(path) is not None
        }
        if not by_stem_reference:
            continue
        ceiling_cache: dict[str, dict[str, Any]] = {}
        for candidate_path in _candidate_wavs(directory):
            stem = _utterance_stem(candidate_path)
            if stem is None or stem not in by_stem_reference:
                continue
            reference_path = by_stem_reference[stem]
            try:
                candidate, candidate_sr = _load_mono(candidate_path)
                reference, reference_sr = _load_mono(reference_path)
            except Exception as exc:  # diagnostic should report bad artifacts, not hide them
                skipped.append({"path": str(candidate_path), "reason": repr(exc)})
                continue
            if candidate_sr != reference_sr:
                skipped.append({
                    "path": str(candidate_path),
                    "reason": f"sample_rate_mismatch candidate={candidate_sr} reference={reference_sr}",
                })
                continue
            if candidate_sr != EXPECTED_SAMPLE_RATE:
                skipped.append({
                    "path": str(candidate_path),
                    "reason": f"unexpected_sample_rate={candidate_sr}",
                })
                continue

            ceiling_metrics: dict[str, Any] | None = None
            ceiling_path = by_stem_ceiling.get(stem)
            if ceiling_path is not None:
                if stem not in ceiling_cache:
                    ceiling, ceiling_sr = _load_mono(ceiling_path)
                    if ceiling_sr != reference_sr:
                        raise RuntimeError(f"ceiling sample-rate mismatch for {stem}")
                    ceiling_cache[stem] = _pair_metrics(ceiling, reference, reference_sr)
                ceiling_metrics = ceiling_cache[stem]

            metrics = _pair_metrics(candidate, reference, reference_sr)
            row: dict[str, Any] = {
                "evaluation_dir": directory.name,
                "utterance_id": stem,
                "candidate": candidate_path.name,
                "candidate_key": _comparison_key(candidate_path),
                "candidate_path": str(candidate_path),
                "reference_path": str(reference_path),
                "ceiling_path": str(ceiling_path) if ceiling_path is not None else None,
                "candidate_samples": int(candidate.numel()),
                "reference_samples": int(reference.numel()),
                "length_match": int(candidate.numel()) == int(reference.numel()),
                **metrics,
            }
            if ceiling_metrics is not None:
                row["ceiling_calibration"] = ceiling_metrics
                row["excess_over_ceiling"] = {
                    "log_spectral_mae_db": metrics["log_spectral_mae_db"] - ceiling_metrics["log_spectral_mae_db"],
                    "spectral_convergence": metrics["spectral_convergence"] - ceiling_metrics["spectral_convergence"],
                    "absolute_rms_delta_db": abs(metrics["rms_delta_db"]) - abs(ceiling_metrics["rms_delta_db"]),
                    "tonal_prominence_p95_db": metrics["tonal_prominence_excess_db_p95"] - ceiling_metrics["tonal_prominence_excess_db_p95"],
                    "high_band_flatness_p95": metrics["high_band_flatness_delta_p95"] - ceiling_metrics["high_band_flatness_delta_p95"],
                }
            comparisons.append(row)

    if not comparisons:
        raise RuntimeError("no comparable final-output WAVs found under evaluation")

    # Aggregate the same candidate suffix/architecture across held-out utterances.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in comparisons:
        grouped.setdefault(str(item["candidate_key"]), []).append(item)
    rankings: list[dict[str, Any]] = []
    for key, items in grouped.items():
        log_excess = [
            float(item.get("excess_over_ceiling", {}).get("log_spectral_mae_db", item["log_spectral_mae_db"]))
            for item in items
        ]
        rms_abs = [abs(float(item["rms_delta_db"])) for item in items]
        tonal_p95 = [float(item["tonal_prominence_excess_db_p95"]) for item in items]
        flatness_p95 = [float(item["high_band_flatness_delta_p95"]) for item in items]
        high_delta = [float(item["band_energy_delta_db"]["high"]) for item in items]
        # Ranking is diagnostic only and intentionally transparent.
        diagnostic_rank_score = (
            sum(log_excess) / len(log_excess)
            + 0.30 * (sum(rms_abs) / len(rms_abs))
            + 0.20 * max(0.0, sum(tonal_p95) / len(tonal_p95))
            + 3.0 * max(0.0, sum(flatness_p95) / len(flatness_p95))
            + 0.10 * abs(sum(high_delta) / len(high_delta))
        )
        rankings.append(
            {
                "candidate_key": key,
                "utterance_count": len(items),
                "diagnostic_rank_score_lower_is_closer": diagnostic_rank_score,
                "mean_log_spectral_mae_excess_over_ceiling_db": sum(log_excess) / len(log_excess),
                "mean_absolute_rms_delta_db": sum(rms_abs) / len(rms_abs),
                "mean_tonal_prominence_excess_p95_db": sum(tonal_p95) / len(tonal_p95),
                "mean_high_band_flatness_delta_p95": sum(flatness_p95) / len(flatness_p95),
                "mean_high_band_energy_delta_db": sum(high_delta) / len(high_delta),
            }
        )
    rankings.sort(key=lambda item: float(item["diagnostic_rank_score_lower_is_closer"]))

    report: dict[str, object] = {
        "status": "generated_vs_reference_diagnostic_complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "evaluation_root": str(evaluation_root),
        "comparison_count": len(comparisons),
        "candidate_group_count": len(rankings),
        "audio_generated": False,
        "model_inference_executed": False,
        "checkpoint_written": False,
        "posthoc_gain_normalization_used": False,
        "metrics_can_accept_product_quality": False,
        "identity_roundtrip_ceiling_used_as_metric_floor": True,
        "ranking_semantics": "lower diagnostic score means closer to reference on these diagnostics only; listening remains required for acceptance",
        "rankings": rankings,
        "comparisons": comparisons,
        "skipped": skipped,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "generated_vs_reference_report.json", report)

    csv_path = output_dir / "generated_vs_reference_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "evaluation_dir",
            "utterance_id",
            "candidate",
            "rms_ratio",
            "rms_delta_db",
            "log_spectral_mae_db",
            "spectral_convergence",
            "high_band_energy_delta_db",
            "air_band_energy_delta_db",
            "tonal_prominence_excess_db_p95",
            "high_band_flatness_delta_p95",
            "log_spectral_mae_excess_over_ceiling_db",
            "length_match",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in comparisons:
            excess = item.get("excess_over_ceiling", {})
            writer.writerow(
                {
                    "evaluation_dir": item["evaluation_dir"],
                    "utterance_id": item["utterance_id"],
                    "candidate": item["candidate"],
                    "rms_ratio": item["rms_ratio"],
                    "rms_delta_db": item["rms_delta_db"],
                    "log_spectral_mae_db": item["log_spectral_mae_db"],
                    "spectral_convergence": item["spectral_convergence"],
                    "high_band_energy_delta_db": item["band_energy_delta_db"]["high"],
                    "air_band_energy_delta_db": item["band_energy_delta_db"]["air"],
                    "tonal_prominence_excess_db_p95": item["tonal_prominence_excess_db_p95"],
                    "high_band_flatness_delta_p95": item["high_band_flatness_delta_p95"],
                    "log_spectral_mae_excess_over_ceiling_db": excess.get("log_spectral_mae_db"),
                    "length_match": item["length_match"],
                }
            )

    compact = {
        "status": report["status"],
        "comparison_count": len(comparisons),
        "json_report": str(output_dir / "generated_vs_reference_report.json"),
        "csv_summary": str(csv_path),
        "top_rankings": rankings[:10],
    }
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(diagnose(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
