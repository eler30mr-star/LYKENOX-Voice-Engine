"""Audit shared mel/pitch conditioning at cross-variant vocoder anomaly timestamps.

This diagnostic does not train, synthesize, render, modify checkpoints, or write WAV files. It reads
the direct generated-vs-reference report, finds anomaly timestamps shared by multiple already-rendered
source variants, then inspects the common owned conditioning contract at those exact locations.

The goal is localization only. In particular it reports:
- cached F0, voiced and periodicity trajectories around each common anomaly;
- centered waveform-frame RMS using the same 1024/256 geometry as pitch-v1;
- the top raw autocorrelation lag candidates at the anomaly frame;
- large neighboring F0 jumps and near-octave competing autocorrelation peaks;
- voiced state changes and periodicity drops around the anomaly.

Policy LYX-POL-001. Metrics may reject/localize; they cannot accept product quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    OwnedVocoderUtterance,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import HOP_LENGTH, SAMPLE_RATE
from run_vocoder_reference_comparison_v1 import _common_anomalies


DIAGNOSTIC_VERSION = "owned-vocoder-common-conditioning-anomaly-diagnostic-v1"
POLICY_ID = "LYX-POL-001"
FRAME_LENGTH = int(PITCH_CONFIG["frame_length"])
MIN_F0_HZ = float(PITCH_CONFIG["min_f0_hz"])
MAX_F0_HZ = float(PITCH_CONFIG["max_f0_hz"])
CONTEXT_RADIUS = 8
TOP_LAGS = 6
MIN_SHARED_CANDIDATES = 4


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _centered_frames(waveform: torch.Tensor) -> torch.Tensor:
    half = FRAME_LENGTH // 2
    padded = F.pad(waveform.view(1, 1, -1), (half, half), mode="reflect")[0, 0]
    frames = padded.unfold(0, FRAME_LENGTH, HOP_LENGTH).to(torch.float32)
    return frames


def _windowed_zero_mean(frame: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(FRAME_LENGTH, dtype=frame.dtype)
    value = frame * window
    return value - value.mean()


def _frame_rms(frame: torch.Tensor) -> float:
    value = _windowed_zero_mean(frame)
    return float(torch.sqrt(value.square().mean().clamp_min(1.0e-12)))


def _lag_bounds() -> tuple[int, int]:
    min_lag = max(1, int(SAMPLE_RATE / MAX_F0_HZ))
    max_lag = min(FRAME_LENGTH - 2, int(SAMPLE_RATE / MIN_F0_HZ))
    return min_lag, max_lag


def _top_autocorrelation_candidates(frame: torch.Tensor) -> list[dict[str, float | int]]:
    value = _windowed_zero_mean(frame)
    spectrum = torch.fft.rfft(value, n=FRAME_LENGTH * 2)
    autocorrelation = torch.fft.irfft(spectrum * spectrum.conj(), n=FRAME_LENGTH * 2).real[:FRAME_LENGTH]
    normalized = autocorrelation / autocorrelation[0].clamp_min(1.0e-8)
    min_lag, max_lag = _lag_bounds()
    candidates = normalized[min_lag : max_lag + 1]
    k = min(TOP_LAGS, int(candidates.numel()))
    values, indices = torch.topk(candidates, k=k)
    rows: list[dict[str, float | int]] = []
    for rank, (score, relative) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        lag = int(relative) + min_lag
        rows.append(
            {
                "rank": rank,
                "lag_samples": lag,
                "f0_hz": SAMPLE_RATE / float(lag),
                "normalized_autocorrelation": float(score),
            }
        )
    return rows


def _near_octave_competitor(candidates: list[dict[str, float | int]]) -> dict[str, object] | None:
    if len(candidates) < 2:
        return None
    primary = candidates[0]
    primary_f0 = float(primary["f0_hz"])
    primary_score = float(primary["normalized_autocorrelation"])
    best: dict[str, object] | None = None
    for candidate in candidates[1:]:
        f0 = float(candidate["f0_hz"])
        score = float(candidate["normalized_autocorrelation"])
        ratio = max(primary_f0, f0) / max(min(primary_f0, f0), 1.0e-9)
        octave_distance = abs(math.log2(max(ratio, 1.0e-9)) - 1.0)
        if octave_distance <= 0.12:
            row = {
                "primary_f0_hz": primary_f0,
                "competitor_f0_hz": f0,
                "frequency_ratio": ratio,
                "primary_periodicity": primary_score,
                "competitor_periodicity": score,
                "periodicity_gap": primary_score - score,
            }
            if best is None or float(row["periodicity_gap"]) < float(best["periodicity_gap"]):
                best = row
    return best


def _neighbor_f0_jump(context: list[dict[str, float | int]]) -> dict[str, object]:
    jumps: list[dict[str, float | int]] = []
    for left, right in zip(context[:-1], context[1:]):
        if float(left["voiced"]) < 0.5 or float(right["voiced"]) < 0.5:
            continue
        a = float(left["f0_hz"])
        b = float(right["f0_hz"])
        if a <= 0.0 or b <= 0.0:
            continue
        ratio = max(a, b) / min(a, b)
        jumps.append(
            {
                "left_frame": int(left["frame_index"]),
                "right_frame": int(right["frame_index"]),
                "left_f0_hz": a,
                "right_f0_hz": b,
                "ratio": ratio,
                "octaves": abs(math.log2(ratio)),
            }
        )
    if not jumps:
        return {"max_ratio": 1.0, "max_octaves": 0.0, "largest_jump": None}
    largest = max(jumps, key=lambda row: float(row["ratio"]))
    return {
        "max_ratio": float(largest["ratio"]),
        "max_octaves": float(largest["octaves"]),
        "largest_jump": largest,
    }


def _conditioning_context(utterance: OwnedVocoderUtterance, frames: torch.Tensor, center: int) -> list[dict[str, float | int]]:
    start = max(0, center - CONTEXT_RADIUS)
    stop = min(int(utterance.mel_frames), center + CONTEXT_RADIUS + 1, int(frames.shape[0]))
    utterance_peak_rms = max(_frame_rms(frame) for frame in frames[: int(utterance.mel_frames)])
    result: list[dict[str, float | int]] = []
    for index in range(start, stop):
        rms = _frame_rms(frames[index])
        result.append(
            {
                "frame_index": index,
                "time_seconds": index * HOP_LENGTH / float(SAMPLE_RATE),
                "f0_hz": float(utterance.f0_hz[index]),
                "voiced": float(utterance.voiced[index]),
                "periodicity": float(utterance.periodicity[index]),
                "windowed_rms": rms,
                "rms_fraction_of_utterance_peak": rms / max(utterance_peak_rms, 1.0e-12),
            }
        )
    return result


def _analyze_event(
    utterance: OwnedVocoderUtterance,
    frames: torch.Tensor,
    common_row: dict[str, Any],
) -> dict[str, object]:
    requested_time = float(common_row["time_seconds"])
    center = min(
        int(utterance.mel_frames) - 1,
        max(0, int(round(requested_time * SAMPLE_RATE / float(HOP_LENGTH)))),
    )
    context = _conditioning_context(utterance, frames, center)
    top_candidates = _top_autocorrelation_candidates(frames[center])
    center_row = next(row for row in context if int(row["frame_index"]) == center)
    voiced_changes = 0
    for left, right in zip(context[:-1], context[1:]):
        if float(left["voiced"]) != float(right["voiced"]):
            voiced_changes += 1
    periodicities = [float(row["periodicity"]) for row in context]
    center_position = [int(row["frame_index"]) for row in context].index(center)
    neighbor_periodicity = [
        periodicities[index]
        for index in range(len(periodicities))
        if index != center_position
    ]
    neighbor_mean = sum(neighbor_periodicity) / max(len(neighbor_periodicity), 1)
    return {
        "utterance_id": utterance.utterance_id,
        "requested_time_seconds": requested_time,
        "analyzed_frame_index": center,
        "analyzed_time_seconds": center * HOP_LENGTH / float(SAMPLE_RATE),
        "shared_candidate_count": int(common_row["candidate_count"]),
        "shared_mean_tonal_prominence_excess_db": float(common_row["mean_tonal_prominence_excess_db"]),
        "shared_mean_air_band_excess_db": float(common_row["mean_air_band_excess_db"]),
        "cached_at_event": {
            "f0_hz": float(center_row["f0_hz"]),
            "voiced": float(center_row["voiced"]),
            "periodicity": float(center_row["periodicity"]),
            "windowed_rms": float(center_row["windowed_rms"]),
            "rms_fraction_of_utterance_peak": float(center_row["rms_fraction_of_utterance_peak"]),
        },
        "raw_autocorrelation_top_candidates": top_candidates,
        "raw_autocorrelation_near_octave_competitor": _near_octave_competitor(top_candidates),
        "neighbor_f0_jump": _neighbor_f0_jump(context),
        "voiced_state_changes_in_context": voiced_changes,
        "periodicity_context_min": min(periodicities),
        "periodicity_context_max": max(periodicities),
        "periodicity_event_minus_neighbor_mean": float(center_row["periodicity"]) - neighbor_mean,
        "conditioning_context": context,
    }


def diagnose(root: Path, *, max_events_per_utterance: int = 4) -> dict[str, object]:
    root = Path(root).resolve()
    report_path = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "generated_vs_reference_diagnostic_v1"
        / "generated_vs_reference_report.json"
    )
    if not report_path.exists():
        raise FileNotFoundError(str(report_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparisons = list(report.get("comparisons", []))
    common = _common_anomalies(comparisons)

    val = collect_owned_vocoder_utterances(root, "val", max_items=3)
    by_id = {item.utterance_id: item for item in val}
    events: list[dict[str, object]] = []
    for utterance_id, rows in common.items():
        utterance = by_id.get(utterance_id)
        if utterance is None:
            continue
        frames = _centered_frames(utterance.waveform)
        selected = [row for row in rows if int(row["candidate_count"]) >= MIN_SHARED_CANDIDATES]
        for row in selected[:max_events_per_utterance]:
            events.append(_analyze_event(utterance, frames, row))

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "generated_vs_reference_diagnostic_v1"
    )
    json_path = output_dir / "common_conditioning_anomaly_report.json"
    csv_path = output_dir / "common_conditioning_anomaly_summary.csv"
    payload: dict[str, object] = {
        "status": "common_conditioning_anomaly_diagnostic_complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "audio_generated": False,
        "training_executed": False,
        "model_inference_executed": False,
        "checkpoint_written": False,
        "source_report": str(report_path),
        "pitch_frame_length": FRAME_LENGTH,
        "pitch_hop_length": HOP_LENGTH,
        "pitch_min_f0_hz": MIN_F0_HZ,
        "pitch_max_f0_hz": MAX_F0_HZ,
        "event_count": len(events),
        "events": events,
        "metrics_can_accept_product_quality": False,
        "next_action": "use shared anomaly conditioning evidence to accept or reject pitch-conditioning root-cause hypothesis",
    }
    _atomic_json(json_path, payload)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "utterance_id",
            "requested_time_seconds",
            "shared_candidate_count",
            "shared_mean_tonal_prominence_excess_db",
            "shared_mean_air_band_excess_db",
            "cached_f0_hz",
            "cached_voiced",
            "cached_periodicity",
            "rms_fraction_of_utterance_peak",
            "max_neighbor_f0_jump_ratio",
            "max_neighbor_f0_jump_octaves",
            "voiced_state_changes_in_context",
            "near_octave_competitor_f0_hz",
            "near_octave_periodicity_gap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            cached = event["cached_at_event"]
            jump = event["neighbor_f0_jump"]
            octave = event["raw_autocorrelation_near_octave_competitor"]
            writer.writerow(
                {
                    "utterance_id": event["utterance_id"],
                    "requested_time_seconds": event["requested_time_seconds"],
                    "shared_candidate_count": event["shared_candidate_count"],
                    "shared_mean_tonal_prominence_excess_db": event["shared_mean_tonal_prominence_excess_db"],
                    "shared_mean_air_band_excess_db": event["shared_mean_air_band_excess_db"],
                    "cached_f0_hz": cached["f0_hz"],
                    "cached_voiced": cached["voiced"],
                    "cached_periodicity": cached["periodicity"],
                    "rms_fraction_of_utterance_peak": cached["rms_fraction_of_utterance_peak"],
                    "max_neighbor_f0_jump_ratio": jump["max_ratio"],
                    "max_neighbor_f0_jump_octaves": jump["max_octaves"],
                    "voiced_state_changes_in_context": event["voiced_state_changes_in_context"],
                    "near_octave_competitor_f0_hz": None if octave is None else octave["competitor_f0_hz"],
                    "near_octave_periodicity_gap": None if octave is None else octave["periodicity_gap"],
                }
            )

    # Compact stdout: enough to decide where to inspect without dumping frame trajectories.
    compact_events = []
    for event in events:
        cached = event["cached_at_event"]
        jump = event["neighbor_f0_jump"]
        octave = event["raw_autocorrelation_near_octave_competitor"]
        compact_events.append(
            {
                "utterance_id": event["utterance_id"],
                "time_seconds": event["requested_time_seconds"],
                "candidate_count": event["shared_candidate_count"],
                "tonal_excess_db": event["shared_mean_tonal_prominence_excess_db"],
                "air_excess_db": event["shared_mean_air_band_excess_db"],
                "cached_f0_hz": cached["f0_hz"],
                "voiced": cached["voiced"],
                "periodicity": cached["periodicity"],
                "rms_fraction_of_peak": cached["rms_fraction_of_utterance_peak"],
                "max_neighbor_f0_jump_ratio": jump["max_ratio"],
                "voiced_state_changes": event["voiced_state_changes_in_context"],
                "near_octave_competitor": octave,
            }
        )
    return {
        "status": payload["status"],
        "audio_generated": False,
        "training_executed": False,
        "model_inference_executed": False,
        "event_count": len(events),
        "events": compact_events,
        "json_report": str(json_path),
        "csv_summary": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-events-per-utterance", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.root, max_events_per_utterance=args.max_events_per_utterance), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
