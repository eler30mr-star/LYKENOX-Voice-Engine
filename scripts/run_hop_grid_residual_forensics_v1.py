"""Forensic test for hop-grid artifacts in learned 512/256 residual vectors.

No training, WAV writing, checkpoint mutation, or minimum-phase rendering occurs here. The script
reconstructs the historical Continuous Source V2 and the controlled conditioning-V2 candidate in
memory and compares their residual analysis vectors directly against the owned Step-3f target.

It tests a specific representation-level hypothesis supported by prior forensics: the learned
residual is not excessively coherent at F0, but its high-band spectral flatness collapses. Because
V2 predicts overlapping 512-sample vectors every 256 samples, adjacent predictions may disagree
about the same underlying overlap samples. That disagreement repeats at the 24 kHz / 256 = 93.75 Hz
frame grid and can create comb-like tonal structure before the fixed renderer.

Metrics localize/reject only and cannot accept product quality. Policy: LYX-POL-001.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v2 import (
    LykenoxContinuousResidualSourceV2,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import extract_pitch_conditioning_v2
from lykenox_voice_engine.training.speech_residual_codebook_v1 import _sqrt_hann, residual_analysis_vectors
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import _ola_vectors
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v2 import (
    CHECKPOINT_SCHEMA_VERSION as BASE_SCHEMA,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_v2_pitch_conditioning_v2 import (
    CHECKPOINT_SCHEMA_VERSION as CANDIDATE_SCHEMA,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    OwnedVocoderUtterance,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import HOP_LENGTH, SAMPLE_RATE

DIAGNOSTIC_VERSION = "owned-hop-grid-residual-forensics-v1"
POLICY_ID = "LYX-POL-001"
VECTOR_SAMPLES = HOP_LENGTH * 2
GRID_HZ = SAMPLE_RATE / float(HOP_LENGTH)
WINDOW_RADIUS_SECONDS = 0.30
EVENTS = {
    "speech_0024_1778f351cc1f_seg_006": (4.00,),
    "speech_0021_6cd35984e877_seg_001": (5.80,),
}


def _load_model(path: Path, schema: str, expected_conditioning: str | None = None) -> LykenoxContinuousResidualSourceV2:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("checkpoint_schema_version") != schema:
        raise RuntimeError(f"checkpoint schema mismatch: {path}")
    if expected_conditioning is not None and payload.get("conditioning_contract") != expected_conditioning:
        raise RuntimeError(f"conditioning contract mismatch: {path}")
    model = LykenoxContinuousResidualSourceV2().cpu()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def _normalized_lag_correlation(value: torch.Tensor, lag: int) -> float:
    if lag < 1 or int(value.numel()) <= lag + 8:
        return float("nan")
    a = value[:-lag].to(torch.float64)
    b = value[lag:].to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1.0e-20)
    return float((a * b).sum() / denom)


def _overlap_consistency(vectors: torch.Tensor, *, center_frame: int, radius_frames: int) -> dict[str, float | int]:
    """Compare underlying-sample estimates from adjacent 512/256 analysis vectors."""
    if vectors.ndim != 2 or vectors.shape[-1] != VECTOR_SAMPLES:
        raise ValueError("vectors must be [frames+1,512]")
    window = _sqrt_hann(dtype=vectors.dtype)
    left_w = window[HOP_LENGTH:]
    right_w = window[:HOP_LENGTH]
    # Ignore endpoints where de-windowing is ill-conditioned. The central overlap still covers most
    # of the duplicated samples and is enough to expose whether neighboring predictions agree.
    valid = (left_w >= 0.15) & (right_w >= 0.15)
    start = max(0, center_frame - radius_frames)
    stop = min(int(vectors.shape[0]) - 1, center_frame + radius_frames + 1)
    errors: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    correlations: list[torch.Tensor] = []
    for index in range(start, stop):
        left = vectors[index, HOP_LENGTH:][valid] / left_w[valid]
        right = vectors[index + 1, :HOP_LENGTH][valid] / right_w[valid]
        errors.append(left - right)
        references.append(0.5 * (left + right))
        lz = left - left.mean()
        rz = right - right.mean()
        denom = torch.sqrt(lz.square().sum() * rz.square().sum()).clamp_min(1.0e-12)
        correlations.append((lz * rz).sum() / denom)
    if not errors:
        return {
            "overlap_pair_count": 0,
            "overlap_relative_disagreement_rms": float("nan"),
            "overlap_adjacent_estimate_correlation_mean": float("nan"),
        }
    error = torch.cat(errors)
    reference = torch.cat(references)
    relative = torch.sqrt(error.square().mean().clamp_min(1.0e-20)) / torch.sqrt(
        reference.square().mean().clamp_min(1.0e-20)
    )
    return {
        "overlap_pair_count": len(errors),
        "overlap_relative_disagreement_rms": float(relative),
        "overlap_adjacent_estimate_correlation_mean": float(torch.stack(correlations).mean()),
    }


def _grid_comb_metrics(value: torch.Tensor) -> dict[str, object]:
    """Measure high-band power concentrated near harmonics of the fixed 93.75 Hz hop grid."""
    value = value.to(torch.float64)
    n = int(value.numel())
    if n < 1024:
        value = F.pad(value, (0, 1024 - n))
        n = int(value.numel())
    window = torch.hann_window(n, periodic=True, dtype=value.dtype)
    spectrum = torch.fft.rfft((value - value.mean()) * window)
    power = spectrum.abs().square().clamp_min(1.0e-30)
    freq = torch.fft.rfftfreq(n, 1.0 / float(SAMPLE_RATE))
    high = (freq >= 3500.0) & (freq <= 11000.0)
    high_power = power[high].sum().clamp_min(1.0e-30)
    grid_mask = torch.zeros_like(high)
    harmonic = 1
    centers: list[float] = []
    while harmonic * GRID_HZ <= 11000.0:
        center = harmonic * GRID_HZ
        if center >= 3500.0:
            centers.append(center)
            grid_mask |= (freq >= center - 8.0) & (freq <= center + 8.0)
        harmonic += 1
    grid_high = grid_mask & high
    fraction = power[grid_high].sum() / high_power

    # Report strongest high-band spectral lines and how close each lies to a frame-grid harmonic.
    high_indices = torch.nonzero(high, as_tuple=False).flatten()
    k = min(8, int(high_indices.numel()))
    top_power, relative = torch.topk(power[high], k=k)
    top_indices = high_indices[relative]
    peaks: list[dict[str, float]] = []
    for idx, pwr in zip(top_indices.tolist(), top_power.tolist()):
        hz = float(freq[idx])
        nearest_index = max(1, int(round(hz / GRID_HZ)))
        nearest = nearest_index * GRID_HZ
        peaks.append(
            {
                "frequency_hz": hz,
                "nearest_grid_harmonic_hz": nearest,
                "distance_to_grid_harmonic_hz": abs(hz - nearest),
                "relative_to_high_band_peak_db": 10.0 * math.log10(max(pwr, 1.0e-30) / max(float(top_power[0]), 1.0e-30)),
            }
        )
    return {
        "hop_grid_frequency_hz": GRID_HZ,
        "high_band_grid_harmonic_power_fraction": float(fraction),
        "top_high_band_peaks": peaks,
    }


def _event_metrics(
    residual: torch.Tensor,
    vectors: torch.Tensor,
    *,
    center_seconds: float,
) -> dict[str, object]:
    center_sample = int(round(center_seconds * SAMPLE_RATE))
    radius_samples = int(round(WINDOW_RADIUS_SECONDS * SAMPLE_RATE))
    start = max(0, center_sample - radius_samples)
    stop = min(int(residual.numel()), center_sample + radius_samples)
    segment = residual[start:stop]
    center_frame = min(int(vectors.shape[0]) - 2, max(0, int(round(center_sample / HOP_LENGTH))))
    radius_frames = max(2, int(round(radius_samples / HOP_LENGTH)))
    return {
        "window_start_seconds": start / float(SAMPLE_RATE),
        "window_end_seconds": stop / float(SAMPLE_RATE),
        "hop_lag_samples": HOP_LENGTH,
        "hop_lag_correlation": _normalized_lag_correlation(segment, HOP_LENGTH),
        **_overlap_consistency(vectors, center_frame=center_frame, radius_frames=radius_frames),
        **_grid_comb_metrics(segment),
    }


def _all_sources(
    utterance: OwnedVocoderUtterance,
    baseline: LykenoxContinuousResidualSourceV2,
    candidate: LykenoxContinuousResidualSourceV2,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    frames = int(utterance.mel_frames)
    samples = frames * HOP_LENGTH
    target_residual, _, _ = extract_owned_real_residual(utterance.waveform.cpu(), frame_count=frames)
    target_vectors = residual_analysis_vectors(target_residual)
    conditioning = extract_pitch_conditioning_v2(
        utterance.waveform.cpu().to(torch.float32),
        frame_count=frames,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        frame_length=int(PITCH_CONFIG["frame_length"]),
        min_f0_hz=float(PITCH_CONFIG["min_f0_hz"]),
        max_f0_hz=float(PITCH_CONFIG["max_f0_hz"]),
        anchor_periodicity_threshold=float(PITCH_CONFIG["voiced_periodicity_threshold"]),
        anchor_rms_fraction=float(PITCH_CONFIG["voiced_rms_fraction"]),
    )
    with torch.no_grad():
        baseline_vectors = baseline.generate(
            utterance.mel.unsqueeze(0).cpu(),
            utterance.f0_hz.unsqueeze(0).cpu(),
            utterance.voiced.unsqueeze(0).cpu(),
            utterance.periodicity.unsqueeze(0).cpu(),
        ).squeeze(0)
        candidate_vectors = candidate.generate(
            utterance.mel.unsqueeze(0).cpu(),
            conditioning.f0_track_hz.unsqueeze(0).cpu(),
            conditioning.energy_confidence.unsqueeze(0).cpu(),
            conditioning.periodic_strength.unsqueeze(0).cpu(),
        ).squeeze(0)
    baseline_residual = _ola_vectors(baseline_vectors.unsqueeze(0), output_samples=samples).squeeze(0)
    candidate_residual = _ola_vectors(candidate_vectors.unsqueeze(0), output_samples=samples).squeeze(0)
    return (
        (target_residual.contiguous(), target_vectors.contiguous()),
        (baseline_residual.contiguous(), baseline_vectors.contiguous()),
        (candidate_residual.contiguous(), candidate_vectors.contiguous()),
    )


def main() -> None:
    baseline_path = ROOT / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2" / "best.pt"
    candidate_path = ROOT / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2_pitch_conditioning_v2" / "best.pt"
    if not baseline_path.exists() or not candidate_path.exists():
        raise FileNotFoundError("required V2 baseline/candidate checkpoints are missing")
    baseline = _load_model(baseline_path, BASE_SCHEMA)
    candidate = _load_model(candidate_path, CANDIDATE_SCHEMA, "lykenox-pitch-conditioning-v2-continuous-strength")
    utterances = collect_owned_vocoder_utterances(ROOT, "val", max_items=3)
    by_id = {item.utterance_id: item for item in utterances}
    rows: list[dict[str, object]] = []
    for utterance_id, times in EVENTS.items():
        utterance = by_id.get(utterance_id)
        if utterance is None:
            continue
        target, historical, controlled = _all_sources(utterance, baseline, candidate)
        for time_seconds in times:
            target_metrics = _event_metrics(*target, center_seconds=time_seconds)
            historical_metrics = _event_metrics(*historical, center_seconds=time_seconds)
            controlled_metrics = _event_metrics(*controlled, center_seconds=time_seconds)
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "time_seconds": time_seconds,
                    "real_step3f": target_metrics,
                    "historical_v2": historical_metrics,
                    "v2_pitch_conditioning_v2": controlled_metrics,
                    "candidate_minus_real": {
                        "hop_lag_correlation": float(controlled_metrics["hop_lag_correlation"]) - float(target_metrics["hop_lag_correlation"]),
                        "overlap_relative_disagreement_rms": float(controlled_metrics["overlap_relative_disagreement_rms"]) - float(target_metrics["overlap_relative_disagreement_rms"]),
                        "overlap_adjacent_estimate_correlation_mean": float(controlled_metrics["overlap_adjacent_estimate_correlation_mean"]) - float(target_metrics["overlap_adjacent_estimate_correlation_mean"]),
                        "high_band_grid_harmonic_power_fraction": float(controlled_metrics["high_band_grid_harmonic_power_fraction"]) - float(target_metrics["high_band_grid_harmonic_power_fraction"]),
                    },
                }
            )

    output = ROOT / "models" / "lykenox_identity" / "evaluation" / "generated_vs_reference_diagnostic_v1" / "hop_grid_residual_forensics_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "hop_grid_residual_forensics_complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "training_executed": False,
        "audio_generated": False,
        "wav_written": False,
        "model_inference_executed": True,
        "renderer_executed": False,
        "checkpoint_written": False,
        "hop_length_samples": HOP_LENGTH,
        "hop_grid_frequency_hz": GRID_HZ,
        "hypothesis": "predicted_512_256_vectors_break_overlap_consistency_and_create_hop_grid_comb_tonality",
        "metrics_can_accept_product_quality": False,
        "events": rows,
        "report": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
