"""Forensic comparison of learned residuals against the real Step-3f residual at stable voiced anomalies.

This diagnostic trains nothing and writes no WAV files. It reconstructs the historical Continuous
Source V2 and the controlled pitch-conditioning-V2 candidate in memory, then compares their residuals
directly with the owned Step-3f real residual at known stable-voiced anomaly windows.

The purpose is to test a specific physical hypothesis: learned sources may be excessively coherent
from pitch cycle to pitch cycle, turning stochastic/aperiodic residual detail into a repeated tonal
pattern before the fixed minimum-phase renderer. Metrics localize/reject only; they cannot accept
product quality. Policy: LYX-POL-001.
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


DIAGNOSTIC_VERSION = "owned-stable-voiced-residual-forensics-v1"
POLICY_ID = "LYX-POL-001"
CYCLE_BINS = 128
WINDOW_RADIUS_SECONDS = 0.30
EVENTS = {
    "speech_0024_1778f351cc1f_seg_006": (4.00,),
    "speech_0021_6cd35984e877_seg_001": (1.10,),
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
    if lag < 1 or value.numel() <= lag + 8:
        return float("nan")
    left = value[:-lag].to(torch.float64)
    right = value[lag:].to(torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    denom = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1.0e-20)
    return float((left * right).sum() / denom)


def _spectral_flatness(value: torch.Tensor, *, low_hz: float = 3500.0) -> float:
    n_fft = 2048
    if value.numel() < n_fft:
        value = F.pad(value, (0, n_fft - int(value.numel())))
    window = torch.hann_window(int(value.numel()), dtype=value.dtype)
    spectrum = torch.fft.rfft(value * window).abs().clamp_min(1.0e-8)
    freqs = torch.fft.rfftfreq(int(value.numel()), 1.0 / float(SAMPLE_RATE))
    band = spectrum[(freqs >= low_hz) & (freqs <= 11000.0)]
    if band.numel() < 4:
        return float("nan")
    return float(torch.exp(torch.log(band).mean()) / band.mean().clamp_min(1.0e-8))


def _canonical_cycles(value: torch.Tensor, *, start: int, stop: int, f0_hz: float) -> torch.Tensor:
    period = max(2, int(round(float(SAMPLE_RATE) / max(float(f0_hz), 1.0))))
    first = start
    cycles: list[torch.Tensor] = []
    while first + period <= stop:
        cycle = value[first : first + period]
        canonical = F.interpolate(
            cycle.view(1, 1, -1), size=CYCLE_BINS, mode="linear", align_corners=True
        )[0, 0]
        canonical = canonical - canonical.mean()
        rms = torch.sqrt(canonical.square().mean().clamp_min(1.0e-10))
        cycles.append(canonical / rms)
        first += period
    if len(cycles) < 3:
        return torch.empty(0, CYCLE_BINS)
    return torch.stack(cycles, dim=0)


def _cycle_statistics(cycles: torch.Tensor) -> dict[str, float | int]:
    if cycles.ndim != 2 or cycles.shape[0] < 3:
        return {
            "cycle_count": int(cycles.shape[0]) if cycles.ndim == 2 else 0,
            "adjacent_cycle_cosine_mean": float("nan"),
            "coherent_cycle_energy_fraction": float("nan"),
            "cycle_deviation_rms": float("nan"),
        }
    normalized = F.normalize(cycles, dim=-1)
    adjacent = (normalized[:-1] * normalized[1:]).sum(dim=-1)
    mean_cycle = cycles.mean(dim=0)
    total_energy = cycles.square().mean()
    coherent_energy = mean_cycle.square().mean()
    deviation = cycles - mean_cycle.unsqueeze(0)
    return {
        "cycle_count": int(cycles.shape[0]),
        "adjacent_cycle_cosine_mean": float(adjacent.mean()),
        "coherent_cycle_energy_fraction": float(coherent_energy / total_energy.clamp_min(1.0e-10)),
        "cycle_deviation_rms": float(torch.sqrt(deviation.square().mean().clamp_min(1.0e-10))),
    }


def _window_metrics(residual: torch.Tensor, *, center_seconds: float, f0_hz: float) -> dict[str, object]:
    center = int(round(center_seconds * SAMPLE_RATE))
    radius = int(round(WINDOW_RADIUS_SECONDS * SAMPLE_RATE))
    start = max(0, center - radius)
    stop = min(int(residual.numel()), center + radius)
    window = residual[start:stop]
    lag = max(1, int(round(float(SAMPLE_RATE) / max(float(f0_hz), 1.0))))
    cycles = _canonical_cycles(residual, start=start, stop=stop, f0_hz=f0_hz)
    return {
        "window_start_seconds": start / float(SAMPLE_RATE),
        "window_end_seconds": stop / float(SAMPLE_RATE),
        "f0_lag_samples": lag,
        "f0_lag_correlation": _normalized_lag_correlation(window, lag),
        "high_band_spectral_flatness": _spectral_flatness(window),
        **_cycle_statistics(cycles),
    }


def _residuals(
    utterance: OwnedVocoderUtterance,
    baseline: LykenoxContinuousResidualSourceV2,
    candidate: LykenoxContinuousResidualSourceV2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = int(utterance.mel_frames)
    samples = frames * HOP_LENGTH
    target, _, _ = extract_owned_real_residual(utterance.waveform.cpu(), frame_count=frames)
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
        base_vectors = baseline.generate(
            utterance.mel.unsqueeze(0).cpu(),
            utterance.f0_hz.unsqueeze(0).cpu(),
            utterance.voiced.unsqueeze(0).cpu(),
            utterance.periodicity.unsqueeze(0).cpu(),
        )
        cand_vectors = candidate.generate(
            utterance.mel.unsqueeze(0).cpu(),
            conditioning.f0_track_hz.unsqueeze(0).cpu(),
            conditioning.energy_confidence.unsqueeze(0).cpu(),
            conditioning.periodic_strength.unsqueeze(0).cpu(),
        )
    base = _ola_vectors(base_vectors, output_samples=samples).squeeze(0)
    cand = _ola_vectors(cand_vectors, output_samples=samples).squeeze(0)
    return target.contiguous(), base.contiguous(), cand.contiguous()


def main() -> None:
    baseline_path = ROOT / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2" / "best.pt"
    candidate_path = ROOT / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2_pitch_conditioning_v2" / "best.pt"
    if not baseline_path.exists() or not candidate_path.exists():
        raise FileNotFoundError("required V2 baseline/candidate checkpoints are missing")
    baseline = _load_model(baseline_path, BASE_SCHEMA)
    candidate = _load_model(
        candidate_path,
        CANDIDATE_SCHEMA,
        "lykenox-pitch-conditioning-v2-continuous-strength",
    )
    utterances = collect_owned_vocoder_utterances(ROOT, "val", max_items=3)
    by_id = {item.utterance_id: item for item in utterances}
    rows: list[dict[str, object]] = []
    for utterance_id, times in EVENTS.items():
        utterance = by_id.get(utterance_id)
        if utterance is None:
            continue
        target, base, cand = _residuals(utterance, baseline, candidate)
        for time_seconds in times:
            frame = min(int(utterance.mel_frames) - 1, max(0, int(round(time_seconds * SAMPLE_RATE / HOP_LENGTH))))
            f0 = float(utterance.f0_hz[frame])
            if f0 <= 0.0:
                continue
            target_metrics = _window_metrics(target, center_seconds=time_seconds, f0_hz=f0)
            base_metrics = _window_metrics(base, center_seconds=time_seconds, f0_hz=f0)
            cand_metrics = _window_metrics(cand, center_seconds=time_seconds, f0_hz=f0)
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "time_seconds": time_seconds,
                    "cached_f0_hz": f0,
                    "target_real_step3f": target_metrics,
                    "historical_v2": base_metrics,
                    "v2_pitch_conditioning_v2": cand_metrics,
                    "candidate_minus_target": {
                        "f0_lag_correlation": float(cand_metrics["f0_lag_correlation"]) - float(target_metrics["f0_lag_correlation"]),
                        "coherent_cycle_energy_fraction": float(cand_metrics["coherent_cycle_energy_fraction"]) - float(target_metrics["coherent_cycle_energy_fraction"]),
                        "high_band_spectral_flatness": float(cand_metrics["high_band_spectral_flatness"]) - float(target_metrics["high_band_spectral_flatness"]),
                        "cycle_deviation_rms": float(cand_metrics["cycle_deviation_rms"]) - float(target_metrics["cycle_deviation_rms"]),
                    },
                }
            )

    output = ROOT / "models" / "lykenox_identity" / "evaluation" / "generated_vs_reference_diagnostic_v1" / "stable_voiced_residual_forensics_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "stable_voiced_residual_forensics_complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "training_executed": False,
        "audio_generated": False,
        "wav_written": False,
        "model_inference_executed": True,
        "checkpoint_written": False,
        "renderer_executed": False,
        "metrics_can_accept_product_quality": False,
        "hypothesis": "learned_residual_is_excessively_cycle_coherent_before_renderer",
        "events": rows,
        "report": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
