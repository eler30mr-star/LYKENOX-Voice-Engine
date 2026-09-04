"""No-training forensic: isolate candidate residual magnitude level vs spectral shape.

Listening established two critical facts:
- speech_0021 candidate magnitude + full target residual phase is natural/correct;
- speech_0022 reference and identity roundtrip are clean, while candidate-magnitude renders retain
  a low grinder-like artifact even when LOW/MID/HIGH magnitude bands are replaced individually.

Therefore this gate freezes the FULL target residual phase and decomposes STFT log magnitude exactly
per frame into:
  log|X(f,t)| = frame_level(t) + spectral_shape(f,t)
where frame_level is the mean log magnitude over frequency and spectral_shape has zero frequency mean.
It emits two exact hybrids:
1) candidate spectral shape + TARGET frame level;
2) TARGET spectral shape + candidate frame level.

This determines whether the remaining speech_0022 artifact is driven by broadband temporal level
modulation or by time-varying spectral shape. No training, optimizer, checkpoint write, renderer
modification, denoise, EQ, duration change, third-party model/service, or product-path normalization
is used. Separate AUDITION files use one common monitor gain per utterance only.
Policy: LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostic_residual_phase_magnitude_forensic_v1 import (
    DEFAULT_SCAN_ITEMS,
    DEFAULT_UTTERANCE_IDS,
    _load_candidate,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import extract_pitch_conditioning_v2
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_time_varying_minimum_phase,
)
from lykenox_voice_engine.training.speech_vocoder_residual_statistics_source_train_v1 import (
    DEFAULT_SEED,
    _utterance_seed,
    synthesize_residual_from_statistics,
)

DIAGNOSTIC_VERSION = "owned-target-phase-magnitude-level-shape-v1"
POLICY_ID = "LYX-POL-001"
OUTPUT_DIR_NAME = "vocoder_target_phase_magnitude_level_shape_v1"
AUDITION_TARGET_RMS_DBFS = -20.0
AUDITION_PEAK_LIMIT = 0.95
EPSILON = 1.0e-6


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().to(torch.float32).contiguous().numpy(), SAMPLE_RATE, subtype="FLOAT")


def _stft(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError("residual must be mono [samples]")
    window = torch.hann_window(N_FFT, dtype=value.dtype, device=value.device)
    return torch.stft(value, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, window=window, center=True, return_complex=True)


def _istft(spectrum: torch.Tensor, *, length: int, dtype: torch.dtype) -> torch.Tensor:
    window = torch.hann_window(N_FFT, dtype=dtype, device=spectrum.device)
    value = torch.istft(spectrum, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, window=window, center=True, length=length)
    if int(value.numel()) != length or not bool(torch.isfinite(value).all()):
        raise RuntimeError("level/shape forensic violated waveform contract")
    return value.to(torch.float32).contiguous()


def _decompose_log_magnitude(magnitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if magnitude.ndim != 2:
        raise ValueError("magnitude must have shape [bins, frames]")
    log_mag = torch.log(magnitude.clamp_min(EPSILON))
    frame_level = log_mag.mean(dim=0, keepdim=True)
    spectral_shape = log_mag - frame_level
    return frame_level, spectral_shape


def _compose_magnitude(frame_level: torch.Tensor, spectral_shape: torch.Tensor) -> torch.Tensor:
    if frame_level.ndim != 2 or frame_level.shape[0] != 1 or spectral_shape.ndim != 2:
        raise ValueError("invalid level/shape geometry")
    if frame_level.shape[1] != spectral_shape.shape[1]:
        raise ValueError("level/shape frame mismatch")
    return torch.exp(frame_level + spectral_shape).clamp_min(EPSILON)


def _residual_from_magnitude_phase(magnitude: torch.Tensor, phase: torch.Tensor, *, length: int, dtype: torch.dtype) -> torch.Tensor:
    if magnitude.shape != phase.shape:
        raise RuntimeError("magnitude/phase geometry mismatch")
    return _istft(torch.polar(magnitude.to(torch.float32), phase.to(torch.float32)), length=length, dtype=dtype)


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.to(torch.float64).square().mean().clamp_min(1.0e-30)))


def _peak(value: torch.Tensor) -> float:
    return float(value.detach().abs().max())


def _common_audition_gain(renders: dict[str, torch.Tensor], reference: torch.Tensor) -> float:
    target_rms = 10.0 ** (AUDITION_TARGET_RMS_DBFS / 20.0)
    gain_by_rms = target_rms / max(_rms(reference), 1.0e-12)
    maximum_peak = max(_peak(value) for value in renders.values())
    gain_by_peak = AUDITION_PEAK_LIMIT / max(maximum_peak, 1.0e-12)
    return max(1.0, min(gain_by_rms, gain_by_peak))


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1.0e-20)
    return float((a * b).sum() / denom)


def run_target_phase_magnitude_level_shape(
    root: Path,
    *,
    utterance_ids: tuple[str, ...] = DEFAULT_UTTERANCE_IDS,
    scan_items: int = DEFAULT_SCAN_ITEMS,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = Path(checkpoint).resolve() if checkpoint is not None else root / "models" / "lykenox_identity" / "training" / "residual_statistics_source_v1" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"candidate checkpoint missing: {checkpoint}")
    output_dir = Path(output_dir).resolve() if output_dir is not None else root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    raw_dir = output_dir / "raw"
    audition_dir = output_dir / "audition"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audition_dir.mkdir(parents=True, exist_ok=True)

    wanted = tuple(dict.fromkeys(utterance_ids))
    candidate = _load_candidate(checkpoint)
    utterances = collect_owned_vocoder_utterances(root, split="val", max_items=max(scan_items, len(wanted)))
    by_id = {u.utterance_id: u for u in utterances}
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise RuntimeError("requested held-out utterances not found: " + ", ".join(missing))

    items: list[dict[str, object]] = []
    with torch.no_grad():
        for utterance_id in wanted:
            utterance = by_id[utterance_id]
            frames = int(utterance.mel_frames)
            expected_samples = frames * HOP_LENGTH
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            target_residual, oracle_cepstrum, _ = extract_owned_real_residual(reference, frame_count=frames)
            conditioning = extract_pitch_conditioning_v2(
                reference,
                frame_count=frames,
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                frame_length=int(PITCH_CONFIG["frame_length"]),
                min_f0_hz=float(PITCH_CONFIG["min_f0_hz"]),
                max_f0_hz=float(PITCH_CONFIG["max_f0_hz"]),
                anchor_periodicity_threshold=float(PITCH_CONFIG["voiced_periodicity_threshold"]),
                anchor_rms_fraction=float(PITCH_CONFIG["voiced_rms_fraction"]),
            )
            source_cepstrum, log_rms, source_periodicity = candidate(
                utterance.mel.unsqueeze(0).cpu(),
                conditioning.f0_track_hz.unsqueeze(0).cpu(),
                conditioning.energy_confidence.unsqueeze(0).cpu(),
                conditioning.periodic_strength.unsqueeze(0).cpu(),
            )
            candidate_residual = synthesize_residual_from_statistics(
                source_cepstrum,
                log_rms,
                source_periodicity,
                conditioning.f0_track_hz.unsqueeze(0).cpu(),
                seed=_utterance_seed(utterance_id, DEFAULT_SEED + 1800000),
            )
            if candidate_residual.ndim == 2:
                candidate_residual = candidate_residual[0]
            candidate_residual = candidate_residual.to(torch.float32).contiguous()
            if int(reference.numel()) != expected_samples or candidate_residual.shape != target_residual.shape:
                raise RuntimeError("held-out length contract changed")

            target_spec = _stft(target_residual)
            candidate_spec = _stft(candidate_residual)
            target_mag = target_spec.abs()
            candidate_mag = candidate_spec.abs()
            target_phase = torch.angle(target_spec)
            target_level, target_shape = _decompose_log_magnitude(target_mag)
            candidate_level, candidate_shape = _decompose_log_magnitude(candidate_mag)

            magnitudes = {
                "candidate_full": candidate_mag,
                "candidate_shape_target_level": _compose_magnitude(target_level, candidate_shape),
                "target_shape_candidate_level": _compose_magnitude(candidate_level, target_shape),
                "target_full": target_mag,
            }
            residuals = {
                key: _residual_from_magnitude_phase(mag, target_phase, length=int(target_residual.numel()), dtype=target_residual.dtype)
                for key, mag in magnitudes.items()
            }
            renders = {
                key: render_time_varying_minimum_phase(
                    residual.unsqueeze(0), oracle_cepstrum.unsqueeze(0), hop_length=HOP_LENGTH, n_fft=N_FFT
                ).squeeze(0)
                for key, residual in residuals.items()
            }
            renders["reference"] = reference
            labels = {
                "reference": "reference",
                "candidate_full": "candidate_mag_target_phase_baseline",
                "candidate_shape_target_level": "candidate_shape_target_frame_level_render",
                "target_shape_candidate_level": "target_shape_candidate_frame_level_render",
                "target_full": "identity_roundtrip_ceiling",
            }

            raw_paths: dict[str, str] = {}
            for key, label in labels.items():
                path = raw_dir / f"{utterance_id}__{label}.wav"
                _write(path, renders[key])
                raw_paths[label] = str(path)

            audition_gain = _common_audition_gain(renders, reference)
            audition_paths: dict[str, str] = {}
            for key, label in labels.items():
                path = audition_dir / f"{utterance_id}__{label}__AUDITION.wav"
                _write(path, renders[key] * audition_gain)
                audition_paths[label] = str(path)

            level_error = (target_level - candidate_level).squeeze(0)
            level_delta_error = level_error[1:] - level_error[:-1]
            shape_l1 = float((target_shape - candidate_shape).abs().mean())
            items.append({
                "utterance_id": utterance_id,
                "audition_gain_linear": audition_gain,
                "audition_gain_db": 20.0 * math.log10(max(audition_gain, 1.0e-12)),
                "frame_log_level_target_candidate_pearson": _pearson(target_level, candidate_level),
                "frame_log_level_mae": float(level_error.abs().mean()),
                "frame_log_level_delta_mae": float(level_delta_error.abs().mean()),
                "spectral_shape_log_l1": shape_l1,
                "raw_paths": raw_paths,
                "audition_paths": audition_paths,
            })

    report: dict[str, object] = {
        "status": "ready_for_target_phase_magnitude_level_shape_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "split": "val",
        "utterance_ids": list(wanted),
        "checkpoint": str(checkpoint),
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "renderer_modified": False,
        "product_posthoc_gain_normalization_used": False,
        "audition_monitor_gain_used": True,
        "audition_monitor_gain_common_within_each_utterance": True,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "predicted_duration_modified": False,
        "metrics_can_accept_product_quality": False,
        "known_listening_evidence": (
            "speech_0021 candidate magnitude plus target phase is natural/correct; "
            "speech_0022 reference and identity roundtrip are clean; speech_0022 candidate-magnitude "
            "renders retain a low grinder-like artifact across LOW/MID/HIGH single-band swaps"
        ),
        "listening_interpretation": {
            "candidate_shape_target_level_cleaner": "broadband_temporal_frame_level_modulation_is_primary_remaining_magnitude_failure",
            "target_shape_candidate_level_cleaner": "time_varying_spectral_shape_is_primary_remaining_magnitude_failure",
            "both_partial": "both_frame_level_and_spectral_shape_contribute",
            "neither_cleaner_but_identity_clean": "remaining_magnitude_failure_is_coupled_and_requires_next_time_frequency_localization_gate",
        },
        "items": items,
        "next_action": "listen to speech_0022 AUDITION baseline, two level/shape hybrids, identity and reference; do not train",
    }
    _atomic_json(output_dir / "target_phase_magnitude_level_shape_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--utterance-id", action="append", dest="utterance_ids", default=None)
    parser.add_argument("--scan-items", type=int, default=DEFAULT_SCAN_ITEMS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    requested = tuple(args.utterance_ids) if args.utterance_ids else DEFAULT_UTTERANCE_IDS
    print(json.dumps(run_target_phase_magnitude_level_shape(
        args.root,
        utterance_ids=requested,
        scan_items=args.scan_items,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
