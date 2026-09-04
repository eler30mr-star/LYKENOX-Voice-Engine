"""No-training magnitude-band cleanup forensic under the accepted target-phase ceiling.

Listening established that residual phase/temporal coherence was the primary defect: candidate
magnitude combined with target residual phase is close to clean, while candidate phase is not.
More critical listening still reports a small remaining lack of cleanliness even under the target
phase ceiling. This diagnostic therefore freezes the full target residual phase and changes ONLY
which STFT-magnitude frequency band comes from the real residual versus the existing candidate.

It also emits separate AUDITION copies with one common scalar monitor gain per utterance. The RAW
renders remain untouched. The monitor gain is identical for every comparison file in an utterance,
so relative level differences are preserved; it is not part of the vocoder/product path and cannot
be used as product acceptance evidence.

No training, optimizer, checkpoint write, renderer modification, EQ, denoise, enhancement, duration
modification, third-party model/service, or product-path gain normalization is used.
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


DIAGNOSTIC_VERSION = "owned-target-phase-magnitude-band-cleanup-v1"
POLICY_ID = "LYX-POL-001"
OUTPUT_DIR_NAME = "vocoder_target_phase_magnitude_band_cleanup_v1"
LOW_BAND_MAX_HZ = 1800.0
MID_BAND_MAX_HZ = 4500.0
AUDITION_TARGET_RMS_DBFS = -20.0
AUDITION_PEAK_LIMIT = 0.95
EPSILON = 1.0e-8


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        waveform.detach().cpu().to(torch.float32).contiguous().numpy(),
        SAMPLE_RATE,
        subtype="FLOAT",
    )


def _stft(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError("residual must be mono [samples]")
    window = torch.hann_window(N_FFT, dtype=value.dtype, device=value.device)
    return torch.stft(
        value,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )


def _istft(spectrum: torch.Tensor, *, length: int, dtype: torch.dtype) -> torch.Tensor:
    window = torch.hann_window(N_FFT, dtype=dtype, device=spectrum.device)
    waveform = torch.istft(
        spectrum,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        length=length,
    )
    if int(waveform.numel()) != length:
        raise RuntimeError("magnitude-band forensic violated exact-length contract")
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("magnitude-band forensic produced non-finite waveform")
    return waveform.to(torch.float32).contiguous()


def _hybrid_residual(
    candidate_magnitude: torch.Tensor,
    target_magnitude: torch.Tensor,
    target_phase: torch.Tensor,
    *,
    target_band_mask: torch.Tensor,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if candidate_magnitude.shape != target_magnitude.shape or target_phase.shape != target_magnitude.shape:
        raise RuntimeError("magnitude/phase geometry mismatch")
    if target_band_mask.ndim != 1 or int(target_band_mask.numel()) != int(target_magnitude.shape[0]):
        raise RuntimeError("band mask geometry mismatch")
    magnitude = torch.where(
        target_band_mask[:, None],
        target_magnitude,
        candidate_magnitude,
    )
    spectrum = torch.polar(magnitude.to(torch.float32), target_phase.to(torch.float32))
    return _istft(spectrum, length=length, dtype=dtype)


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.to(torch.float64).square().mean().clamp_min(1.0e-30)))


def _peak(value: torch.Tensor) -> float:
    return float(value.detach().abs().max())


def _common_audition_gain(renders: dict[str, torch.Tensor], reference: torch.Tensor) -> float:
    target_rms = 10.0 ** (AUDITION_TARGET_RMS_DBFS / 20.0)
    reference_rms = _rms(reference)
    gain_by_rms = target_rms / max(reference_rms, 1.0e-12)
    maximum_peak = max(_peak(value) for value in renders.values())
    gain_by_peak = AUDITION_PEAK_LIMIT / max(maximum_peak, 1.0e-12)
    # Audition copies are only for easier human comparison. Never attenuate below RAW level.
    return max(1.0, min(gain_by_rms, gain_by_peak))


def _band_log_magnitude_l1(
    target_magnitude: torch.Tensor,
    candidate_magnitude: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    target = torch.log(target_magnitude[mask].clamp_min(1.0e-6))
    candidate = torch.log(candidate_magnitude[mask].clamp_min(1.0e-6))
    return float((target - candidate).abs().mean())


def run_target_phase_magnitude_band_cleanup(
    root: Path,
    *,
    utterance_ids: tuple[str, ...] = DEFAULT_UTTERANCE_IDS,
    scan_items: int = DEFAULT_SCAN_ITEMS,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root
        / "models"
        / "lykenox_identity"
        / "training"
        / "residual_statistics_source_v1"
        / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"candidate checkpoint missing: {checkpoint}")

    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / OUTPUT_DIR_NAME
    )
    raw_dir = output_dir / "raw"
    audition_dir = output_dir / "audition"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audition_dir.mkdir(parents=True, exist_ok=True)

    wanted = tuple(dict.fromkeys(utterance_ids))
    if not wanted:
        raise ValueError("at least one utterance id is required")

    candidate = _load_candidate(checkpoint)
    utterances = collect_owned_vocoder_utterances(
        root,
        split="val",
        max_items=max(scan_items, len(wanted)),
    )
    by_id = {utterance.utterance_id: utterance for utterance in utterances}
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise RuntimeError("requested held-out utterances not found: " + ", ".join(missing))

    frequencies = torch.fft.rfftfreq(N_FFT, d=1.0 / float(SAMPLE_RATE))
    low_mask = frequencies < LOW_BAND_MAX_HZ
    mid_mask = (frequencies >= LOW_BAND_MAX_HZ) & (frequencies < MID_BAND_MAX_HZ)
    high_mask = frequencies >= MID_BAND_MAX_HZ
    all_mask = torch.ones_like(low_mask, dtype=torch.bool)
    none_mask = torch.zeros_like(low_mask, dtype=torch.bool)

    items: list[dict[str, object]] = []
    with torch.no_grad():
        for utterance_id in wanted:
            utterance = by_id[utterance_id]
            frames = int(utterance.mel_frames)
            expected_samples = frames * HOP_LENGTH
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            if int(reference.numel()) != expected_samples:
                raise RuntimeError("held-out waveform length contract changed")

            target_residual, oracle_cepstrum, _ = extract_owned_real_residual(
                reference,
                frame_count=frames,
            )
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
            if candidate_residual.shape != target_residual.shape:
                raise RuntimeError("candidate residual length differs from target residual")

            target_spec = _stft(target_residual)
            candidate_spec = _stft(candidate_residual)
            if target_spec.shape != candidate_spec.shape:
                raise RuntimeError("target/candidate STFT geometry mismatch")
            target_mag = target_spec.abs()
            candidate_mag = candidate_spec.abs()
            target_phase = torch.angle(target_spec)
            length = int(target_residual.numel())

            residuals = {
                "candidate_mag_target_phase": _hybrid_residual(
                    candidate_mag, target_mag, target_phase,
                    target_band_mask=none_mask,
                    length=length,
                    dtype=target_residual.dtype,
                ),
                "target_low_mag_target_phase": _hybrid_residual(
                    candidate_mag, target_mag, target_phase,
                    target_band_mask=low_mask,
                    length=length,
                    dtype=target_residual.dtype,
                ),
                "target_mid_mag_target_phase": _hybrid_residual(
                    candidate_mag, target_mag, target_phase,
                    target_band_mask=mid_mask,
                    length=length,
                    dtype=target_residual.dtype,
                ),
                "target_high_mag_target_phase": _hybrid_residual(
                    candidate_mag, target_mag, target_phase,
                    target_band_mask=high_mask,
                    length=length,
                    dtype=target_residual.dtype,
                ),
                "target_full_mag_target_phase": _hybrid_residual(
                    candidate_mag, target_mag, target_phase,
                    target_band_mask=all_mask,
                    length=length,
                    dtype=target_residual.dtype,
                ),
            }

            renders = {
                key: render_time_varying_minimum_phase(
                    residual.unsqueeze(0),
                    oracle_cepstrum.unsqueeze(0),
                    hop_length=HOP_LENGTH,
                    n_fft=N_FFT,
                ).squeeze(0)
                for key, residual in residuals.items()
            }
            renders["reference"] = reference
            for value in renders.values():
                if value.shape != reference.shape:
                    raise RuntimeError("magnitude-band render/reference shape mismatch")
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError("magnitude-band render contains non-finite values")

            labels = {
                "reference": "reference",
                "candidate_mag_target_phase": "candidate_mag_target_phase_baseline",
                "target_low_mag_target_phase": "target_low_mag_target_phase_render",
                "target_mid_mag_target_phase": "target_mid_mag_target_phase_render",
                "target_high_mag_target_phase": "target_high_mag_target_phase_render",
                "target_full_mag_target_phase": "identity_roundtrip_ceiling",
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

            items.append(
                {
                    "utterance_id": utterance_id,
                    "audition_gain_linear": audition_gain,
                    "audition_gain_db": 20.0 * math.log10(max(audition_gain, 1.0e-12)),
                    "candidate_vs_target_log_magnitude_l1": {
                        "low_0_to_1800_hz": _band_log_magnitude_l1(target_mag, candidate_mag, low_mask),
                        "mid_1800_to_4500_hz": _band_log_magnitude_l1(target_mag, candidate_mag, mid_mask),
                        "high_4500_to_nyquist_hz": _band_log_magnitude_l1(target_mag, candidate_mag, high_mask),
                    },
                    "raw_paths": raw_paths,
                    "audition_paths": audition_paths,
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_target_phase_magnitude_band_cleanup_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "split": "val",
        "utterance_ids": list(wanted),
        "checkpoint": str(checkpoint),
        "low_band_max_hz": LOW_BAND_MAX_HZ,
        "mid_band_max_hz": MID_BAND_MAX_HZ,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "renderer_modified": False,
        "product_posthoc_gain_normalization_used": False,
        "audition_monitor_gain_used": True,
        "audition_monitor_gain_common_within_each_utterance": True,
        "audition_target_rms_dbfs": AUDITION_TARGET_RMS_DBFS,
        "audition_peak_limit": AUDITION_PEAK_LIMIT,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "posthoc_enhancement_used": False,
        "predicted_duration_modified": False,
        "third_party_model_used": False,
        "metrics_can_accept_product_quality": False,
        "known_listening_evidence": (
            "speech_0021 target temporal phase increments plus candidate anchor is near-clean; "
            "smooth group-delay anchor is worse; full target phase/anchor is very good but still has "
            "minor cleanliness defects and raw listening level is too low for confident inspection"
        ),
        "listening_interpretation": {
            "target_low_improves": "remaining_cleanup_error_is_concentrated_below_1800_hz",
            "target_mid_improves": "remaining_cleanup_error_is_concentrated_between_1800_and_4500_hz",
            "target_high_improves": "remaining_wind_phone_texture_is_concentrated_above_4500_hz",
            "multiple_band_swaps_improve": "remaining_magnitude_error_spans_multiple_frequency_regions",
            "identity_roundtrip_ceiling_not_clean": "remaining_defect_is_not_candidate_magnitude_and_requires_rechecking_oracle_renderer_analysis_path",
        },
        "items": items,
        "next_action": (
            "listen_to_AUDITION files first: candidate_mag_target_phase_baseline, target_low, target_mid, "
            "target_high, and identity_roundtrip_ceiling; do not train or modify renderer"
        ),
    }
    _atomic_json(output_dir / "target_phase_magnitude_band_cleanup_report.json", report)
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
    print(
        json.dumps(
            run_target_phase_magnitude_band_cleanup(
                args.root,
                utterance_ids=requested,
                scan_items=args.scan_items,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
