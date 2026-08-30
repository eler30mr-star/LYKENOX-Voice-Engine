"""Read-only level and intelligibility attribution audit for accepted vocoder v4.2.

The accepted v4.2 oracle is already close to reference RMS, so this audit does not add gain
or retrain anything.  It keeps three held-out utterances on the teacher duration grid and
swaps predicted mel and predicted F0/voicing independently before v4.2.  A fifth route uses
fully predicted durations.  The report separates total RMS from active-speech RMS and
above-300-Hz energy so a periodic or low-band signal cannot masquerade as useful volume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.models.vocoder import VOCODER_GENERATOR_V4_2_ARCHITECTURE
from lykenox_voice_engine.runtime.speech_conditioning import prepare_speech_vocoder_conditioning
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    FRAME_CONTEXT_VERSION,
    TRAINER_CONTRACT_VERSION as ACOUSTIC_TRAINER_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import (
    _wave_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    VOCODER_GRID_ARTIFACT_VERSION,
    frame_grid_artifact_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    target_relative_presence_loss,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance import (
    _load_reference_waveform,
)


AUDIT_VERSION = "vocoder-v4-2-level-attribution-audit-v1"
VALIDATION_INDICES = (0, 1, 2)
TEACHER_GRID_VARIANTS = (
    "v4_2_oracle",
    "v4_2_predicted_mel_target_prosody",
    "v4_2_target_mel_predicted_prosody",
    "v4_2_predicted_mel_predicted_prosody_teacher_grid",
)
FULL_ROUTE_VARIANT = "v4_2_full_reference_free"
OUTPUT_DIR_NAME = "vocoder_v4_2_level_attribution_audit_v1"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _db_ratio(value: float, reference: float) -> float:
    if value <= 0.0 or reference <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(value / reference)


def _reference_active_frame_mask(
    waveform: torch.Tensor,
    hop_length: int,
    *,
    relative_threshold: float = 0.10,
) -> torch.Tensor:
    """Select energetic reference frames without altering audio or using model output."""
    if waveform.ndim != 1 or waveform.numel() < hop_length:
        raise ValueError("waveform must be one-dimensional and at least one hop long")
    if waveform.numel() % hop_length != 0:
        raise ValueError("waveform length must be divisible by hop_length")
    frames = waveform.reshape(-1, hop_length)
    frame_rms = torch.sqrt(frames.square().mean(dim=1).clamp_min(1e-20))
    threshold = torch.maximum(
        frame_rms.max() * float(relative_threshold),
        torch.tensor(1e-5, device=waveform.device, dtype=waveform.dtype),
    )
    mask = frame_rms >= threshold
    if not bool(mask.any()):
        mask[torch.argmax(frame_rms)] = True
    return mask


def _active_rms(waveform: torch.Tensor, frame_mask: torch.Tensor, hop_length: int) -> float:
    if waveform.ndim != 1 or waveform.numel() % hop_length != 0:
        raise ValueError("waveform must be one-dimensional and hop-aligned")
    frames = waveform.reshape(-1, hop_length)
    if frame_mask.ndim != 1 or frame_mask.numel() != frames.shape[0]:
        raise ValueError("frame_mask must match waveform frame count")
    selected = frames[frame_mask]
    if selected.numel() == 0:
        return 0.0
    return float(torch.sqrt(selected.square().mean().clamp_min(1e-20)))


def _crest_factor_db(peak: float, rms: float) -> float:
    return _db_ratio(peak, rms)


def _grid_report(waveform: torch.Tensor, sample_rate: int, hop_length: int) -> dict[str, object]:
    result = frame_grid_artifact_metrics(
        waveform,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    return {
        "version": VOCODER_GRID_ARTIFACT_VERSION,
        "frame_rate_hz": round(float(result.frame_rate_hz), 6),
        "hop_autocorrelation": round(float(result.hop_autocorrelation[0]), 6),
        "double_hop_autocorrelation": round(float(result.double_hop_autocorrelation[0]), 6),
        "grid_harmonic_power_fraction": round(
            float(result.grid_harmonic_power_fraction[0]), 6
        ),
        "severe_grid_artifact": bool(result.severe_grid_artifact[0]),
    }


def _base_level_metrics(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    active_mask: torch.Tensor,
) -> dict[str, object]:
    wave = _wave_metrics(waveform, sample_rate)
    rms = float(wave["rms"])
    peak = float(wave["peak"])
    bands = wave.get("band_power_fraction")
    if not isinstance(bands, dict):
        raise RuntimeError("wave metrics did not return band fractions")
    above_300_fraction = float(bands["300_3000"]) + float(bands["3000_nyquist"])
    above_300_rms = rms * math.sqrt(max(0.0, above_300_fraction))
    return {
        **wave,
        "active_rms": round(_active_rms(waveform, active_mask, hop_length), 7),
        "crest_factor_db": round(_crest_factor_db(peak, rms), 4),
        "above_300hz_power_fraction": round(above_300_fraction, 7),
        "above_300hz_rms": round(above_300_rms, 7),
        "grid_artifact": _grid_report(waveform, sample_rate, hop_length),
    }


def _teacher_grid_metrics(
    waveform: torch.Tensor,
    reference: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    active_mask: torch.Tensor,
    reference_metrics: dict[str, object],
) -> dict[str, object]:
    metrics = _base_level_metrics(
        waveform,
        sample_rate=sample_rate,
        hop_length=hop_length,
        active_mask=active_mask,
    )
    presence = target_relative_presence_loss(
        waveform.unsqueeze(0),
        reference.unsqueeze(0),
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    reference_rms = float(reference_metrics["rms"])
    reference_peak = float(reference_metrics["peak"])
    reference_active_rms = float(reference_metrics["active_rms"])
    reference_above_300_rms = float(reference_metrics["above_300hz_rms"])
    return {
        **metrics,
        "rms_relative_to_reference_db": round(
            _db_ratio(float(metrics["rms"]), reference_rms), 4
        ),
        "active_rms_relative_to_reference_db": round(
            _db_ratio(float(metrics["active_rms"]), reference_active_rms), 4
        ),
        "peak_relative_to_reference_db": round(
            _db_ratio(float(metrics["peak"]), reference_peak), 4
        ),
        "above_300hz_rms_relative_to_reference_db": round(
            _db_ratio(float(metrics["above_300hz_rms"]), reference_above_300_rms), 4
        ),
        "presence_loss": round(float(presence.loss.detach()), 6),
        "presence_1k_8k_error_db": round(
            float(presence.presence_1k_8k_error_db.detach()), 6
        ),
        "presence_band_80_300": round(
            float(presence.prediction_band_fractions[0].detach()), 7
        ),
        "presence_band_300_1000": round(
            float(presence.prediction_band_fractions[1].detach()), 7
        ),
        "presence_band_1k_3k": round(
            float(presence.prediction_band_fractions[2].detach()), 7
        ),
        "presence_band_3k_8k": round(
            float(presence.prediction_band_fractions[3].detach()), 7
        ),
    }


def _protected_paths(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
        "v4_1_best": training / "vocoder_source_filter_v4_1" / "best.pt",
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_prior_last": training / "vocoder_direct_waveform_v6" / "last.pt",
        "v6_prior_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v6_clarity_last": training / "vocoder_direct_waveform_v6_clarity_guard_v1" / "last.pt",
        "v6_clarity_best": training / "vocoder_direct_waveform_v6_clarity_guard_v1" / "best.pt",
        "v7_last": training / "vocoder_source_free_v7_first_epoch" / "last.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
    }


def run_v4_2_level_attribution_audit(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected_paths(root)
    required = ("acoustic_v2_best", "v4_2_best")
    missing = [name for name in required if not protected[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing required level-audit checkpoints: {missing}")
    before = {name: _sha256(path) for name, path in protected.items()}

    acoustic, acoustic_payload = load_acoustic_prosody_checkpoint(
        protected["acoustic_v2_best"]
    )
    acoustic_run = acoustic_payload.get("run_config")
    if not isinstance(acoustic_run, dict):
        raise RuntimeError("Acoustic v2 checkpoint is missing run_config")
    acoustic_identity_exact = (
        acoustic.config.frame_context_version == FRAME_CONTEXT_VERSION
        and acoustic_run.get("trainer_contract_version")
        == ACOUSTIC_TRAINER_CONTRACT_VERSION
        and acoustic_run.get("frame_context_version") == FRAME_CONTEXT_VERSION
    )
    if not acoustic_identity_exact:
        raise RuntimeError("Level audit requires accepted acoustic frame-context v2")

    vocoder, _discriminator, vocoder_payload = load_v4_2_checkpoint(
        protected["v4_2_best"]
    )
    vocoder_identity_exact = (
        vocoder.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and vocoder_payload.get("generator_architecture")
        == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not vocoder_identity_exact:
        raise RuntimeError("Level audit requires accepted v4.2 vocoder identity")
    if (
        acoustic.config.mel_bins != vocoder.config.mel_bins
        or acoustic.config.sample_rate != vocoder.config.sample_rate
        or acoustic.config.hop_length != vocoder.config.hop_length
    ):
        raise RuntimeError("Acoustic/v4.2 feature contract mismatch")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    acoustic.cpu().eval()
    vocoder.cpu().eval()
    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        acoustic.config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("Not enough validation items for v4.2 level attribution")

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / OUTPUT_DIR_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "level_attribution_report.json"
    items: list[dict[str, object]] = []
    structural_checks: list[bool] = []

    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            item = dataset[dataset_index]
            base_item = dataset.base[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("Level audit requires cached target F0/voicing")

            teacher_frames = int(batch.mel_lengths[0])
            teacher_samples = teacher_frames * acoustic.config.hop_length
            if int(batch.durations[0].sum()) != teacher_frames:
                raise RuntimeError("Teacher duration grid does not match target mel")

            teacher_output = acoustic(batch.token_ids, batch.token_mask, batch.durations)
            teacher_prediction = prepare_speech_vocoder_conditioning(teacher_output)
            predicted_mel = teacher_prediction.mel[:, :teacher_frames]
            predicted_f0 = teacher_prediction.f0_hz[:, :teacher_frames]
            predicted_voiced = teacher_prediction.voiced[:, :teacher_frames]
            target_mel = batch.mel[:, :teacher_frames]
            target_f0 = batch.f0_hz[:, :teacher_frames]
            target_voiced = batch.voiced[:, :teacher_frames]
            teacher_shape_exact = (
                predicted_mel.shape[1] == teacher_frames
                and predicted_f0.shape[1] == teacher_frames
                and predicted_voiced.shape[1] == teacher_frames
            )
            if not teacher_shape_exact:
                raise RuntimeError("Teacher-grid acoustic prediction length mismatch")

            reference = _load_reference_waveform(
                Path(str(base_item["wav_path"])),
                sample_rate=acoustic.config.sample_rate,
                samples=teacher_samples,
            )
            active_mask = _reference_active_frame_mask(
                reference,
                acoustic.config.hop_length,
            )
            reference_metrics = _base_level_metrics(
                reference,
                sample_rate=acoustic.config.sample_rate,
                hop_length=acoustic.config.hop_length,
                active_mask=active_mask,
            )

            variant_inputs = {
                "v4_2_oracle": (target_mel, target_f0, target_voiced),
                "v4_2_predicted_mel_target_prosody": (
                    predicted_mel,
                    target_f0,
                    target_voiced,
                ),
                "v4_2_target_mel_predicted_prosody": (
                    target_mel,
                    predicted_f0,
                    predicted_voiced,
                ),
                "v4_2_predicted_mel_predicted_prosody_teacher_grid": (
                    predicted_mel,
                    predicted_f0,
                    predicted_voiced,
                ),
            }
            variant_reports: dict[str, object] = {}
            prefix = f"{audit_index:02d}"
            reference_path = output_dir / f"{prefix}_reference.wav"
            sf.write(
                str(reference_path),
                reference.numpy(),
                acoustic.config.sample_rate,
                subtype="PCM_16",
            )
            for variant_name in TEACHER_GRID_VARIANTS:
                mel, f0_hz, voiced = variant_inputs[variant_name]
                waveform_batch = vocoder(mel, f0_hz, voiced)
                shape_exact = tuple(waveform_batch.shape) == (1, teacher_samples)
                finite = bool(torch.isfinite(waveform_batch).all())
                structural_checks.extend((shape_exact, finite))
                if not shape_exact or not finite:
                    raise RuntimeError(f"Invalid v4.2 output in {variant_name}")
                waveform = waveform_batch[0].detach().cpu().to(torch.float32).contiguous()
                wav_path = output_dir / f"{prefix}_{variant_name}.wav"
                sf.write(
                    str(wav_path),
                    waveform.numpy(),
                    acoustic.config.sample_rate,
                    subtype="PCM_16",
                )
                variant_reports[variant_name] = {
                    "wav_path": str(wav_path),
                    **_teacher_grid_metrics(
                        waveform,
                        reference,
                        sample_rate=acoustic.config.sample_rate,
                        hop_length=acoustic.config.hop_length,
                        active_mask=active_mask,
                        reference_metrics=reference_metrics,
                    ),
                }

            full_output = acoustic(batch.token_ids, batch.token_mask)
            full_prediction = prepare_speech_vocoder_conditioning(full_output)
            full_frames = int(full_output["mel_lengths"][0])
            full_mel = full_prediction.mel[:, :full_frames]
            full_f0 = full_prediction.f0_hz[:, :full_frames]
            full_voiced = full_prediction.voiced[:, :full_frames]
            full_wave_batch = vocoder(full_mel, full_f0, full_voiced)
            full_samples = full_frames * acoustic.config.hop_length
            full_shape_exact = tuple(full_wave_batch.shape) == (1, full_samples)
            full_finite = bool(torch.isfinite(full_wave_batch).all())
            structural_checks.extend((full_shape_exact, full_finite))
            if not full_shape_exact or not full_finite:
                raise RuntimeError("Invalid full reference-free v4.2 output")
            full_wave = full_wave_batch[0].detach().cpu().to(torch.float32).contiguous()
            full_active_mask = _reference_active_frame_mask(
                full_wave,
                acoustic.config.hop_length,
            )
            full_path = output_dir / f"{prefix}_{FULL_ROUTE_VARIANT}.wav"
            sf.write(
                str(full_path),
                full_wave.numpy(),
                acoustic.config.sample_rate,
                subtype="PCM_16",
            )
            full_metrics = _base_level_metrics(
                full_wave,
                sample_rate=acoustic.config.sample_rate,
                hop_length=acoustic.config.hop_length,
                active_mask=full_active_mask,
            )
            full_metrics.update(
                {
                    "wav_path": str(full_path),
                    "predicted_mel_frames": full_frames,
                    "teacher_mel_frames": teacher_frames,
                    "predicted_to_teacher_duration_ratio": round(
                        full_frames / max(1, teacher_frames), 6
                    ),
                    "rms_relative_to_reference_db": round(
                        _db_ratio(
                            float(full_metrics["rms"]),
                            float(reference_metrics["rms"]),
                        ),
                        4,
                    ),
                    "peak_relative_to_reference_db": round(
                        _db_ratio(
                            float(full_metrics["peak"]),
                            float(reference_metrics["peak"]),
                        ),
                        4,
                    ),
                    "comparison_note": (
                        "Full-route duration differs; total level is comparable, but "
                        "teacher-grid active-RMS and presence comparisons are authoritative."
                    ),
                }
            )

            items.append(
                {
                    "audit_index": audit_index,
                    "dataset_index": dataset_index,
                    "utterance_id": str(item["utterance_id"]),
                    "text": str(item["text"]),
                    "teacher_mel_frames": teacher_frames,
                    "reference": {
                        "wav_path": str(reference_path),
                        **reference_metrics,
                    },
                    "teacher_grid_variants": variant_reports,
                    FULL_ROUTE_VARIANT: full_metrics,
                }
            )

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    structural_gate_pass = all(structural_checks) and checkpoints_unchanged
    report: dict[str, object] = {
        "status": "needs_review" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "acoustic_identity_exact": acoustic_identity_exact,
        "v4_2_identity_exact": vocoder_identity_exact,
        "grid_artifact_version": VOCODER_GRID_ARTIFACT_VERSION,
        "teacher_grid_variants": list(TEACHER_GRID_VARIANTS),
        "full_route_variant": FULL_ROUTE_VARIANT,
        "validation_item_count": len(items),
        "structural_gate_pass": structural_gate_pass,
        "checkpoints_present": {name: value is not None for name, value in before.items()},
        "checkpoints_unchanged": checkpoints_unchanged,
        "training_started": False,
        "training_authorized": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "predicted_duration_modified": False,
        "items": items,
        "output_dir": str(output_dir),
        "interpretation": (
            "Compare oracle with each teacher-grid substitution. A level loss appearing "
            "only with predicted mel assigns the primary cause to the acoustic mel path; "
            "a loss appearing only with predicted prosody assigns it to F0/voicing; a loss "
            "only when both are predicted indicates interaction. Overall RMS is not a pass "
            "unless active-speech RMS, above-300-Hz RMS, presence, and intelligibility remain."
        ),
        "next_gate": "inspect_v4_2_level_attribution_before_any_training_or_gain_change",
    }
    _atomic_json(report_path, report)
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_v4_2_level_attribution_audit(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
