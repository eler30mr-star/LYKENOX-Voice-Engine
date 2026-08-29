"""Full-utterance oracle acceptance gate for trained LYKENOX vocoder v4.3.

V4.3 numerically completed persistent training, but product acceptance remains blocked until
complete held-out oracle-conditioned utterances are listened to.  This audit compares the
last corrective baseline (v4.2) against v4.3 on the same three fixed validation items using
only target mel + target F0 + target voicing.

No training occurs, no checkpoint is mutated, no gain normalization is applied, and the
real/reference waveform plus target pitch are audit-only inputs rather than product-runtime
requirements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import (
    VOCODER_GENERATOR_V4_2_ARCHITECTURE,
    VOCODER_GENERATOR_V4_3_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import (
    _wave_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import LogMelEnvelopeLoss
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
    target_relative_local_spectral_contrast_loss,
)
from lykenox_voice_engine.training.speech_vocoder_losses import (
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import (
    load_v4_2_checkpoint,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance import (
    _above_300_fraction,
    _db_ratio,
    _load_reference_waveform,
)
from lykenox_voice_engine.training.speech_vocoder_v4_3_artifact import (
    load_v4_3_checkpoint,
)


AUDIT_VERSION = "vocoder-v4-3-full-utterance-oracle-acceptance-v1"
VALIDATION_INDICES = (0, 1, 2)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _require_completed_training(report_path: Path) -> dict[str, object]:
    if not report_path.exists():
        raise FileNotFoundError(f"v4.3 training report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("v4.3 training report is not a JSON object")
    if (
        report.get("status") != "pass"
        or not bool(report.get("persistent_training_complete", False))
        or report.get("architecture") != VOCODER_GENERATOR_V4_3_ARCHITECTURE
    ):
        raise RuntimeError(
            "v4.3 full-utterance acceptance requires a completed passing persistent run"
        )
    return report


def _quality_metrics(
    generator,
    envelope_loss: LogMelEnvelopeLoss,
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    with torch.no_grad():
        reconstruction = multi_resolution_reconstruction_loss(
            prediction.unsqueeze(0),
            reference.unsqueeze(0),
        ).total
        envelope = envelope_loss(
            prediction.unsqueeze(0),
            reference.unsqueeze(0),
        )
        balance = target_relative_spectral_balance_loss(
            prediction.unsqueeze(0),
            reference.unsqueeze(0),
            sample_rate=generator.config.sample_rate,
        )
        contrast = target_relative_local_spectral_contrast_loss(
            prediction.unsqueeze(0),
            reference.unsqueeze(0),
            hop_length=generator.config.hop_length,
        )
    wave = _wave_metrics(prediction, generator.config.sample_rate)
    reference_wave = _wave_metrics(reference, generator.config.sample_rate)
    prediction_rms = float(wave["rms"])
    reference_rms = float(reference_wave["rms"])
    return {
        **wave,
        "reconstruction_loss": round(float(reconstruction), 6),
        "envelope_loss": round(float(envelope.total), 6),
        "envelope_level_l1": round(float(envelope.log_mel_l1), 6),
        "envelope_spectral_slope_l1": round(float(envelope.spectral_slope_l1), 6),
        "envelope_temporal_delta_l1": round(float(envelope.temporal_delta_l1), 6),
        "spectral_balance_loss": round(float(balance.loss), 6),
        "local_spectral_contrast_loss": round(float(contrast.loss), 6),
        "prediction_mean_abs_local_contrast": round(
            float(contrast.prediction_mean_abs_contrast), 6
        ),
        "target_mean_abs_local_contrast": round(
            float(contrast.target_mean_abs_contrast), 6
        ),
        "rms_relative_to_reference_db": round(
            _db_ratio(prediction_rms, reference_rms), 3
        ),
        "above_300hz_fraction": round(_above_300_fraction(wave), 6),
        "reference_above_300hz_fraction": round(
            _above_300_fraction(reference_wave), 6
        ),
    }


def run_v4_3_full_utterance_oracle_acceptance(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    v4_3_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_mel_filtered_carrier_v4_3"
    )
    v4_3_path = v4_3_dir / "best.pt"
    training_report_path = v4_3_dir / "training_report.json"
    v4_2_path = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_source_filter_v4_2"
        / "best.pt"
    )
    if not v4_3_path.exists():
        raise FileNotFoundError(f"Persistent v4.3 best checkpoint not found: {v4_3_path}")
    if not v4_2_path.exists():
        raise FileNotFoundError(f"v4.2 comparison checkpoint not found: {v4_2_path}")

    training_report = _require_completed_training(training_report_path)
    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_v4_3_full_utterance_oracle_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_utterance_oracle_report.json"

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    v4_3, _v4_3_discriminator, v4_3_payload = load_v4_3_checkpoint(v4_3_path)
    v4_2, _v4_2_discriminator, v4_2_payload = load_v4_2_checkpoint(v4_2_path)
    v4_3.cpu().eval()
    v4_2.cpu().eval()

    v4_3_identity_exact = (
        v4_3.architecture == VOCODER_GENERATOR_V4_3_ARCHITECTURE
        and v4_3_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_3_ARCHITECTURE
    )
    v4_2_identity_exact = (
        v4_2.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and v4_2_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not v4_3_identity_exact or not v4_2_identity_exact:
        raise RuntimeError("Oracle acceptance checkpoint architecture identity mismatch")

    speech_config = LykenoxSpeechConfig()
    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        speech_config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("Not enough held-out validation items for oracle acceptance")

    envelope_loss = LogMelEnvelopeLoss(speech_config).cpu()
    items: list[dict[str, object]] = []
    structural_checks: list[bool] = []
    metric_names = (
        "envelope_loss",
        "reconstruction_loss",
        "spectral_balance_loss",
        "local_spectral_contrast_loss",
    )
    improvement_counts = {name: 0 for name in metric_names}

    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            item = dataset[dataset_index]
            base_item = dataset.base[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("Oracle acceptance requires cached target F0/voicing")

            frames = int(batch.mel_lengths[0])
            expected_samples = frames * speech_config.hop_length
            teacher_duration_sum = int(batch.durations[0].sum())
            if teacher_duration_sum != frames:
                raise RuntimeError("Teacher duration grid does not match full mel grid")

            mel = batch.mel[:, :frames]
            f0_hz = batch.f0_hz[:, :frames]
            voiced = batch.voiced[:, :frames]
            v4_2_wave_batch = v4_2(mel, f0_hz, voiced)
            v4_3_wave_batch = v4_3(mel, f0_hz, voiced)
            shapes_exact = (
                tuple(v4_2_wave_batch.shape) == (1, expected_samples)
                and tuple(v4_3_wave_batch.shape) == (1, expected_samples)
            )
            finite = bool(torch.isfinite(v4_2_wave_batch).all()) and bool(
                torch.isfinite(v4_3_wave_batch).all()
            )
            structural_checks.extend((shapes_exact, finite))
            if not shapes_exact or not finite:
                raise RuntimeError("Full-utterance v4.3 oracle structural contract failed")

            reference = _load_reference_waveform(
                Path(str(base_item["wav_path"])),
                sample_rate=speech_config.sample_rate,
                samples=expected_samples,
            )
            v4_2_wave = v4_2_wave_batch[0].detach().cpu().to(torch.float32).contiguous()
            v4_3_wave = v4_3_wave_batch[0].detach().cpu().to(torch.float32).contiguous()

            prefix = f"{audit_index:02d}"
            reference_path = output_dir / f"{prefix}_reference.wav"
            v4_2_wav_path = output_dir / f"{prefix}_v4_2_oracle.wav"
            v4_3_wav_path = output_dir / f"{prefix}_v4_3_oracle.wav"
            sf.write(str(reference_path), reference.numpy(), speech_config.sample_rate, subtype="PCM_16")
            sf.write(str(v4_2_wav_path), v4_2_wave.numpy(), speech_config.sample_rate, subtype="PCM_16")
            sf.write(str(v4_3_wav_path), v4_3_wave.numpy(), speech_config.sample_rate, subtype="PCM_16")

            reference_metrics = _wave_metrics(reference, speech_config.sample_rate)
            v4_2_metrics = _quality_metrics(v4_2, envelope_loss, v4_2_wave, reference)
            v4_3_metrics = _quality_metrics(v4_3, envelope_loss, v4_3_wave, reference)
            comparison: dict[str, bool] = {}
            for metric_name in metric_names:
                improved = float(v4_3_metrics[metric_name]) < float(v4_2_metrics[metric_name])
                comparison[f"{metric_name}_improved"] = improved
                improvement_counts[metric_name] += int(improved)

            items.append(
                {
                    "audit_index": audit_index,
                    "dataset_index": dataset_index,
                    "utterance_id": str(item["utterance_id"]),
                    "text": str(item["text"]),
                    "teacher_mel_frames": frames,
                    "teacher_duration_sum_frames": teacher_duration_sum,
                    "duration_seconds": round(expected_samples / speech_config.sample_rate, 4),
                    "target_voiced_fraction": round(float(voiced.mean()), 6),
                    "reference": {"wav_path": str(reference_path), **reference_metrics},
                    "v4_2_oracle": {"wav_path": str(v4_2_wav_path), **v4_2_metrics},
                    "v4_3_oracle": {"wav_path": str(v4_3_wav_path), **v4_3_metrics},
                    "v4_3_vs_v4_2": comparison,
                }
            )

    structural_gate_pass = all(structural_checks)
    report: dict[str, object] = {
        "status": "needs_listening" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "v4_3_checkpoint": str(v4_3_path),
        "v4_2_comparison_checkpoint": str(v4_2_path),
        "v4_3_identity_exact": v4_3_identity_exact,
        "v4_2_identity_exact": v4_2_identity_exact,
        "persistent_training_status": training_report.get("status"),
        "persistent_training_complete": bool(training_report.get("persistent_training_complete", False)),
        "persistent_best_epoch": int(training_report.get("best_epoch", 0)),
        "persistent_global_step": int(training_report.get("global_step", 0)),
        "structural_gate_pass": structural_gate_pass,
        "full_utterance_count": len(items),
        "v4_3_metric_improvement_counts_vs_v4_2": {
            name: improvement_counts[name] for name in metric_names
        },
        "oracle_conditioning_used_for_audit_only": True,
        "reference_audio_required_for_product_inference": False,
        "waveform_pitch_target_required_for_product_inference": False,
        "full_utterance_perceptual_acceptance": False,
        "items": items,
        "listening_protocol": (
            "For each numbered item listen in this order: reference, v4.2 oracle, v4.3 oracle. "
            "Judge the residual metallic/insect-like chillido first, then consonant/word "
            "intelligibility, natural formant detail, and apparent output level. Objective "
            "loss improvement does not grant acceptance by itself."
        ),
        "acceptance_rule": (
            "Accept the v4.3 waveform stage only if the residual v4.2 chillido is absent or "
            "materially resolved across all three complete held-out oracle utterances while "
            "intelligibility and level remain usable. Otherwise assign the remaining failure "
            "before any further training."
        ),
        "next_gate": (
            "listen_v4_3_full_utterance_oracle_pairs_and_accept_or_revise_vocoder"
            if structural_gate_pass
            else "fix_v4_3_full_utterance_oracle_audit_before_listening"
        ),
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_v4_3_full_utterance_oracle_acceptance(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
