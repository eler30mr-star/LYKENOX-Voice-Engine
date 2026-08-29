"""Full-utterance oracle acceptance gate for trained LYKENOX vocoder v4.4.

V4.4 has completed its bounded persistent run numerically, but product acceptance remains
blocked until complete held-out oracle-conditioned utterances are listened to.  Because
v4.3 regressed perceptually relative to v4.2, this audit keeps both historical baselines:

    reference -> v4.2 oracle -> v4.3 oracle -> v4.4 oracle

All synthesized candidates use the same target mel + target F0 + target voicing and teacher
frame grid.  No training occurs, no checkpoint is mutated, no gain normalization is applied,
and reference audio / target pitch remain audit-only inputs rather than product-runtime
requirements.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import (
    VOCODER_GENERATOR_V4_2_ARCHITECTURE,
    VOCODER_GENERATOR_V4_3_ARCHITECTURE,
    VOCODER_GENERATOR_V4_4_ARCHITECTURE,
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
from lykenox_voice_engine.training.speech_vocoder_harmonic_exposure_loss import (
    target_relative_harmonic_exposure_loss,
)
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
from lykenox_voice_engine.training.speech_vocoder_v4_4_artifact import (
    load_v4_4_checkpoint,
)


AUDIT_VERSION = "vocoder-v4-4-full-utterance-oracle-acceptance-v1"
VALIDATION_INDICES = (0, 1, 2)
HARMONIC_EXPOSURE_AUDIT_HARMONICS = 8


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_completed_training(report_path: Path) -> dict[str, object]:
    if not report_path.exists():
        raise FileNotFoundError(f"v4.4 training report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("v4.4 training report is not a JSON object")
    if (
        report.get("status") != "pass"
        or not bool(report.get("persistent_training_complete", False))
        or report.get("architecture") != VOCODER_GENERATOR_V4_4_ARCHITECTURE
    ):
        raise RuntimeError(
            "v4.4 full-utterance acceptance requires a completed passing persistent run"
        )
    return report


def _quality_metrics(
    generator,
    envelope_loss: LogMelEnvelopeLoss,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    f0_hz: torch.Tensor,
) -> dict[str, object]:
    with torch.no_grad():
        prediction_batch = prediction.unsqueeze(0)
        reference_batch = reference.unsqueeze(0)
        reconstruction = multi_resolution_reconstruction_loss(
            prediction_batch,
            reference_batch,
        ).total
        envelope = envelope_loss(prediction_batch, reference_batch)
        balance = target_relative_spectral_balance_loss(
            prediction_batch,
            reference_batch,
            sample_rate=generator.config.sample_rate,
        )
        contrast = target_relative_local_spectral_contrast_loss(
            prediction_batch,
            reference_batch,
            hop_length=generator.config.hop_length,
        )
        harmonic = target_relative_harmonic_exposure_loss(
            prediction_batch,
            reference_batch,
            f0_hz,
            sample_rate=generator.config.sample_rate,
            hop_length=generator.config.hop_length,
            harmonics=HARMONIC_EXPOSURE_AUDIT_HARMONICS,
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
            float(contrast.prediction_mean_abs_contrast),
            6,
        ),
        "target_mean_abs_local_contrast": round(
            float(contrast.target_mean_abs_contrast),
            6,
        ),
        "harmonic_exposure_loss": round(float(harmonic.loss), 6),
        "prediction_mean_harmonic_exposure": round(
            float(harmonic.prediction_mean_exposure),
            6,
        ),
        "target_mean_harmonic_exposure": round(
            float(harmonic.target_mean_exposure),
            6,
        ),
        "harmonic_exposure_valid_fraction": round(float(harmonic.valid_fraction), 6),
        "rms_relative_to_reference_db": round(
            _db_ratio(prediction_rms, reference_rms),
            3,
        ),
        "above_300hz_fraction": round(_above_300_fraction(wave), 6),
        "reference_above_300hz_fraction": round(
            _above_300_fraction(reference_wave),
            6,
        ),
    }


def _comparison(
    candidate: dict[str, object],
    baseline: dict[str, object],
    metric_names: tuple[str, ...],
) -> dict[str, bool]:
    return {
        f"{name}_improved": float(candidate[name]) < float(baseline[name])
        for name in metric_names
    }


def run_v4_4_full_utterance_oracle_acceptance(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    v4_4_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_dynamic_filter_hybrid_v4_4"
    )
    v4_4_path = v4_4_dir / "best.pt"
    training_report_path = v4_4_dir / "training_report.json"
    v4_3_path = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_mel_filtered_carrier_v4_3"
        / "best.pt"
    )
    v4_2_path = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_source_filter_v4_2"
        / "best.pt"
    )
    if not v4_4_path.exists():
        raise FileNotFoundError(f"Persistent v4.4 best checkpoint not found: {v4_4_path}")
    if not v4_3_path.exists():
        raise FileNotFoundError(f"v4.3 comparison checkpoint not found: {v4_3_path}")
    if not v4_2_path.exists():
        raise FileNotFoundError(f"v4.2 comparison checkpoint not found: {v4_2_path}")

    training_report = _require_completed_training(training_report_path)
    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_v4_4_full_utterance_oracle_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_utterance_oracle_report.json"

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    v4_4, _v4_4_discriminator, v4_4_payload = load_v4_4_checkpoint(v4_4_path)
    v4_3, _v4_3_discriminator, v4_3_payload = load_v4_3_checkpoint(v4_3_path)
    v4_2, _v4_2_discriminator, v4_2_payload = load_v4_2_checkpoint(v4_2_path)
    v4_4.cpu().eval()
    v4_3.cpu().eval()
    v4_2.cpu().eval()

    v4_4_identity_exact = (
        v4_4.architecture == VOCODER_GENERATOR_V4_4_ARCHITECTURE
        and v4_4_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_4_ARCHITECTURE
    )
    v4_3_identity_exact = (
        v4_3.architecture == VOCODER_GENERATOR_V4_3_ARCHITECTURE
        and v4_3_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_3_ARCHITECTURE
    )
    v4_2_identity_exact = (
        v4_2.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and v4_2_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not all((v4_4_identity_exact, v4_3_identity_exact, v4_2_identity_exact)):
        raise RuntimeError("Oracle acceptance checkpoint architecture identity mismatch")

    speech_config = LykenoxSpeechConfig()
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        speech_config,
        duration_root=find_clean_duration_root(root),
        include_pitch_targets=True,
    )
    if len(dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("Not enough held-out validation items for oracle acceptance")

    envelope_loss = LogMelEnvelopeLoss(speech_config).cpu()
    metric_names = (
        "envelope_loss",
        "reconstruction_loss",
        "spectral_balance_loss",
        "local_spectral_contrast_loss",
        "harmonic_exposure_loss",
    )
    improvement_vs_v4_3 = {name: 0 for name in metric_names}
    improvement_vs_v4_2 = {name: 0 for name in metric_names}
    structural_checks: list[bool] = []
    items: list[dict[str, object]] = []

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
            v4_4_wave_batch = v4_4(mel, f0_hz, voiced)

            shapes_exact = all(
                tuple(wave.shape) == (1, expected_samples)
                for wave in (v4_2_wave_batch, v4_3_wave_batch, v4_4_wave_batch)
            )
            finite = all(
                bool(torch.isfinite(wave).all())
                for wave in (v4_2_wave_batch, v4_3_wave_batch, v4_4_wave_batch)
            )
            structural_checks.extend((shapes_exact, finite))
            if not shapes_exact or not finite:
                raise RuntimeError("Full-utterance v4.4 oracle structural contract failed")

            reference = _load_reference_waveform(
                Path(str(base_item["wav_path"])),
                sample_rate=speech_config.sample_rate,
                samples=expected_samples,
            )
            v4_2_wave = v4_2_wave_batch[0].detach().cpu().to(torch.float32).contiguous()
            v4_3_wave = v4_3_wave_batch[0].detach().cpu().to(torch.float32).contiguous()
            v4_4_wave = v4_4_wave_batch[0].detach().cpu().to(torch.float32).contiguous()

            prefix = f"{audit_index:02d}"
            reference_path = output_dir / f"{prefix}_reference.wav"
            v4_2_wav_path = output_dir / f"{prefix}_v4_2_oracle.wav"
            v4_3_wav_path = output_dir / f"{prefix}_v4_3_oracle.wav"
            v4_4_wav_path = output_dir / f"{prefix}_v4_4_oracle.wav"
            sf.write(
                str(reference_path),
                reference.numpy(),
                speech_config.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                str(v4_2_wav_path),
                v4_2_wave.numpy(),
                speech_config.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                str(v4_3_wav_path),
                v4_3_wave.numpy(),
                speech_config.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                str(v4_4_wav_path),
                v4_4_wave.numpy(),
                speech_config.sample_rate,
                subtype="PCM_16",
            )

            reference_metrics = _wave_metrics(reference, speech_config.sample_rate)
            v4_2_metrics = _quality_metrics(
                v4_2,
                envelope_loss,
                v4_2_wave,
                reference,
                f0_hz,
            )
            v4_3_metrics = _quality_metrics(
                v4_3,
                envelope_loss,
                v4_3_wave,
                reference,
                f0_hz,
            )
            v4_4_metrics = _quality_metrics(
                v4_4,
                envelope_loss,
                v4_4_wave,
                reference,
                f0_hz,
            )
            comparison_v4_3 = _comparison(v4_4_metrics, v4_3_metrics, metric_names)
            comparison_v4_2 = _comparison(v4_4_metrics, v4_2_metrics, metric_names)
            for name in metric_names:
                improvement_vs_v4_3[name] += int(comparison_v4_3[f"{name}_improved"])
                improvement_vs_v4_2[name] += int(comparison_v4_2[f"{name}_improved"])

            items.append(
                {
                    "audit_index": audit_index,
                    "dataset_index": dataset_index,
                    "utterance_id": str(item["utterance_id"]),
                    "text": str(item["text"]),
                    "teacher_mel_frames": frames,
                    "teacher_duration_sum_frames": teacher_duration_sum,
                    "duration_seconds": round(
                        expected_samples / speech_config.sample_rate,
                        4,
                    ),
                    "target_voiced_fraction": round(float(voiced.mean()), 6),
                    "reference": {
                        "wav_path": str(reference_path),
                        **reference_metrics,
                    },
                    "v4_2_oracle": {
                        "wav_path": str(v4_2_wav_path),
                        **v4_2_metrics,
                    },
                    "v4_3_oracle": {
                        "wav_path": str(v4_3_wav_path),
                        **v4_3_metrics,
                    },
                    "v4_4_oracle": {
                        "wav_path": str(v4_4_wav_path),
                        **v4_4_metrics,
                    },
                    "v4_4_vs_v4_3": comparison_v4_3,
                    "v4_4_vs_v4_2": comparison_v4_2,
                }
            )

    structural_gate_pass = all(structural_checks)
    report: dict[str, object] = {
        "status": "needs_listening" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "v4_4_checkpoint": str(v4_4_path),
        "v4_3_comparison_checkpoint": str(v4_3_path),
        "v4_2_comparison_checkpoint": str(v4_2_path),
        "v4_4_identity_exact": v4_4_identity_exact,
        "v4_3_identity_exact": v4_3_identity_exact,
        "v4_2_identity_exact": v4_2_identity_exact,
        "persistent_training_status": training_report.get("status"),
        "persistent_training_complete": bool(
            training_report.get("persistent_training_complete", False)
        ),
        "persistent_best_epoch": int(training_report.get("best_epoch", 0)),
        "persistent_global_step": int(training_report.get("global_step", 0)),
        "structural_gate_pass": structural_gate_pass,
        "full_utterance_count": len(items),
        "harmonic_exposure_audit_harmonics": HARMONIC_EXPOSURE_AUDIT_HARMONICS,
        "v4_4_metric_improvement_counts_vs_v4_3": {
            name: improvement_vs_v4_3[name] for name in metric_names
        },
        "v4_4_metric_improvement_counts_vs_v4_2": {
            name: improvement_vs_v4_2[name] for name in metric_names
        },
        "oracle_conditioning_used_for_audit_only": True,
        "reference_audio_required_for_product_inference": False,
        "waveform_pitch_target_required_for_product_inference": False,
        "full_utterance_perceptual_acceptance": False,
        "items": items,
        "listening_protocol": (
            "For each numbered item listen in this order: reference, v4.2 oracle, v4.3 "
            "oracle, v4.4 oracle. Judge first the radio-mistuned / metallic interference "
            "attached to voiced speech, then word and consonant intelligibility, natural "
            "formant detail, aperiodic consonant detail, and apparent output level. Do not "
            "grant acceptance from objective losses alone."
        ),
        "acceptance_rule": (
            "Accept the v4.4 waveform stage only if the radio-mistuned / metallic carrier "
            "interference is absent or materially resolved across all three complete held-out "
            "oracle utterances, and v4.4 is perceptually no worse than the stronger historical "
            "v4.2 baseline while preserving intelligibility, consonants/formants and usable "
            "level. Improvement versus rejected v4.3 alone is insufficient."
        ),
        "next_gate": (
            "listen_v4_4_full_utterance_oracle_sets_and_accept_or_revise_vocoder"
            if structural_gate_pass
            else "fix_v4_4_full_utterance_oracle_audit_before_listening"
        ),
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_v4_4_full_utterance_oracle_acceptance(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
