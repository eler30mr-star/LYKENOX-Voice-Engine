"""Full-utterance oracle listening gate for trained LYKENOX vocoder v5.

V5 is the first non-sinusoidal corrective candidate after v4.x path attribution implicated
an explicit coherent periodic carrier.  Numerical training completion is not product
acceptance.  This audit renders the same three held-out utterances with identical target
mel + target F0 + target voicing and keeps the strongest historical v4.x references:

    reference -> v4.2 oracle -> v4.4 oracle -> v5 oracle

No training occurs, no checkpoint is mutated, no gain normalization is applied, and target
pitch / reference waveform are audit-only inputs rather than product-runtime requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import (
    VOCODER_GENERATOR_V4_2_ARCHITECTURE,
    VOCODER_GENERATOR_V4_4_ARCHITECTURE,
    VOCODER_GENERATOR_V5_ARCHITECTURE,
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
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance import (
    _above_300_fraction,
    _db_ratio,
    _load_reference_waveform,
)
from lykenox_voice_engine.training.speech_vocoder_v4_4_artifact import load_v4_4_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v5_artifact import load_v5_checkpoint


AUDIT_VERSION = "vocoder-v5-full-utterance-oracle-acceptance-v1"
VALIDATION_INDICES = (0, 1, 2)
DIAGNOSTIC_HARMONICS = 8


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_completed_training(report_path: Path) -> dict[str, object]:
    if not report_path.exists():
        raise FileNotFoundError(f"v5 training report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("v5 training report is not a JSON object")
    required = (
        report.get("status") == "pass",
        bool(report.get("persistent_training_complete", False)),
        report.get("architecture") == VOCODER_GENERATOR_V5_ARCHITECTURE,
        report.get("source_family") == "stochastic_glottal_pulse_noise",
        report.get("explicit_sinusoidal_carrier") is False,
        int(report.get("deterministic_harmonics", -1)) == 0,
    )
    if not all(required):
        raise RuntimeError("v5 oracle acceptance requires a completed passing v5 run")
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
        # Diagnostic only.  V4.4 proved this metric cannot grant perceptual acceptance.
        harmonic = target_relative_harmonic_exposure_loss(
            prediction_batch,
            reference_batch,
            f0_hz,
            sample_rate=generator.config.sample_rate,
            hop_length=generator.config.hop_length,
            harmonics=DIAGNOSTIC_HARMONICS,
        )

    wave = _wave_metrics(prediction, generator.config.sample_rate)
    reference_wave = _wave_metrics(reference, generator.config.sample_rate)
    return {
        **wave,
        "reconstruction_loss": round(float(reconstruction), 6),
        "envelope_loss": round(float(envelope.total), 6),
        "spectral_balance_loss": round(float(balance.loss), 6),
        "local_spectral_contrast_loss": round(float(contrast.loss), 6),
        "harmonic_exposure_diagnostic_loss": round(float(harmonic.loss), 6),
        "rms_relative_to_reference_db": round(
            _db_ratio(float(wave["rms"]), float(reference_wave["rms"])),
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


def run_v5_full_utterance_oracle_acceptance(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    v5_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_stochastic_glottal_filter_v5"
    )
    v5_path = v5_dir / "best.pt"
    training_report_path = v5_dir / "training_report.json"
    v4_4_path = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_dynamic_filter_hybrid_v4_4"
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
    for label, path in (("v5", v5_path), ("v4.4", v4_4_path), ("v4.2", v4_2_path)):
        if not path.exists():
            raise FileNotFoundError(f"{label} comparison checkpoint not found: {path}")

    hashes_before = {str(path): _sha256(path) for path in (v5_path, v4_4_path, v4_2_path)}
    training_report = _require_completed_training(training_report_path)

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_v5_full_utterance_oracle_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_utterance_oracle_report.json"

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    v5, _v5_discriminator, v5_payload = load_v5_checkpoint(v5_path)
    v4_4, _v4_4_discriminator, v4_4_payload = load_v4_4_checkpoint(v4_4_path)
    v4_2, _v4_2_discriminator, v4_2_payload = load_v4_2_checkpoint(v4_2_path)
    v5.cpu().eval()
    v4_4.cpu().eval()
    v4_2.cpu().eval()

    v5_identity_exact = (
        v5.architecture == VOCODER_GENERATOR_V5_ARCHITECTURE
        and v5_payload.get("generator_architecture") == VOCODER_GENERATOR_V5_ARCHITECTURE
        and v5.source_family == "stochastic_glottal_pulse_noise"
        and v5.explicit_sinusoidal_carrier is False
        and v5.deterministic_harmonics == 0
    )
    v4_4_identity_exact = (
        v4_4.architecture == VOCODER_GENERATOR_V4_4_ARCHITECTURE
        and v4_4_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_4_ARCHITECTURE
    )
    v4_2_identity_exact = (
        v4_2.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and v4_2_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not all((v5_identity_exact, v4_4_identity_exact, v4_2_identity_exact)):
        raise RuntimeError("v5 oracle checkpoint architecture identity mismatch")

    speech_config = LykenoxSpeechConfig()
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        speech_config,
        duration_root=find_clean_duration_root(root),
        include_pitch_targets=True,
    )
    if len(dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("Not enough held-out validation items for v5 oracle acceptance")

    envelope_loss = LogMelEnvelopeLoss(speech_config).cpu()
    metric_names = (
        "envelope_loss",
        "reconstruction_loss",
        "spectral_balance_loss",
        "local_spectral_contrast_loss",
        "harmonic_exposure_diagnostic_loss",
    )
    improvement_vs_v4_4 = {name: 0 for name in metric_names}
    improvement_vs_v4_2 = {name: 0 for name in metric_names}
    structural_checks: list[bool] = []
    items: list[dict[str, object]] = []

    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            item = dataset[dataset_index]
            base_item = dataset.base[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("v5 oracle acceptance requires cached target F0/voicing")

            frames = int(batch.mel_lengths[0])
            expected_samples = frames * speech_config.hop_length
            teacher_duration_sum = int(batch.durations[0].sum())
            if teacher_duration_sum != frames:
                raise RuntimeError("Teacher duration grid does not match full mel grid")

            mel = batch.mel[:, :frames]
            f0_hz = batch.f0_hz[:, :frames]
            voiced = batch.voiced[:, :frames]
            v4_2_wave_batch = v4_2(mel, f0_hz, voiced)
            v4_4_wave_batch = v4_4(mel, f0_hz, voiced)
            v5_wave_batch = v5(mel, f0_hz, voiced)
            waves = (v4_2_wave_batch, v4_4_wave_batch, v5_wave_batch)
            shapes_exact = all(tuple(wave.shape) == (1, expected_samples) for wave in waves)
            finite = all(bool(torch.isfinite(wave).all()) for wave in waves)
            structural_checks.extend((shapes_exact, finite))
            if not shapes_exact or not finite:
                raise RuntimeError("Full-utterance v5 oracle structural contract failed")

            reference = _load_reference_waveform(
                Path(str(base_item["wav_path"])),
                sample_rate=speech_config.sample_rate,
                samples=expected_samples,
            )
            v4_2_wave = v4_2_wave_batch[0].detach().cpu().to(torch.float32).contiguous()
            v4_4_wave = v4_4_wave_batch[0].detach().cpu().to(torch.float32).contiguous()
            v5_wave = v5_wave_batch[0].detach().cpu().to(torch.float32).contiguous()

            prefix = f"{audit_index:02d}"
            paths = {
                "reference": output_dir / f"{prefix}_reference.wav",
                "v4_2_oracle": output_dir / f"{prefix}_v4_2_oracle.wav",
                "v4_4_oracle": output_dir / f"{prefix}_v4_4_oracle.wav",
                "v5_oracle": output_dir / f"{prefix}_v5_oracle.wav",
            }
            for path, waveform in (
                (paths["reference"], reference),
                (paths["v4_2_oracle"], v4_2_wave),
                (paths["v4_4_oracle"], v4_4_wave),
                (paths["v5_oracle"], v5_wave),
            ):
                sf.write(str(path), waveform.numpy(), speech_config.sample_rate, subtype="PCM_16")

            reference_metrics = _wave_metrics(reference, speech_config.sample_rate)
            v4_2_metrics = _quality_metrics(v4_2, envelope_loss, v4_2_wave, reference, f0_hz)
            v4_4_metrics = _quality_metrics(v4_4, envelope_loss, v4_4_wave, reference, f0_hz)
            v5_metrics = _quality_metrics(v5, envelope_loss, v5_wave, reference, f0_hz)
            comparison_v4_4 = _comparison(v5_metrics, v4_4_metrics, metric_names)
            comparison_v4_2 = _comparison(v5_metrics, v4_2_metrics, metric_names)
            for name in metric_names:
                improvement_vs_v4_4[name] += int(comparison_v4_4[f"{name}_improved"])
                improvement_vs_v4_2[name] += int(comparison_v4_2[f"{name}_improved"])

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
                    "reference": {"wav_path": str(paths["reference"]), **reference_metrics},
                    "v4_2_oracle": {"wav_path": str(paths["v4_2_oracle"]), **v4_2_metrics},
                    "v4_4_oracle": {"wav_path": str(paths["v4_4_oracle"]), **v4_4_metrics},
                    "v5_oracle": {"wav_path": str(paths["v5_oracle"]), **v5_metrics},
                    "v5_vs_v4_4": comparison_v4_4,
                    "v5_vs_v4_2": comparison_v4_2,
                }
            )

    hashes_after = {str(path): _sha256(path) for path in (v5_path, v4_4_path, v4_2_path)}
    checkpoints_unchanged = hashes_before == hashes_after
    structural_gate_pass = all(structural_checks) and checkpoints_unchanged
    report: dict[str, object] = {
        "status": "needs_listening" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "v5_checkpoint": str(v5_path),
        "v4_4_comparison_checkpoint": str(v4_4_path),
        "v4_2_comparison_checkpoint": str(v4_2_path),
        "v5_identity_exact": v5_identity_exact,
        "v4_4_identity_exact": v4_4_identity_exact,
        "v4_2_identity_exact": v4_2_identity_exact,
        "source_family": v5.source_family,
        "explicit_sinusoidal_carrier": v5.explicit_sinusoidal_carrier,
        "deterministic_harmonics": v5.deterministic_harmonics,
        "persistent_training_status": training_report.get("status"),
        "persistent_training_complete": bool(training_report.get("persistent_training_complete", False)),
        "persistent_best_epoch": int(training_report.get("best_epoch", 0)),
        "persistent_global_step": int(training_report.get("global_step", 0)),
        "checkpoints_unchanged": checkpoints_unchanged,
        "structural_gate_pass": structural_gate_pass,
        "full_utterance_count": len(items),
        "v5_metric_improvement_counts_vs_v4_4": improvement_vs_v4_4,
        "v5_metric_improvement_counts_vs_v4_2": improvement_vs_v4_2,
        "harmonic_exposure_is_diagnostic_only": True,
        "oracle_conditioning_used_for_audit_only": True,
        "reference_audio_required_for_product_inference": False,
        "waveform_pitch_target_required_for_product_inference": False,
        "full_utterance_perceptual_acceptance": False,
        "items": items,
        "listening_protocol": (
            "For each numbered set listen in this order: reference, v4.2 oracle, v4.4 oracle, "
            "v5 oracle. Judge the radio-mistuned / metallic interference first, then voice "
            "body, intelligibility, consonants, formant detail, roughness/noise and useful level. "
            "Objective metrics do not grant acceptance."
        ),
        "acceptance_rule": (
            "Accept v5 only if the radio-mistuned / metallic carrier interference is absent or "
            "materially resolved across all three full held-out utterances and v5 preserves "
            "usable intelligibility, consonants/formants and level. Improvement over rejected "
            "v4.4 alone is insufficient; v5 must also be perceptually preferable to the stronger "
            "historical v4.2 baseline."
        ),
        "next_gate": (
            "listen_v5_full_utterance_oracle_sets_and_accept_or_revise_vocoder"
            if structural_gate_pass
            else "fix_v5_full_utterance_oracle_audit_before_listening"
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
            run_v5_full_utterance_oracle_acceptance(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
