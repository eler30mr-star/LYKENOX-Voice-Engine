"""Read-only full-utterance A/B gate for isolated acoustic mel refinement.

Compares the accepted acoustic frame-context-v2 base checkpoint against the one-epoch
``mel_decoder``-only refinement through the accepted v4.2 vocoder.  Teacher durations are
held fixed.  Two paired routes are written per utterance: target prosody for pure mel
isolation, and predicted prosody for acoustic/vocoder interaction.  The base and refined
models must produce bit-exact duration/F0/voicing outputs; only mel may differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.models.vocoder import VOCODER_GENERATOR_V4_2_ARCHITECTURE
from lykenox_voice_engine.runtime.speech_conditioning import prepare_speech_vocoder_conditioning
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    FRAME_CONTEXT_VERSION,
    TRAINER_CONTRACT_VERSION as BASE_TRAINER_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_artifact import (
    TRAINABLE_PREFIX,
    file_sha256,
    require_frozen_state_exact,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_audit import (
    _mel_bin_centers_hz,
    _mel_metrics,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_loss import (
    ACOUSTIC_MEL_FIDELITY_LOSS_VERSION,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_train import (
    HARD_EPOCH_LIMIT,
    TRAINER_CONTRACT_VERSION as REFINEMENT_TRAINER_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance import (
    _load_reference_waveform,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_level_attribution_audit import (
    _base_level_metrics,
    _reference_active_frame_mask,
    _teacher_grid_metrics,
)


AUDIT_VERSION = "acoustic-mel-fidelity-full-utterance-v4-2-ab-v1"
VALIDATION_INDICES = (0, 1, 2)
OUTPUT_DIR_NAME = "acoustic_mel_fidelity_v1_full_utterance_v4_2_ab"
VARIANTS = (
    "v4_2_oracle",
    "base_mel_target_prosody",
    "refined_mel_target_prosody",
    "base_mel_predicted_prosody",
    "refined_mel_predicted_prosody",
)


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


def _protected_paths(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
        "refined_best": training / "acoustic_mel_fidelity_v1" / "best.pt",
        "refined_last": training / "acoustic_mel_fidelity_v1" / "last.pt",
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v6_clarity_best": training / "vocoder_direct_waveform_v6_clarity_guard_v1" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
    }


def _require_refined_identity(
    refined_payload: dict[str, object],
    *,
    base_sha256: str,
) -> None:
    run = refined_payload.get("run_config")
    metadata = refined_payload.get("training_metadata")
    if not isinstance(run, dict) or not isinstance(metadata, dict):
        raise RuntimeError("refined checkpoint is missing run/training metadata")
    if run.get("trainer_contract_version") != REFINEMENT_TRAINER_CONTRACT_VERSION:
        raise RuntimeError("refined checkpoint trainer contract mismatch")
    if run.get("loss_version") != ACOUSTIC_MEL_FIDELITY_LOSS_VERSION:
        raise RuntimeError("refined checkpoint loss version mismatch")
    if run.get("base_checkpoint_sha256") != base_sha256:
        raise RuntimeError("refined checkpoint base identity mismatch")
    if run.get("trainable_parameter_prefix") != TRAINABLE_PREFIX:
        raise RuntimeError("refined checkpoint trainable prefix mismatch")
    if int(run.get("hard_epoch_limit", -1)) != HARD_EPOCH_LIMIT:
        raise RuntimeError("refined checkpoint hard epoch limit mismatch")
    history = metadata.get("history")
    if int(refined_payload.get("epoch", -1)) != 2 or int(refined_payload.get("next_item_offset", -1)) != 0:
        raise RuntimeError("refined checkpoint is not closed exactly after epoch 1")
    if not isinstance(history, list) or len(history) != 1 or int(metadata.get("best_epoch", -1)) != 1:
        raise RuntimeError("refined checkpoint did not preserve the one-epoch acceptance gate")


def run_acoustic_mel_fidelity_full_utterance_ab(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected_paths(root)
    required = ("acoustic_v2_best", "refined_best", "refined_last", "v4_2_best")
    missing = [name for name in required if not protected[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing required A/B checkpoints: {missing}")
    before = {name: _sha256(path) for name, path in protected.items()}

    base, base_payload = load_acoustic_prosody_checkpoint(protected["acoustic_v2_best"])
    refined, refined_payload = load_acoustic_prosody_checkpoint(protected["refined_best"])
    base_run = base_payload.get("run_config")
    if not isinstance(base_run, dict):
        raise RuntimeError("base acoustic checkpoint is missing run_config")
    base_identity_exact = (
        base.config.frame_context_version == FRAME_CONTEXT_VERSION
        and base_run.get("trainer_contract_version") == BASE_TRAINER_CONTRACT_VERSION
        and base_run.get("frame_context_version") == FRAME_CONTEXT_VERSION
    )
    if not base_identity_exact:
        raise RuntimeError("A/B requires accepted acoustic frame-context v2 base")
    base_sha = file_sha256(protected["acoustic_v2_best"])
    _require_refined_identity(refined_payload, base_sha256=base_sha)
    require_frozen_state_exact(refined, base)

    vocoder, _discriminator, vocoder_payload = load_v4_2_checkpoint(protected["v4_2_best"])
    v4_2_identity_exact = (
        vocoder.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and vocoder_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not v4_2_identity_exact:
        raise RuntimeError("A/B requires accepted v4.2 vocoder")
    if (
        base.config.sample_rate != vocoder.config.sample_rate
        or base.config.hop_length != vocoder.config.hop_length
        or base.config.mel_bins != vocoder.config.mel_bins
    ):
        raise RuntimeError("acoustic/v4.2 feature contract mismatch")

    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        base.config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("not enough validation items for full-utterance A/B")

    base.cpu().eval()
    refined.cpu().eval()
    vocoder.cpu().eval()
    centers = _mel_bin_centers_hz(
        sample_rate=base.config.sample_rate,
        n_fft=base.config.n_fft,
        mel_bins=base.config.mel_bins,
    )
    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_utterance_ab_report.json"
    items: list[dict[str, object]] = []
    structural_checks: list[bool] = []

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            item = dataset[dataset_index]
            base_item = dataset.base[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("A/B requires cached target F0/voicing")
            frames = int(batch.mel_lengths[0])
            samples = frames * base.config.hop_length
            if int(batch.durations[0].sum()) != frames:
                raise RuntimeError("teacher duration grid does not match target mel")

            base_output = base(batch.token_ids, batch.token_mask, batch.durations)
            refined_output = refined(batch.token_ids, batch.token_mask, batch.durations)
            exact_non_mel_outputs = (
                torch.equal(base_output["duration_prediction"], refined_output["duration_prediction"])
                and torch.equal(base_output["regulated_durations"], refined_output["regulated_durations"])
                and torch.equal(base_output["f0_prediction_hz"], refined_output["f0_prediction_hz"])
                and torch.equal(base_output["voicing_logits"], refined_output["voicing_logits"])
                and torch.equal(base_output["mel_lengths"], refined_output["mel_lengths"])
                and torch.equal(base_output["mel_mask"], refined_output["mel_mask"])
            )
            structural_checks.append(exact_non_mel_outputs)
            if not exact_non_mel_outputs:
                raise RuntimeError("refinement changed duration/prosody/frame outputs")

            base_conditioning = prepare_speech_vocoder_conditioning(base_output)
            refined_conditioning = prepare_speech_vocoder_conditioning(refined_output)
            predicted_prosody_exact = (
                torch.equal(base_conditioning.f0_hz[:, :frames], refined_conditioning.f0_hz[:, :frames])
                and torch.equal(base_conditioning.voiced[:, :frames], refined_conditioning.voiced[:, :frames])
            )
            structural_checks.append(predicted_prosody_exact)
            if not predicted_prosody_exact:
                raise RuntimeError("prepared predicted prosody differs between base/refined")

            target_mel = batch.mel[:, :frames]
            target_f0 = batch.f0_hz[:, :frames]
            target_voiced = batch.voiced[:, :frames]
            base_mel = base_conditioning.mel[:, :frames]
            refined_mel = refined_conditioning.mel[:, :frames]
            pred_f0 = base_conditioning.f0_hz[:, :frames]
            pred_voiced = base_conditioning.voiced[:, :frames]

            reference = _load_reference_waveform(
                Path(str(base_item["wav_path"])),
                sample_rate=base.config.sample_rate,
                samples=samples,
            )
            active_mask = _reference_active_frame_mask(reference, base.config.hop_length)
            reference_metrics = _base_level_metrics(
                reference,
                sample_rate=base.config.sample_rate,
                hop_length=base.config.hop_length,
                active_mask=active_mask,
            )

            variant_inputs = {
                "v4_2_oracle": (target_mel, target_f0, target_voiced),
                "base_mel_target_prosody": (base_mel, target_f0, target_voiced),
                "refined_mel_target_prosody": (refined_mel, target_f0, target_voiced),
                "base_mel_predicted_prosody": (base_mel, pred_f0, pred_voiced),
                "refined_mel_predicted_prosody": (refined_mel, pred_f0, pred_voiced),
            }
            prefix = f"{audit_index:02d}"
            reference_path = output_dir / f"{prefix}_reference.wav"
            sf.write(str(reference_path), reference.numpy(), base.config.sample_rate, subtype="PCM_16")
            variants: dict[str, object] = {}
            for variant_name in VARIANTS:
                mel, f0_hz, voiced = variant_inputs[variant_name]
                waveform_batch = vocoder(mel, f0_hz, voiced)
                shape_exact = tuple(waveform_batch.shape) == (1, samples)
                finite = bool(torch.isfinite(waveform_batch).all())
                structural_checks.extend((shape_exact, finite))
                if not shape_exact or not finite:
                    raise RuntimeError(f"invalid waveform for {variant_name}")
                wave = waveform_batch[0].detach().cpu().to(torch.float32).contiguous()
                wav_path = output_dir / f"{prefix}_{variant_name}.wav"
                sf.write(str(wav_path), wave.numpy(), base.config.sample_rate, subtype="PCM_16")
                variants[variant_name] = {
                    "wav_path": str(wav_path),
                    **_teacher_grid_metrics(
                        wave,
                        reference,
                        sample_rate=base.config.sample_rate,
                        hop_length=base.config.hop_length,
                        active_mask=active_mask,
                        reference_metrics=reference_metrics,
                    ),
                }

            base_mel_metrics = _mel_metrics(
                base_mel[0], target_mel[0], centers_hz=centers
            )
            refined_mel_metrics = _mel_metrics(
                refined_mel[0], target_mel[0], centers_hz=centers
            )
            items.append(
                {
                    "audit_index": audit_index,
                    "dataset_index": dataset_index,
                    "utterance_id": str(item["utterance_id"]),
                    "text": str(item["text"]),
                    "teacher_mel_frames": frames,
                    "reference": {"wav_path": str(reference_path), **reference_metrics},
                    "base_vs_refined_non_mel_outputs_exact": exact_non_mel_outputs,
                    "prepared_predicted_prosody_exact": predicted_prosody_exact,
                    "base_mel_metrics": {k: round(float(v), 6) for k, v in base_mel_metrics.items()},
                    "refined_mel_metrics": {k: round(float(v), 6) for k, v in refined_mel_metrics.items()},
                    "variants": variants,
                }
            )

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    structural_gate_pass = all(structural_checks) and checkpoints_unchanged
    report: dict[str, object] = {
        "status": "needs_listening" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "base_acoustic_identity_exact": base_identity_exact,
        "refined_checkpoint_identity_exact": True,
        "v4_2_identity_exact": v4_2_identity_exact,
        "teacher_duration_grid_used": True,
        "duration_outputs_exact": True,
        "f0_voicing_outputs_exact": True,
        "structural_gate_pass": structural_gate_pass,
        "checkpoints_unchanged": checkpoints_unchanged,
        "training_started": False,
        "epoch2_training_authorized": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "validation_item_count": len(items),
        "variants": list(VARIANTS),
        "items": items,
        "output_dir": str(output_dir),
        "listening_order": (
            "reference -> v4_2_oracle -> base_mel_target_prosody -> refined_mel_target_prosody; "
            "then base_mel_predicted_prosody -> refined_mel_predicted_prosody"
        ),
        "decision_rule": (
            "Refined mel must be at least as intelligible as base and must not add nasal, dark, "
            "metallic, whiny, grid-locked, or consonant-loss artifacts. Metrics may reject but "
            "cannot authorize epoch 2 without listening."
        ),
        "next_gate": "listen_base_vs_refined_full_utterances_before_any_epoch2",
    }
    _atomic_json(report_path, report)
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_acoustic_mel_fidelity_full_utterance_ab(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
