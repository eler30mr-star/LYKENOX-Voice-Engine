"""Read-only full-utterance A/B gate for the one-epoch mel residual postnet.

The accepted acoustic-v2 base and vocoder-v4.2 remain immutable. Three fixed held-out
utterances are rendered on teacher durations. Each utterance contains two controlled
base/postnet pairs: target prosody isolates the postnet's mel effect, while identical
predicted prosody tests the real acoustic-to-vocoder interaction. No gain, EQ, denoising,
training, duration modification, or second postnet epoch is authorized here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.models.speech.mel_postnet import MEL_POSTNET_ARCHITECTURE_V1
from lykenox_voice_engine.models.vocoder import VOCODER_GENERATOR_V4_2_ARCHITECTURE
from lykenox_voice_engine.runtime.speech_conditioning import prepare_speech_vocoder_conditioning
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_audit import (
    _mel_bin_centers_hz,
    _mel_metrics,
)
from lykenox_voice_engine.training.speech_acoustic_mel_postnet_artifact import (
    base_checkpoint_path,
    file_sha256,
    load_mel_postnet_checkpoint,
    postnet_output_dir,
)
from lykenox_voice_engine.training.speech_acoustic_mel_postnet_train import (
    HARD_EPOCH_LIMIT,
    TRAINER_CONTRACT_VERSION,
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


AUDIT_VERSION = "acoustic-mel-postnet-full-utterance-v4-2-ab-v1"
VALIDATION_INDICES = (0, 1, 2)
OUTPUT_DIR_NAME = "acoustic_mel_postnet_v1_full_utterance_v4_2_ab"
VARIANTS = (
    "v4_2_oracle",
    "base_mel_target_prosody",
    "postnet_mel_target_prosody",
    "base_mel_predicted_prosody",
    "postnet_mel_predicted_prosody",
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
        "postnet_best": training / "acoustic_mel_postnet_v1" / "best.pt",
        "postnet_last": training / "acoustic_mel_postnet_v1" / "last.pt",
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "rejected_mel_best": training / "acoustic_mel_fidelity_v1" / "best.pt",
        "rejected_mel_last": training / "acoustic_mel_fidelity_v1" / "last.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
    }


def _require_postnet_epoch1_identity(payload: dict[str, object], *, base_sha256: str) -> None:
    run_config = payload.get("run_config")
    metadata = payload.get("training_metadata")
    if not isinstance(run_config, dict) or not isinstance(metadata, dict):
        raise RuntimeError("postnet checkpoint is missing run/training metadata")
    if payload.get("architecture") != MEL_POSTNET_ARCHITECTURE_V1:
        raise RuntimeError("postnet checkpoint architecture mismatch")
    if payload.get("base_checkpoint_sha256") != base_sha256:
        raise RuntimeError("postnet checkpoint base SHA mismatch")
    if run_config.get("trainer_contract_version") != TRAINER_CONTRACT_VERSION:
        raise RuntimeError("postnet checkpoint trainer identity mismatch")
    if int(run_config.get("hard_epoch_limit", -1)) != HARD_EPOCH_LIMIT:
        raise RuntimeError("postnet checkpoint hard epoch gate mismatch")
    if int(payload.get("epoch", -1)) != 2 or int(payload.get("next_item_offset", -1)) != 0:
        raise RuntimeError("postnet checkpoint is not closed exactly after epoch 1")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != 1:
        raise RuntimeError("postnet checkpoint must contain exactly one completed epoch")
    if int(metadata.get("best_epoch", -1)) != 1:
        raise RuntimeError("postnet best checkpoint was not selected from epoch 1")


def run_mel_postnet_full_utterance_ab(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected_paths(root)
    required = ("acoustic_v2_best", "postnet_best", "postnet_last", "v4_2_best")
    missing = [name for name in required if not protected[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing required postnet A/B checkpoints: {missing}")
    before = {name: _sha256(path) for name, path in protected.items()}

    base_path = base_checkpoint_path(root)
    if base_path != protected["acoustic_v2_best"]:
        raise RuntimeError("postnet A/B base checkpoint path contract changed")
    base_sha = file_sha256(base_path)
    candidate, candidate_payload = load_mel_postnet_checkpoint(
        protected["postnet_best"],
        base_checkpoint=base_path,
    )
    _require_postnet_epoch1_identity(candidate_payload, base_sha256=base_sha)
    base = candidate.base_model
    base.cpu().eval()
    candidate.cpu().eval()
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise RuntimeError("accepted acoustic base became trainable during A/B")

    vocoder, _discriminator, vocoder_payload = load_v4_2_checkpoint(protected["v4_2_best"])
    vocoder.cpu().eval()
    v4_2_identity_exact = (
        vocoder.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and vocoder_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not v4_2_identity_exact:
        raise RuntimeError("postnet A/B requires accepted vocoder v4.2")
    if (
        candidate.config.sample_rate != vocoder.config.sample_rate
        or candidate.config.hop_length != vocoder.config.hop_length
        or candidate.config.mel_bins != vocoder.config.mel_bins
    ):
        raise RuntimeError("postnet/v4.2 feature contract mismatch")

    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        candidate.config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("not enough held-out validation items for postnet A/B")

    centers = _mel_bin_centers_hz(
        sample_rate=candidate.config.sample_rate,
        n_fft=candidate.config.n_fft,
        mel_bins=candidate.config.mel_bins,
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
                raise RuntimeError("postnet A/B requires cached target F0/voicing")
            frames = int(batch.mel_lengths[0])
            samples = frames * candidate.config.hop_length
            if int(batch.durations[0].sum()) != frames:
                raise RuntimeError("teacher durations do not match target mel length")

            base_output = base(batch.token_ids, batch.token_mask, batch.durations)
            postnet_output = candidate(batch.token_ids, batch.token_mask, batch.durations)
            exact_non_mel_outputs = all(
                torch.equal(base_output[key], postnet_output[key])
                for key in (
                    "duration_prediction",
                    "regulated_durations",
                    "f0_prediction_hz",
                    "voicing_logits",
                    "mel_lengths",
                    "mel_mask",
                )
            )
            base_mel_identity = torch.equal(base_output["mel"], postnet_output["base_mel"])
            structural_checks.extend((exact_non_mel_outputs, base_mel_identity))
            if not exact_non_mel_outputs or not base_mel_identity:
                raise RuntimeError("postnet changed non-mel outputs or lost base mel identity")

            base_conditioning = prepare_speech_vocoder_conditioning(base_output)
            postnet_conditioning = prepare_speech_vocoder_conditioning(postnet_output)
            predicted_prosody_exact = (
                torch.equal(
                    base_conditioning.f0_hz[:, :frames],
                    postnet_conditioning.f0_hz[:, :frames],
                )
                and torch.equal(
                    base_conditioning.voiced[:, :frames],
                    postnet_conditioning.voiced[:, :frames],
                )
            )
            structural_checks.append(predicted_prosody_exact)
            if not predicted_prosody_exact:
                raise RuntimeError("prepared predicted prosody changed through postnet")

            target_mel = batch.mel[:, :frames]
            target_f0 = batch.f0_hz[:, :frames]
            target_voiced = batch.voiced[:, :frames]
            base_mel = base_conditioning.mel[:, :frames]
            postnet_mel = postnet_conditioning.mel[:, :frames]
            pred_f0 = base_conditioning.f0_hz[:, :frames]
            pred_voiced = base_conditioning.voiced[:, :frames]

            reference = _load_reference_waveform(
                Path(str(base_item["wav_path"])),
                sample_rate=candidate.config.sample_rate,
                samples=samples,
            )
            active_mask = _reference_active_frame_mask(
                reference,
                candidate.config.hop_length,
            )
            reference_metrics = _base_level_metrics(
                reference,
                sample_rate=candidate.config.sample_rate,
                hop_length=candidate.config.hop_length,
                active_mask=active_mask,
            )
            variant_inputs = {
                "v4_2_oracle": (target_mel, target_f0, target_voiced),
                "base_mel_target_prosody": (base_mel, target_f0, target_voiced),
                "postnet_mel_target_prosody": (postnet_mel, target_f0, target_voiced),
                "base_mel_predicted_prosody": (base_mel, pred_f0, pred_voiced),
                "postnet_mel_predicted_prosody": (postnet_mel, pred_f0, pred_voiced),
            }

            prefix = f"{audit_index:02d}"
            reference_path = output_dir / f"{prefix}_reference.wav"
            sf.write(
                str(reference_path),
                reference.numpy(),
                candidate.config.sample_rate,
                subtype="PCM_16",
            )
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
                sf.write(
                    str(wav_path),
                    wave.numpy(),
                    candidate.config.sample_rate,
                    subtype="PCM_16",
                )
                variants[variant_name] = {
                    "wav_path": str(wav_path),
                    **_teacher_grid_metrics(
                        wave,
                        reference,
                        sample_rate=candidate.config.sample_rate,
                        hop_length=candidate.config.hop_length,
                        active_mask=active_mask,
                        reference_metrics=reference_metrics,
                    ),
                }

            items.append(
                {
                    "audit_index": audit_index,
                    "dataset_index": dataset_index,
                    "utterance_id": str(item["utterance_id"]),
                    "text": str(item["text"]),
                    "teacher_mel_frames": frames,
                    "reference": {"wav_path": str(reference_path), **reference_metrics},
                    "non_mel_outputs_exact": exact_non_mel_outputs,
                    "base_mel_identity_exact": base_mel_identity,
                    "prepared_predicted_prosody_exact": predicted_prosody_exact,
                    "base_mel_metrics": {
                        key: round(float(value), 6)
                        for key, value in _mel_metrics(
                            base_mel[0], target_mel[0], centers_hz=centers
                        ).items()
                    },
                    "postnet_mel_metrics": {
                        key: round(float(value), 6)
                        for key, value in _mel_metrics(
                            postnet_mel[0], target_mel[0], centers_hz=centers
                        ).items()
                    },
                    "variants": variants,
                }
            )

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    structural_gate_pass = all(structural_checks) and checkpoints_unchanged
    report: dict[str, object] = {
        "status": "needs_listening" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "postnet_architecture": MEL_POSTNET_ARCHITECTURE_V1,
        "postnet_checkpoint_identity_exact": True,
        "base_acoustic_identity_exact": True,
        "v4_2_identity_exact": v4_2_identity_exact,
        "teacher_duration_grid_used": True,
        "duration_outputs_exact": True,
        "f0_voicing_outputs_exact": True,
        "base_mel_identity_exact": True,
        "structural_gate_pass": structural_gate_pass,
        "checkpoints_unchanged": checkpoints_unchanged,
        "training_started": False,
        "epoch2_training_authorized": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "predicted_duration_modified": False,
        "validation_item_count": len(items),
        "variants": list(VARIANTS),
        "items": items,
        "output_dir": str(output_dir),
        "listening_order": (
            "reference -> v4_2_oracle -> base_mel_target_prosody -> "
            "postnet_mel_target_prosody -> base_mel_predicted_prosody -> "
            "postnet_mel_predicted_prosody"
        ),
        "acceptance_rule": (
            "Postnet is acceptable only if full held-out speech is audibly more intelligible "
            "or clearer than base without new metallic, nasal, noisy, grid-like, or harsh artifacts. "
            "Metrics may reject but cannot accept. Ambiguous listening does not authorize epoch 2."
        ),
        "next_gate": "listen_postnet_vs_base_full_utterances_before_any_epoch2",
    }
    _atomic_json(report_path, report)
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_mel_postnet_full_utterance_ab(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
