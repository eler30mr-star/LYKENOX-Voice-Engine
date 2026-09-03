"""Controlled retrain of Continuous Residual Source V2 under pitch-conditioning V2.

This path changes exactly one functional variable relative to the accepted V2 training recipe:
conditioning semantics. The model class, architecture, Step-3f targets, loss function, renderer,
training budget, teacher-forcing schedule and optimizer family remain unchanged.

Legacy conditioning columns ``f0_hz / voiced / periodicity`` are replaced, in the same three input
slots, by ``f0_track_hz / energy_confidence / periodic_strength`` from the owned deterministic
pitch-conditioning V2 contract. The run warm-starts the complete model state from the existing owned
Continuous Source V2 best checkpoint. It writes to a separate run directory and never overwrites the
historical V2 baseline.

No external model/weight/service, codebook, post-hoc gain normalization, EQ, denoise or duration
modification is used. Metrics remain rejection-only. Policy: LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v2 import (
    CONTINUOUS_SOURCE_ARCHITECTURE_V2,
    HOP_LENGTH,
    LykenoxContinuousResidualSourceV2,
)
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import (
    PITCH_CONDITIONING_V2,
    PitchConditioningV2,
    extract_pitch_conditioning_v2,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import (
    _load_or_build_target,
    _segment,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v2 import (
    CHECKPOINT_SCHEMA_VERSION as BASE_V2_CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_EVAL_EVERY,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_UPDATES,
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_SEED,
    DEFAULT_TRAIN_ITEMS,
    DEFAULT_VAL_ITEMS,
    DEFAULT_WEIGHT_DECAY,
    POLICY_ID,
    _deterministic_crop,
    _loss_terms,
    _teacher_ratio,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    OwnedVocoderUtterance,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    RENDERER_VERSION,
    SAMPLE_RATE,
)


TRAINER_VERSION = "owned-continuous-residual-source-v2-pitch-conditioning-v2"
CHECKPOINT_SCHEMA_VERSION = "owned-continuous-residual-source-v2-pitch-conditioning-v2-checkpoint-v1"
RUN_DIR_NAME = "continuous_residual_source_v2_pitch_conditioning_v2"
BASE_V2_RUN_DIR_NAME = "continuous_residual_source_v2"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _extract_conditioning(utterance: OwnedVocoderUtterance) -> PitchConditioningV2:
    return extract_pitch_conditioning_v2(
        utterance.waveform.cpu().to(torch.float32),
        frame_count=int(utterance.mel_frames),
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        frame_length=int(PITCH_CONFIG["frame_length"]),
        min_f0_hz=float(PITCH_CONFIG["min_f0_hz"]),
        max_f0_hz=float(PITCH_CONFIG["max_f0_hz"]),
        anchor_periodicity_threshold=float(PITCH_CONFIG["voiced_periodicity_threshold"]),
        anchor_rms_fraction=float(PITCH_CONFIG["voiced_rms_fraction"]),
    )


def _conditioning_map(utterances: list[OwnedVocoderUtterance]) -> dict[str, PitchConditioningV2]:
    result: dict[str, PitchConditioningV2] = {}
    for utterance in utterances:
        result[utterance.utterance_id] = _extract_conditioning(utterance)
    return result


def _segment_with_conditioning_v2(
    utterance: OwnedVocoderUtterance,
    conditioning: PitchConditioningV2,
    target: dict[str, object],
    *,
    start: int,
    frames: int,
) -> tuple[torch.Tensor, ...]:
    base = _segment(utterance, target, start=start, frames=frames)
    end = start + frames
    if conditioning.f0_track_hz.shape != (int(utterance.mel_frames),):
        raise RuntimeError("conditioning-v2 F0 geometry mismatch")
    return (
        base[0],
        conditioning.f0_track_hz[start:end].unsqueeze(0).cpu(),
        conditioning.energy_confidence[start:end].unsqueeze(0).cpu(),
        conditioning.periodic_strength[start:end].unsqueeze(0).cpu(),
        base[4],
        base[5],
        base[6],
        base[7],
    )


def _load_base_v2_weights(root: Path, model: LykenoxContinuousResidualSourceV2) -> dict[str, object]:
    path = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / BASE_V2_RUN_DIR_NAME
        / "best.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"Historical Continuous Source V2 best checkpoint is required: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("checkpoint_schema_version") != BASE_V2_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("historical V2 checkpoint schema mismatch")
    if payload.get("architecture") != CONTINUOUS_SOURCE_ARCHITECTURE_V2:
        raise RuntimeError("historical V2 architecture mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    return {
        "path": str(path),
        "update": int(payload.get("update", -1)),
        "best_val_total": float(payload.get("best_val_total", math.inf)),
    }


def _save_checkpoint(
    path: Path,
    model: LykenoxContinuousResidualSourceV2,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    best_val: float,
    config: dict[str, object],
) -> None:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": CONTINUOUS_SOURCE_ARCHITECTURE_V2,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "renderer_version": RENDERER_VERSION,
        "update": int(update),
        "best_val_total": float(best_val),
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def _evaluate_complete(
    model: LykenoxContinuousResidualSourceV2,
    root: Path,
    utterances: list[OwnedVocoderUtterance],
    conditioning_by_id: dict[str, PitchConditioningV2],
) -> dict[str, float]:
    model.eval()
    totals: list[dict[str, float]] = []
    for index, utterance in enumerate(utterances):
        target = _load_or_build_target(root, utterance)
        conditioning = conditioning_by_id[utterance.utterance_id]
        tensors = _segment_with_conditioning_v2(
            utterance,
            conditioning,
            target,
            start=0,
            frames=int(utterance.mel_frames),
        )
        _, terms = _loss_terms(
            model,
            tensors,
            teacher_ratio=0.0,
            teacher_seed=200000 + index,
        )
        totals.append(terms)
    if not totals:
        raise RuntimeError("no held-out utterances available for conditioning-v2 validation")
    keys = tuple(totals[0])
    return {key: sum(item[key] for item in totals) / float(len(totals)) for key in keys}


def train_continuous_residual_source_v2_pitch_conditioning_v2(
    root: Path,
    *,
    train_items: int = DEFAULT_TRAIN_ITEMS,
    val_items: int = DEFAULT_VAL_ITEMS,
    segment_frames: int = DEFAULT_SEGMENT_FRAMES,
    max_updates: int = DEFAULT_MAX_UPDATES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    resume: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if segment_frames < 32 or max_updates < 1 or train_items < 1 or val_items < 1:
        raise ValueError("invalid controlled V2 conditioning retrain limits")
    root = Path(root).resolve()
    torch.manual_seed(int(seed))
    run_dir = root / "models" / "lykenox_identity" / "training" / RUN_DIR_NAME
    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"

    train_set = collect_owned_vocoder_utterances(root, "train", max_items=train_items)
    val_set = collect_owned_vocoder_utterances(root, "val", max_items=val_items)
    train_conditioning = _conditioning_map(train_set)
    val_conditioning = _conditioning_map(val_set)

    model = LykenoxContinuousResidualSourceV2().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    start_update = 0
    best_val = math.inf
    base_warm_start: dict[str, object] | None = None
    if resume and latest.exists():
        try:
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(latest, map_location="cpu")
        if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("conditioning-v2 controlled checkpoint schema mismatch")
        if checkpoint.get("architecture") != CONTINUOUS_SOURCE_ARCHITECTURE_V2:
            raise RuntimeError("conditioning-v2 controlled architecture mismatch")
        if checkpoint.get("conditioning_contract") != PITCH_CONDITIONING_V2:
            raise RuntimeError("conditioning-v2 checkpoint contract mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_update = int(checkpoint["update"])
        best_val = float(checkpoint.get("best_val_total", math.inf))
    else:
        base_warm_start = _load_base_v2_weights(root, model)

    config: dict[str, object] = {
        "train_items": train_items,
        "val_items": val_items,
        "segment_frames": segment_frames,
        "max_updates": max_updates,
        "learning_rate": learning_rate,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "grad_clip": DEFAULT_GRAD_CLIP,
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "architecture_unchanged_from_continuous_source_v2": True,
        "loss_function_unchanged_from_continuous_source_v2": True,
        "teacher_forcing_schedule_unchanged_from_continuous_source_v2": True,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "conditioning_slots": ["f0_track_hz", "energy_confidence", "periodic_strength"],
        "legacy_binary_voiced_used_as_model_conditioning": False,
        "base_v2_complete_weight_warm_start": base_warm_start,
        "complete_heldout_validation": True,
    }

    history_path = run_dir / "history.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    for update in range(start_update + 1, max_updates + 1):
        model.train()
        utterance_index = (int(seed) + update * 7919) % len(train_set)
        utterance = train_set[utterance_index]
        target = _load_or_build_target(root, utterance)
        conditioning = train_conditioning[utterance.utterance_id]
        frames = min(segment_frames, int(utterance.mel_frames))
        start = _deterministic_crop(
            int(utterance.mel_frames), frames, update=update, seed=seed + utterance_index
        )
        tensors = _segment_with_conditioning_v2(
            utterance,
            conditioning,
            target,
            start=start,
            frames=frames,
        )
        ratio = _teacher_ratio(update, max_updates)

        optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss_terms(
            model,
            tensors,
            teacher_ratio=ratio,
            teacher_seed=seed + update,
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise RuntimeError("controlled conditioning-v2 retrain gradient norm became non-finite")
        optimizer.step()

        record: dict[str, object] = {
            "update": update,
            "utterance_id": utterance.utterance_id,
            "start_frame": start,
            "teacher_forcing_ratio": ratio,
            "grad_norm": grad_norm,
            **terms,
        }
        if update % DEFAULT_EVAL_EVERY == 0 or update == max_updates:
            validation = _evaluate_complete(model, root, val_set, val_conditioning)
            record["validation_complete_utterances"] = validation
            if validation["total"] < best_val:
                best_val = validation["total"]
                _save_checkpoint(best, model, optimizer, update=update, best_val=best_val, config=config)

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if update % DEFAULT_CHECKPOINT_EVERY == 0 or update == max_updates:
            _save_checkpoint(latest, model, optimizer, update=update, best_val=best_val, config=config)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report: dict[str, object] = {
        "status": "continuous_residual_source_v2_pitch_conditioning_v2_training_complete",
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": CONTINUOUS_SOURCE_ARCHITECTURE_V2,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "updates": max_updates,
        "best_val_total": best_val,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "architecture_changed": False,
        "loss_function_changed": False,
        "step3f_target_changed": False,
        "renderer_changed": False,
        "conditioning_contract_changed": True,
        "base_v2_complete_weight_warm_start": base_warm_start,
        "codebook_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "next_action": "render_against_historical_v2_baseline_and_identity_ceiling_then_listen",
    }
    _atomic_json(run_dir / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--val-items", type=int, default=DEFAULT_VAL_ITEMS)
    parser.add_argument("--segment-frames", type=int, default=DEFAULT_SEGMENT_FRAMES)
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_continuous_residual_source_v2_pitch_conditioning_v2(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        segment_frames=args.segment_frames,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
