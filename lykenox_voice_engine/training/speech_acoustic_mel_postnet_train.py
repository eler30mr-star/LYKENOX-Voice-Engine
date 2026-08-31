"""Bounded exactly-resumable first-epoch trainer for the mel residual postnet.

The accepted acoustic-v2 base is immutable and never stored as trainable candidate state.
Only ``postnet.*`` parameters are optimized. Training is hard-blocked after one epoch and
full-utterance v4.2 listening is required before any further work can be considered.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from lykenox_voice_engine.models.speech.mel_postnet import MEL_POSTNET_ARCHITECTURE_V1
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    FRAME_CONTEXT_VERSION,
    TRAINER_CONTRACT_VERSION as BASE_TRAINER_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_loss import (
    ACOUSTIC_MEL_FIDELITY_LOSS_VERSION,
    acoustic_mel_fidelity_loss,
)
from lykenox_voice_engine.training.speech_acoustic_mel_postnet_artifact import (
    MEL_POSTNET_CHECKPOINT_VERSION,
    MEL_POSTNET_HIDDEN_CHANNELS,
    base_checkpoint_path,
    build_candidate_from_base,
    file_sha256,
    load_mel_postnet_checkpoint,
    postnet_output_dir,
    save_mel_postnet_checkpoint,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    build_acoustic_prosody_provenance,
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)


TRAINER_CONTRACT_VERSION = "acoustic-mel-postnet-first-epoch-resumable-v1"
TRAIN_ORDER_VERSION = "epoch-shuffle-v1"
HARD_EPOCH_LIMIT = 1
DEFAULT_TIME_BUDGET_SECONDS = 300.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 20.0


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _run_config(
    *,
    base_sha256: str,
    train_count: int,
    val_count: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    grad_clip: float,
    checkpoint_every_updates: int,
    dataset_item_limit: int | None,
) -> dict[str, object]:
    return {
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "checkpoint_version": MEL_POSTNET_CHECKPOINT_VERSION,
        "architecture": MEL_POSTNET_ARCHITECTURE_V1,
        "loss_version": ACOUSTIC_MEL_FIDELITY_LOSS_VERSION,
        "train_order_version": TRAIN_ORDER_VERSION,
        "base_trainer_contract_version": BASE_TRAINER_CONTRACT_VERSION,
        "frame_context_version": FRAME_CONTEXT_VERSION,
        "base_checkpoint_sha256": base_sha256,
        "hidden_channels": MEL_POSTNET_HIDDEN_CHANNELS,
        "trainable_surface": "postnet_only",
        "hard_epoch_limit": HARD_EPOCH_LIMIT,
        "train_count": int(train_count),
        "val_count": int(val_count),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "weight_decay": 0.0,
        "grad_clip": float(grad_clip),
        "checkpoint_every_updates": int(checkpoint_every_updates),
        "dataset_item_limit": None if dataset_item_limit is None else int(dataset_item_limit),
        "teacher_duration_grid": True,
        "duration_training": False,
        "prosody_training": False,
        "vocoder_training": False,
    }


def _epoch_order(item_count: int, *, seed: int, epoch: int) -> list[int]:
    order = list(range(item_count))
    random.Random(seed + epoch).shuffle(order)
    return order


def _batch(dataset: LykenoxAlignedSpeechDataset, indices: list[int]):
    return collate_aligned_speech([dataset[index] for index in indices]).to("cpu")


def _snapshot(result) -> dict[str, float]:
    return {
        "total": float(result.total.detach()),
        "mel_l1": float(result.mel_l1.detach()),
        "centered_shape": float(result.centered_shape.detach()),
        "spectral_delta": float(result.spectral_delta.detach()),
        "temporal_delta": float(result.temporal_delta.detach()),
        "clarity_underpresence": float(result.clarity_underpresence.detach()),
    }


def _loss(candidate, batch):
    output = candidate(batch.token_ids, batch.token_mask, batch.durations)
    if not torch.equal(output["regulated_durations"], batch.durations):
        raise RuntimeError("mel postnet trainer changed teacher durations")
    return acoustic_mel_fidelity_loss(
        output["mel"],
        batch.mel,
        batch.mel_mask,
        sample_rate=candidate.config.sample_rate,
        n_fft=candidate.config.n_fft,
    )


def _validation(candidate, dataset: LykenoxAlignedSpeechDataset, *, batch_size: int) -> dict[str, float]:
    sums = {
        "total": 0.0,
        "mel_l1": 0.0,
        "centered_shape": 0.0,
        "spectral_delta": 0.0,
        "temporal_delta": 0.0,
        "clarity_underpresence": 0.0,
    }
    count = 0
    candidate.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            current = _snapshot(_loss(candidate, _batch(dataset, indices)))
            if not all(math.isfinite(value) for value in current.values()):
                raise RuntimeError("non-finite mel postnet validation metric")
            for name, value in current.items():
                sums[name] += value
            count += 1
    if count < 1:
        raise RuntimeError("mel postnet validation dataset is empty")
    return {name: value / count for name, value in sums.items()}


def _empty_accumulator() -> dict[str, object]:
    return {
        "update_count": 0,
        "sums": {
            "total": 0.0,
            "mel_l1": 0.0,
            "centered_shape": 0.0,
            "spectral_delta": 0.0,
            "temporal_delta": 0.0,
            "clarity_underpresence": 0.0,
        },
    }


def _accumulate(accumulator: dict[str, object], snapshot: dict[str, float]) -> None:
    sums = accumulator.get("sums")
    if not isinstance(sums, dict):
        raise RuntimeError("mel postnet accumulator is invalid")
    for name, value in snapshot.items():
        sums[name] = float(sums.get(name, 0.0)) + float(value)
    accumulator["update_count"] = int(accumulator.get("update_count", 0)) + 1


def _accumulator_means(accumulator: dict[str, object]) -> dict[str, float]:
    count = int(accumulator.get("update_count", 0))
    sums = accumulator.get("sums")
    if count < 1 or not isinstance(sums, dict):
        raise RuntimeError("cannot summarize empty mel postnet epoch")
    return {name: float(value) / count for name, value in sums.items()}


def run_mel_postnet_training(
    root: Path,
    *,
    output_dir: Path | None = None,
    batch_size: int = 2,
    seed: int = 1701,
    learning_rate: float = 1e-4,
    grad_clip: float = 5.0,
    checkpoint_every_updates: int = 8,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    checkpoint_reserve_seconds: float = DEFAULT_CHECKPOINT_RESERVE_SECONDS,
    max_updates_this_run: int | None = None,
    dataset_item_limit: int | None = None,
) -> dict[str, object]:
    if batch_size < 1 or learning_rate <= 0.0 or grad_clip <= 0.0:
        raise ValueError("invalid mel postnet optimizer configuration")
    if checkpoint_every_updates < 1:
        raise ValueError("checkpoint_every_updates must be positive")
    if time_budget_seconds <= checkpoint_reserve_seconds + 2.0:
        raise ValueError("time budget too small for safe checkpointing")
    if max_updates_this_run is not None and max_updates_this_run < 1:
        raise ValueError("max_updates_this_run must be positive")
    if dataset_item_limit is not None and dataset_item_limit < 1:
        raise ValueError("dataset_item_limit must be positive")

    root = Path(root).resolve()
    base_path = base_checkpoint_path(root)
    if not base_path.exists():
        raise FileNotFoundError(f"accepted acoustic-v2 checkpoint not found: {base_path}")
    base_sha = file_sha256(base_path)
    base, base_payload = load_acoustic_prosody_checkpoint(base_path)
    base_run = base_payload.get("run_config")
    if not isinstance(base_run, dict):
        raise RuntimeError("accepted acoustic-v2 checkpoint is missing run identity")
    if base_run.get("trainer_contract_version") != BASE_TRAINER_CONTRACT_VERSION:
        raise RuntimeError("mel postnet requires accepted acoustic-v2 trainer identity")
    if base.config.frame_context_version != FRAME_CONTEXT_VERSION:
        raise RuntimeError("mel postnet requires accepted frame-context architecture")

    duration_root = find_clean_duration_root(root)
    provenance = build_acoustic_prosody_provenance(
        root,
        duration_root=duration_root,
        config=base.config,
    )
    if base_payload.get("training_provenance") != provenance:
        raise RuntimeError("accepted acoustic-v2 provenance changed")

    train_dataset = LykenoxAlignedSpeechDataset(
        root,
        "train",
        base.config,
        duration_root=duration_root,
        include_pitch_targets=False,
    )
    val_dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        base.config,
        duration_root=duration_root,
        include_pitch_targets=False,
    )
    effective_train_count = min(
        len(train_dataset),
        len(train_dataset) if dataset_item_limit is None else dataset_item_limit,
    )
    run_config = _run_config(
        base_sha256=base_sha,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        batch_size=batch_size,
        seed=seed,
        learning_rate=learning_rate,
        grad_clip=grad_clip,
        checkpoint_every_updates=checkpoint_every_updates,
        dataset_item_limit=dataset_item_limit,
    )

    output_dir = Path(output_dir).resolve() if output_dir is not None else postnet_output_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    progress_path = output_dir / "training_progress.json"
    report_path = output_dir / "training_report.json"
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    if last_path.exists():
        candidate, payload = load_mel_postnet_checkpoint(
            last_path,
            base_checkpoint=base_path,
            expected_provenance=provenance,
            expected_run_config=run_config,
        )
        optimizer = torch.optim.AdamW(
            candidate.postnet.parameters(), lr=learning_rate, weight_decay=0.0
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        epoch = int(payload["epoch"])
        next_item_offset = int(payload["next_item_offset"])
        global_step = int(payload["global_step"])
        metadata = payload.get("training_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("mel postnet checkpoint is missing training metadata")
        initial_validation = dict(metadata["initial_validation"])
        best_validation = dict(metadata["best_validation"])
        best_epoch = int(metadata["best_epoch"])
        history = list(metadata.get("history", []))
        accumulator = dict(metadata.get("epoch_accumulator", _empty_accumulator()))
    else:
        torch.manual_seed(seed)
        candidate = build_candidate_from_base(base_path)
        optimizer = torch.optim.AdamW(
            candidate.postnet.parameters(), lr=learning_rate, weight_decay=0.0
        )
        initial_validation = _validation(candidate, val_dataset, batch_size=batch_size)
        best_validation = dict(initial_validation)
        best_epoch = 0
        history: list[dict[str, object]] = []
        accumulator = _empty_accumulator()
        epoch = 1
        next_item_offset = 0
        global_step = 0

    names = candidate.trainable_parameter_names()
    if not names or any(not name.startswith("postnet.") for name in names):
        raise RuntimeError("mel postnet trainer may optimize only postnet parameters")
    if any(parameter.requires_grad for parameter in candidate.base_model.parameters()):
        raise RuntimeError("base acoustic model became trainable")
    candidate.eval()

    def metadata_payload() -> dict[str, object]:
        return {
            "initial_validation": dict(initial_validation),
            "best_validation": dict(best_validation),
            "best_epoch": int(best_epoch),
            "history": list(history),
            "epoch_accumulator": accumulator,
            "only_postnet_trainable": True,
            "epoch2_training_blocked": True,
        }

    def save_last() -> None:
        save_mel_postnet_checkpoint(
            last_path,
            candidate,
            optimizer,
            base_sha256=base_sha,
            epoch=epoch,
            next_item_offset=next_item_offset,
            global_step=global_step,
            training_provenance=provenance,
            run_config=run_config,
            training_metadata=metadata_payload(),
        )

    if epoch > HARD_EPOCH_LIMIT or history:
        return {
            "status": "gate_reached",
            "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "architecture": MEL_POSTNET_ARCHITECTURE_V1,
            "epochs_completed": len(history),
            "current_epoch": epoch,
            "next_item_offset": next_item_offset,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "candidate_numerically_improved": best_epoch == 1,
            "only_postnet_trainable": True,
            "base_acoustic_immutable": True,
            "epoch2_training_blocked": True,
            "training_complete": False,
            "next_gate": (
                "run_mel_postnet_full_utterance_v4_2_ab"
                if best_epoch == 1
                else "reject_mel_postnet_without_perceptual_run"
            ),
        }

    started = time.perf_counter()
    updates_this_run = 0
    stop_reason: str | None = None
    order = _epoch_order(effective_train_count, seed=seed, epoch=epoch)
    if next_item_offset < 0 or next_item_offset > len(order):
        raise RuntimeError("mel postnet next_item_offset is invalid")

    while next_item_offset < len(order):
        elapsed = time.perf_counter() - started
        if elapsed >= time_budget_seconds - checkpoint_reserve_seconds:
            stop_reason = "time_budget_reached"
            save_last()
            break
        if max_updates_this_run is not None and updates_this_run >= max_updates_this_run:
            stop_reason = "max_updates_this_run_reached"
            save_last()
            break

        indices = order[next_item_offset : min(next_item_offset + batch_size, len(order))]
        batch = _batch(train_dataset, indices)
        optimizer.zero_grad(set_to_none=True)
        result = _loss(candidate, batch)
        if not torch.isfinite(result.total):
            raise RuntimeError(f"non-finite mel postnet loss at global step {global_step}")
        result.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(candidate.postnet.parameters(), grad_clip)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"non-finite mel postnet gradient at global step {global_step}")
        optimizer.step()

        _accumulate(accumulator, _snapshot(result))
        next_item_offset += len(indices)
        global_step += 1
        updates_this_run += 1
        if global_step % checkpoint_every_updates == 0:
            save_last()

    if stop_reason is not None:
        report = {
            "status": "incomplete",
            "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "architecture": MEL_POSTNET_ARCHITECTURE_V1,
            "stop_reason": stop_reason,
            "epoch": epoch,
            "next_item_offset": next_item_offset,
            "effective_train_count": effective_train_count,
            "global_step": global_step,
            "updates_this_run": updates_this_run,
            "only_postnet_trainable": True,
            "base_acoustic_immutable": True,
            "epoch2_training_blocked": True,
            "next_gate": "rerun_same_command_to_resume_first_epoch",
        }
        _atomic_json(progress_path, report)
        return report

    validation = _validation(candidate, val_dataset, batch_size=batch_size)
    train_means = _accumulator_means(accumulator)
    improved = validation["total"] < initial_validation["total"]
    history.append({"epoch": 1, "train": train_means, "validation": validation})
    if improved:
        best_validation = dict(validation)
        best_epoch = 1
    epoch = 2
    next_item_offset = 0
    accumulator = _empty_accumulator()
    save_last()
    if improved:
        save_mel_postnet_checkpoint(
            best_path,
            candidate,
            optimizer,
            base_sha256=base_sha,
            epoch=epoch,
            next_item_offset=next_item_offset,
            global_step=global_step,
            training_provenance=provenance,
            run_config=run_config,
            training_metadata=metadata_payload(),
        )

    report = {
        "status": "gate_reached",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "architecture": MEL_POSTNET_ARCHITECTURE_V1,
        "loss_version": ACOUSTIC_MEL_FIDELITY_LOSS_VERSION,
        "base_checkpoint_sha256": base_sha,
        "epochs_completed": 1,
        "current_epoch": 2,
        "next_item_offset": 0,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "candidate_numerically_improved": improved,
        "initial_validation": initial_validation,
        "epoch1_validation": validation,
        "train_epoch1": train_means,
        "only_postnet_trainable": True,
        "base_acoustic_immutable": True,
        "teacher_duration_grid_used": True,
        "duration_training": False,
        "prosody_training": False,
        "vocoder_training": False,
        "epoch2_training_blocked": True,
        "training_complete": False,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if improved else None,
        "next_gate": (
            "run_mel_postnet_full_utterance_v4_2_ab"
            if improved
            else "reject_mel_postnet_without_perceptual_run"
        ),
    }
    _atomic_json(progress_path, report)
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_mel_postnet_training(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
