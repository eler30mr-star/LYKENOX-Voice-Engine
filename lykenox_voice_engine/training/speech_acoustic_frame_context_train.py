"""Bounded, exactly resumable persistent v2 acoustic training with frame context.

This trainer is a new experiment identity. It never resumes or fine-tunes the rejected
piecewise-constant v1 checkpoint. It reuses the validated LYKENOX supervision contracts
(alignment-v3, mel-v1, speech-pitch-cache-v1) while selecting the accepted bounded-smoke
frame architecture ``token-progress-conv-v1``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.models.speech.config import FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    build_acoustic_prosody_provenance,
    load_acoustic_prosody_checkpoint,
    save_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_train import (
    DEFAULT_CHECKPOINT_RESERVE_SECONDS,
    DEFAULT_TIME_BUDGET_SECONDS,
    EpochAccumulator,
    _atomic_json,
    _batch_from_indices,
    _checkpoint_metadata,
    _compute_losses,
    _epoch_order,
    _snapshot,
    _validation_metrics,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CACHE_VERSION


TRAINER_CONTRACT_VERSION = "acoustic-frame-context-bounded-resumable-v2"
TRAIN_ORDER_VERSION = "epoch-shuffle-v1"
FRAME_CONTEXT_VERSION = FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1


def _run_config(
    *,
    config: LykenoxSpeechConfig,
    train_count: int,
    val_count: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    duration_weight: float,
    f0_weight: float,
    voicing_weight: float,
    min_delta: float,
    checkpoint_every_updates: int,
) -> dict[str, object]:
    if config.frame_context_version != FRAME_CONTEXT_VERSION:
        raise RuntimeError("v2 persistent trainer requires token-progress-conv-v1")
    return {
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "train_order_version": TRAIN_ORDER_VERSION,
        "frame_context_version": config.frame_context_version,
        "frame_context_layers": int(config.frame_context_layers),
        "frame_context_kernel_size": int(config.frame_context_kernel_size),
        "train_count": int(train_count),
        "val_count": int(val_count),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "duration_weight": float(duration_weight),
        "f0_weight": float(f0_weight),
        "voicing_weight": float(voicing_weight),
        "min_delta": float(min_delta),
        "checkpoint_every_updates": int(checkpoint_every_updates),
        "pitch_cache_version": PITCH_CACHE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
    }


def run_acoustic_frame_context_training(
    root: Path,
    *,
    output_dir: Path | None = None,
    batch_size: int = 2,
    max_epochs: int = 36,
    patience: int = 6,
    seed: int = 1337,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    duration_weight: float = 0.10,
    f0_weight: float = 0.25,
    voicing_weight: float = 0.25,
    min_delta: float = 1e-4,
    checkpoint_every_updates: int = 16,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    checkpoint_reserve_seconds: float = DEFAULT_CHECKPOINT_RESERVE_SECONDS,
    max_updates_this_run: int | None = None,
) -> dict[str, object]:
    if batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("batch_size, max_epochs and patience must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0 or grad_clip <= 0.0:
        raise ValueError("invalid optimizer configuration")
    if min(duration_weight, f0_weight, voicing_weight) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if f0_weight == 0.0 or voicing_weight == 0.0:
        raise ValueError("v2 persistent training requires non-zero F0/voicing weights")
    if checkpoint_every_updates < 1:
        raise ValueError("checkpoint_every_updates must be positive")
    if time_budget_seconds <= checkpoint_reserve_seconds + 5.0:
        raise ValueError("time budget is too small for safe acoustic checkpointing")
    if max_updates_this_run is not None and max_updates_this_run < 1:
        raise ValueError("max_updates_this_run must be positive when supplied")

    root = Path(root).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "training" / "acoustic_frame_context_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    progress_path = output_dir / "training_progress.json"
    report_path = output_dir / "training_report.json"

    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    frontend = SpanishTextFrontend()
    config = LykenoxSpeechConfig(
        vocab_size=frontend.vocab_size,
        frame_context_version=FRAME_CONTEXT_VERSION,
    )
    duration_root = find_clean_duration_root(root)
    train_dataset = LykenoxAlignedSpeechDataset(
        root,
        "train",
        config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    val_dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    provenance = build_acoustic_prosody_provenance(
        root,
        duration_root=duration_root,
        config=config,
    )
    run_config = _run_config(
        config=config,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        duration_weight=duration_weight,
        f0_weight=f0_weight,
        voicing_weight=voicing_weight,
        min_delta=min_delta,
        checkpoint_every_updates=checkpoint_every_updates,
    )

    if last_path.exists():
        model, payload = load_acoustic_prosody_checkpoint(
            last_path,
            expected_provenance=provenance,
            expected_run_config=run_config,
        )
        if model.config.frame_context_version != FRAME_CONTEXT_VERSION:
            raise RuntimeError("Refusing to resume a non-v2 acoustic checkpoint")
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        epoch = int(payload["epoch"])
        next_item_offset = int(payload["next_item_offset"])
        global_step = int(payload["global_step"])
        metadata = payload.get("training_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("v2 resume checkpoint is missing training metadata")
        initial_validation = dict(metadata["initial_validation"])
        best_validation = dict(metadata["best_validation"])
        best_epoch = int(metadata["best_epoch"])
        epochs_completed = int(metadata["epochs_completed"])
        no_improvement_epochs = int(metadata["no_improvement_epochs"])
        history = list(metadata.get("history", []))
        accumulator = EpochAccumulator.from_payload(
            metadata.get("epoch_accumulator"), epoch=epoch
        )
    else:
        torch.manual_seed(seed)
        model = LykenoxSpeechAcousticModel(config).cpu().train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        initial_validation = _validation_metrics(
            model,
            val_dataset,
            batch_size=batch_size,
            duration_weight=duration_weight,
            f0_weight=f0_weight,
            voicing_weight=voicing_weight,
        )
        best_validation = dict(initial_validation)
        best_epoch = 0
        epoch = 1
        next_item_offset = 0
        global_step = 0
        epochs_completed = 0
        no_improvement_epochs = 0
        history: list[dict[str, object]] = []
        accumulator = EpochAccumulator()

    if report_path.exists() and (epoch > max_epochs or no_improvement_epochs >= patience):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(report, dict) and report.get("status") in {"pass", "needs_review"}:
            return report

    updates_this_run = 0
    stop_reason: str | None = None

    def save_last() -> None:
        metadata = _checkpoint_metadata(
            initial_validation=initial_validation,
            best_validation=best_validation,
            best_epoch=best_epoch,
            epochs_completed=epochs_completed,
            no_improvement_epochs=no_improvement_epochs,
            history=history,
            accumulator=accumulator,
            epoch=epoch,
        )
        save_acoustic_prosody_checkpoint(
            last_path,
            model,
            optimizer,
            frontend=frontend,
            epoch=epoch,
            next_item_offset=next_item_offset,
            global_step=global_step,
            training_provenance=provenance,
            run_config=run_config,
            training_metadata=metadata,
        )

    while epoch <= max_epochs and no_improvement_epochs < patience:
        order = _epoch_order(len(train_dataset), seed=seed, epoch=epoch)
        if next_item_offset < 0 or next_item_offset > len(order):
            raise RuntimeError("v2 checkpoint next_item_offset is invalid")

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

            batch_indices = order[
                next_item_offset : min(next_item_offset + batch_size, len(order))
            ]
            batch = _batch_from_indices(train_dataset, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            losses = _compute_losses(
                model,
                batch,
                duration_weight=duration_weight,
                f0_weight=f0_weight,
                voicing_weight=voicing_weight,
            )
            if not torch.isfinite(losses.total):
                raise RuntimeError(f"Non-finite v2 loss at global step {global_step}")
            losses.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"Non-finite v2 gradient at global step {global_step}")
            optimizer.step()

            accumulator.add(_snapshot(losses))
            next_item_offset += len(batch_indices)
            global_step += 1
            updates_this_run += 1
            if global_step % checkpoint_every_updates == 0:
                save_last()

        if stop_reason is not None:
            break

        validation = _validation_metrics(
            model,
            val_dataset,
            batch_size=batch_size,
            duration_weight=duration_weight,
            f0_weight=f0_weight,
            voicing_weight=voicing_weight,
        )
        train_means = accumulator.means()
        epochs_completed = epoch
        improved = validation["total"] < best_validation["total"] - min_delta
        if improved:
            best_validation = dict(validation)
            best_epoch = epoch
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train": train_means,
                "validation": dict(validation),
                "improved": bool(improved),
            }
        )
        completed_epoch = epoch
        epoch += 1
        next_item_offset = 0
        accumulator = EpochAccumulator()

        metadata = _checkpoint_metadata(
            initial_validation=initial_validation,
            best_validation=best_validation,
            best_epoch=best_epoch,
            epochs_completed=epochs_completed,
            no_improvement_epochs=no_improvement_epochs,
            history=history,
            accumulator=accumulator,
            epoch=epoch,
        )
        if improved:
            save_acoustic_prosody_checkpoint(
                best_path,
                model,
                optimizer,
                frontend=frontend,
                epoch=epoch,
                next_item_offset=0,
                global_step=global_step,
                training_provenance=provenance,
                run_config=run_config,
                training_metadata={**metadata, "selected_after_epoch": completed_epoch},
            )
        save_last()

    if stop_reason is not None:
        progress: dict[str, object] = {
            "status": "incomplete",
            "stop_reason": stop_reason,
            "device": "cpu",
            "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "frame_context_version": FRAME_CONTEXT_VERSION,
            "epochs_completed": epochs_completed,
            "current_epoch": epoch,
            "next_item_offset": next_item_offset,
            "global_step": global_step,
            "updates_this_run": updates_this_run,
            "best_epoch": best_epoch,
            "initial_validation": {k: round(v, 6) for k, v in initial_validation.items()},
            "best_validation": {k: round(v, 6) for k, v in best_validation.items()},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
            "next_gate": "rerun_same_v2_command_to_resume",
        }
        _atomic_json(progress_path, progress)
        return progress

    stop_reason = "early_stopping" if no_improvement_epochs >= patience else "max_epochs_reached"
    save_last()
    trained_best_exists = best_epoch > 0 and best_path.exists()
    selection_improved = best_validation["total"] < initial_validation["total"]
    status = "pass" if trained_best_exists and selection_improved else "needs_review"
    report: dict[str, object] = {
        "status": status,
        "stop_reason": stop_reason,
        "device": "cpu",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "frame_context_version": FRAME_CONTEXT_VERSION,
        "frame_context_layers": config.frame_context_layers,
        "frame_context_kernel_size": config.frame_context_kernel_size,
        "pitch_cache_version": PITCH_CACHE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "parameters": model.parameter_count(),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "initial_validation": {k: round(v, 6) for k, v in initial_validation.items()},
        "best_validation": {k: round(v, 6) for k, v in best_validation.items()},
        "validation_total_improved": selection_improved,
        "history": history,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "next_gate": (
            "audit_persistent_acoustic_frame_context_v2_before_duration_fix"
            if status == "pass"
            else "review_v2_persistent_training_before_more_work"
        ),
        "warning": (
            "A pass closes persistent v2 supervised training only. The held-out v2 audit "
            "must still confirm frame expressivity and prosody quality before predicted-"
            "duration inference or unseen-text synthesis."
        ),
    }
    _atomic_json(report_path, report)
    _atomic_json(progress_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--duration-weight", type=float, default=0.10)
    parser.add_argument("--f0-weight", type=float, default=0.25)
    parser.add_argument("--voicing-weight", type=float, default=0.25)
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_acoustic_frame_context_training(
                args.root,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                seed=args.seed,
                learning_rate=args.learning_rate,
                duration_weight=args.duration_weight,
                f0_weight=args.f0_weight,
                voicing_weight=args.voicing_weight,
                time_budget_seconds=args.time_budget_seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
