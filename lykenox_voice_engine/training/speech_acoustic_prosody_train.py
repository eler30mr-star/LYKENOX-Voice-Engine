"""Bounded, exactly resumable persistent training for LYKENOX Speech acoustics.

The trainer consumes only validated persistent LYKENOX supervision:
- alignment-v3 teacher durations;
- mel-v1 acoustic targets;
- speech-pitch-cache-v1 F0/voicing targets.

Rerun the same command to continue.  The exact data provenance, model configuration,
training identity, optimizer state, torch RNG state, epoch position, partial-epoch metrics,
and history are checkpointed so a short Windows command budget does not change the
experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    build_acoustic_prosody_provenance,
    load_acoustic_prosody_checkpoint,
    save_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    AlignedSpeechBatch,
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_losses import SpeechLosses, speech_training_losses
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CACHE_VERSION


TRAINER_CONTRACT_VERSION = "acoustic-prosody-bounded-resumable-v1"
TRAIN_ORDER_VERSION = "epoch-shuffle-v1"
DEFAULT_TIME_BUDGET_SECONDS = 70.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 8.0


@dataclass
class EpochAccumulator:
    total_sum: float = 0.0
    acoustic_sum: float = 0.0
    duration_sum: float = 0.0
    f0_sum: float = 0.0
    voicing_sum: float = 0.0
    update_count: int = 0

    @classmethod
    def from_payload(cls, payload: object, *, epoch: int) -> "EpochAccumulator":
        if not isinstance(payload, dict) or int(payload.get("epoch", -1)) != int(epoch):
            return cls()
        return cls(
            total_sum=float(payload.get("total_sum", 0.0)),
            acoustic_sum=float(payload.get("acoustic_sum", 0.0)),
            duration_sum=float(payload.get("duration_sum", 0.0)),
            f0_sum=float(payload.get("f0_sum", 0.0)),
            voicing_sum=float(payload.get("voicing_sum", 0.0)),
            update_count=int(payload.get("update_count", 0)),
        )

    def add(self, snapshot: dict[str, float]) -> None:
        self.total_sum += snapshot["total"]
        self.acoustic_sum += snapshot["acoustic"]
        self.duration_sum += snapshot["duration"]
        self.f0_sum += snapshot["f0"]
        self.voicing_sum += snapshot["voicing"]
        self.update_count += 1

    def means(self) -> dict[str, float]:
        if self.update_count < 1:
            raise RuntimeError("Cannot summarize an empty acoustic epoch accumulator")
        count = float(self.update_count)
        return {
            "total": self.total_sum / count,
            "acoustic": self.acoustic_sum / count,
            "duration": self.duration_sum / count,
            "f0": self.f0_sum / count,
            "voicing": self.voicing_sum / count,
        }

    def to_payload(self, *, epoch: int) -> dict[str, object]:
        return {
            "epoch": int(epoch),
            "total_sum": self.total_sum,
            "acoustic_sum": self.acoustic_sum,
            "duration_sum": self.duration_sum,
            "f0_sum": self.f0_sum,
            "voicing_sum": self.voicing_sum,
            "update_count": self.update_count,
        }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _snapshot(losses: SpeechLosses) -> dict[str, float]:
    if losses.f0 is None or losses.voicing is None:
        raise RuntimeError("Persistent acoustic training requires F0 and voicing losses")
    return {
        "total": float(losses.total.detach().cpu()),
        "acoustic": float(losses.acoustic.detach().cpu()),
        "duration": float(losses.duration.detach().cpu()),
        "f0": float(losses.f0.detach().cpu()),
        "voicing": float(losses.voicing.detach().cpu()),
    }


def _compute_losses(
    model: LykenoxSpeechAcousticModel,
    batch: AlignedSpeechBatch,
    *,
    duration_weight: float,
    f0_weight: float,
    voicing_weight: float,
) -> SpeechLosses:
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("Persistent acoustic training requires cached pitch targets")
    output = model(batch.token_ids, batch.token_mask, batch.durations)
    if output["mel"].shape != batch.mel.shape:
        raise RuntimeError("Persistent acoustic model did not preserve target mel shape")
    if output["f0_prediction_hz"].shape != batch.f0_hz.shape:
        raise RuntimeError("Persistent acoustic F0 output is off the target frame grid")
    if output["voicing_logits"].shape != batch.voiced.shape:
        raise RuntimeError("Persistent acoustic voicing output is off the target frame grid")
    if not torch.equal(output["mel_mask"], batch.mel_mask):
        raise RuntimeError("Persistent acoustic mel mask differs from target mask")
    if not torch.equal(output["mel_lengths"], batch.mel_lengths):
        raise RuntimeError("Persistent acoustic regulated lengths differ from teacher lengths")
    return speech_training_losses(
        mel_prediction=output["mel"],
        mel_target=batch.mel,
        mel_mask=batch.mel_mask,
        duration_prediction=output["duration_prediction"],
        duration_target=batch.durations,
        token_mask=batch.token_mask,
        duration_weight=duration_weight,
        f0_prediction_hz=output["f0_prediction_hz"],
        f0_target_hz=batch.f0_hz,
        voicing_logits=output["voicing_logits"],
        voicing_target=batch.voiced,
        f0_weight=f0_weight,
        voicing_weight=voicing_weight,
    )


def _epoch_order(item_count: int, *, seed: int, epoch: int) -> list[int]:
    order = list(range(item_count))
    random.Random(seed + epoch).shuffle(order)
    return order


def _batch_from_indices(
    dataset: LykenoxAlignedSpeechDataset,
    indices: list[int],
) -> AlignedSpeechBatch:
    return collate_aligned_speech([dataset[index] for index in indices]).to("cpu")


def _validation_metrics(
    model: LykenoxSpeechAcousticModel,
    dataset: LykenoxAlignedSpeechDataset,
    *,
    batch_size: int,
    duration_weight: float,
    f0_weight: float,
    voicing_weight: float,
) -> dict[str, float]:
    values: dict[str, list[float]] = {
        "total": [],
        "acoustic": [],
        "duration": [],
        "f0": [],
        "voicing": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            batch = _batch_from_indices(dataset, indices)
            current = _snapshot(
                _compute_losses(
                    model,
                    batch,
                    duration_weight=duration_weight,
                    f0_weight=f0_weight,
                    voicing_weight=voicing_weight,
                )
            )
            for name, value in current.items():
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite validation {name} loss")
                values[name].append(value)
    model.train()
    return {name: statistics.fmean(series) for name, series in values.items()}


def _run_config(
    *,
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
    return {
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "train_order_version": TRAIN_ORDER_VERSION,
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


def _checkpoint_metadata(
    *,
    initial_validation: dict[str, float],
    best_validation: dict[str, float],
    best_epoch: int,
    epochs_completed: int,
    no_improvement_epochs: int,
    history: list[dict[str, object]],
    accumulator: EpochAccumulator,
    epoch: int,
) -> dict[str, object]:
    return {
        "initial_validation": dict(initial_validation),
        "best_validation": dict(best_validation),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epochs_completed),
        "no_improvement_epochs": int(no_improvement_epochs),
        "history": list(history),
        "epoch_accumulator": accumulator.to_payload(epoch=epoch),
    }


def run_acoustic_prosody_training(
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
        raise ValueError("persistent prosody training requires non-zero F0/voicing weights")
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
        else root / "models" / "lykenox_identity" / "training" / "acoustic_prosody_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    progress_path = output_dir / "training_progress.json"
    report_path = output_dir / "training_report.json"

    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    frontend = SpanishTextFrontend()
    config = LykenoxSpeechConfig(vocab_size=frontend.vocab_size)
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
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        epoch = int(payload["epoch"])
        next_item_offset = int(payload["next_item_offset"])
        global_step = int(payload["global_step"])
        metadata = payload.get("training_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("Acoustic resume checkpoint is missing training metadata")
        initial_validation = dict(metadata["initial_validation"])
        best_validation = dict(metadata["best_validation"])
        best_epoch = int(metadata["best_epoch"])
        epochs_completed = int(metadata["epochs_completed"])
        no_improvement_epochs = int(metadata["no_improvement_epochs"])
        history = list(metadata.get("history", []))
        accumulator = EpochAccumulator.from_payload(
            metadata.get("epoch_accumulator"),
            epoch=epoch,
        )
    else:
        torch.manual_seed(seed)
        model = LykenoxSpeechAcousticModel(config).cpu().train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
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
            raise RuntimeError("Acoustic checkpoint next_item_offset is invalid")

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
                raise RuntimeError(f"Non-finite acoustic total loss at global step {global_step}")
            losses.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"Non-finite acoustic gradient at global step {global_step}")
            optimizer.step()

            accumulator.add(_snapshot(losses))
            next_item_offset += len(batch_indices)
            global_step += 1
            updates_this_run += 1

            if global_step % checkpoint_every_updates == 0:
                save_last()

        if stop_reason is not None:
            break

        # A checkpoint may intentionally resume at offset == len(order), meaning all
        # updates for this epoch were done but held-out validation still needs to run.
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
            "next_gate": "rerun_same_command_to_resume",
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
            "audit_persistent_acoustic_prosody_before_end_to_end"
            if status == "pass"
            else "review_persistent_acoustic_training_before_more_work"
        ),
        "warning": (
            "A pass closes persistent supervised acoustic training only. Product inference "
            "still requires corrected predicted-duration semantics and an end-to-end audit "
            "using predicted rather than waveform-derived F0/voicing."
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
            run_acoustic_prosody_training(
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
