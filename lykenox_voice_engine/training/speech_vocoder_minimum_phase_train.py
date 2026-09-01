"""Bounded resumable trainer for the owned minimum-phase vocoder.

This is the first training path for this architecture that is wired exclusively to the
architecture-coupled Loss V2 objective.  It trains only the owned frame-rate cepstral
predictor; excitation and rendering remain fixed DSP.  Runs are update-bounded, exactly
resumable from ``last.pt`` and select ``best.pt`` only from fixed validation segments.

Metrics may reject instability but cannot accept voice quality.  Audible acceptance is done
separately on complete held-out utterances rendered from ``best.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_v1 import (
    PREDICTOR_ARCHITECTURE,
    LykenoxFrameRateCepstralPredictorV1,
)
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    OwnedVocoderSegment,
    collect_owned_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_artifact import (
    CHECKPOINT_SCHEMA_VERSION,
    load_minimum_phase_checkpoint,
    save_minimum_phase_checkpoint,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_objective import (
    ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION,
    ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
    OwnedMinimumPhaseObjectiveV2,
    active_weights,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    RENDERER_VERSION,
    render_owned_minimum_phase_vocoder_path,
)


TRAINER_VERSION = "owned-minimum-phase-resumable-trainer-v1"
TRAIN_ORDER_VERSION = "epoch-permutation-v1"
DEFAULT_SEGMENT_MEL_FRAMES = 64
DEFAULT_TRAIN_ITEMS = 48
DEFAULT_VAL_ITEMS = 12
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_UPDATES = 400
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_EVAL_EVERY = 25
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_DATA_SEED = 20260901
DEFAULT_MODEL_SEED = 20260903
DEFAULT_ORDER_SEED = 20260905
DEFAULT_NOISE_SEED = 97


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _run_config(
    *,
    segment_mel_frames: int,
    train_items: int,
    val_items: int,
    batch_size: int,
    max_updates: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    eval_every: int,
    checkpoint_every: int,
    data_seed: int,
    model_seed: int,
    order_seed: int,
    noise_seed: int,
) -> dict[str, object]:
    return {
        "trainer_version": TRAINER_VERSION,
        "train_order_version": TRAIN_ORDER_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "predictor_architecture": PREDICTOR_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "objective_version": ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
        "loss_weight_contract_version": ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION,
        "data_contract_version": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "active_weights": active_weights(),
        "segment_mel_frames": int(segment_mel_frames),
        "train_items": int(train_items),
        "val_items": int(val_items),
        "batch_size": int(batch_size),
        "max_updates": int(max_updates),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "eval_every": int(eval_every),
        "checkpoint_every": int(checkpoint_every),
        "data_seed": int(data_seed),
        "model_seed": int(model_seed),
        "order_seed": int(order_seed),
        "noise_seed": int(noise_seed),
    }


def _epoch_order(count: int, *, order_seed: int, epoch: int) -> list[int]:
    if count < 1 or epoch < 1:
        raise ValueError("count and epoch must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(order_seed) + int(epoch) * 1000003)
    return torch.randperm(count, generator=generator).tolist()


def _stack_batch(segments: list[OwnedVocoderSegment], indices: list[int]) -> tuple[torch.Tensor, ...]:
    selected = [segments[index] for index in indices]
    if not selected:
        raise ValueError("empty minimum-phase training batch")
    for segment in selected:
        if segment.conditioning_contract_version != OWNED_VOCODER_SEGMENT_CONTRACT_VERSION:
            raise RuntimeError("minimum-phase trainer received the wrong data contract")
    return (
        torch.stack([item.mel for item in selected], dim=0),
        torch.stack([item.f0_hz for item in selected], dim=0),
        torch.stack([item.voiced for item in selected], dim=0),
        torch.stack([item.periodicity for item in selected], dim=0),
        torch.stack([item.waveform for item in selected], dim=0),
    )


def _forward(
    model: LykenoxFrameRateCepstralPredictorV1,
    objective: OwnedMinimumPhaseObjectiveV2,
    batch: tuple[torch.Tensor, ...],
    *,
    noise_seed: int,
) -> tuple[Any, torch.Tensor]:
    mel, f0_hz, voiced, periodicity, target = batch
    cepstrum = model(mel, f0_hz, voiced, periodicity)
    prediction, _ = render_owned_minimum_phase_vocoder_path(
        cepstrum,
        f0_hz,
        voiced,
        periodicity,
        noise_seed=noise_seed,
    )
    expected_samples = mel.shape[1] * HOP_LENGTH
    if prediction.shape[-1] != expected_samples:
        raise RuntimeError("minimum-phase trainer violated exact output-length contract")
    losses = objective(prediction, target, mel)
    return losses, prediction


def _metric_row(losses: Any) -> dict[str, float]:
    return {
        "total": float(losses.total.detach()),
        "reconstruction": float(losses.reconstruction.detach()),
        "envelope": float(losses.envelope.detach()),
        "presence": float(losses.presence.detach()),
        "spectral_balance": float(losses.spectral_balance.detach()),
    }


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("cannot average empty metric rows")
    keys = tuple(rows[0].keys())
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def _evaluate(
    model: LykenoxFrameRateCepstralPredictorV1,
    objective: OwnedMinimumPhaseObjectiveV2,
    segments: list[OwnedVocoderSegment],
    *,
    batch_size: int,
    noise_seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for offset in range(0, len(segments), batch_size):
            indices = list(range(offset, min(offset + batch_size, len(segments))))
            losses, _ = _forward(
                model,
                objective,
                _stack_batch(segments, indices),
                noise_seed=noise_seed,
            )
            rows.append(_metric_row(losses))
    if was_training:
        model.train()
    return _mean_rows(rows)


def run_minimum_phase_training(
    root: Path,
    *,
    output_dir: Path | None = None,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    train_items: int = DEFAULT_TRAIN_ITEMS,
    val_items: int = DEFAULT_VAL_ITEMS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_updates: int = DEFAULT_MAX_UPDATES,
    max_updates_this_run: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    eval_every: int = DEFAULT_EVAL_EVERY,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    data_seed: int = DEFAULT_DATA_SEED,
    model_seed: int = DEFAULT_MODEL_SEED,
    order_seed: int = DEFAULT_ORDER_SEED,
    noise_seed: int = DEFAULT_NOISE_SEED,
) -> dict[str, object]:
    if segment_mel_frames < 16 or train_items < 2 or val_items < 1 or batch_size < 1:
        raise ValueError("invalid minimum-phase data configuration")
    if max_updates < 1 or (max_updates_this_run is not None and max_updates_this_run < 1):
        raise ValueError("update budgets must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0 or grad_clip <= 0.0:
        raise ValueError("invalid optimizer configuration")
    if eval_every < 1 or checkpoint_every < 1:
        raise ValueError("eval/checkpoint cadence must be positive")

    root = Path(root).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "training" / "vocoder_minimum_phase_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    progress_path = output_dir / "training_progress.json"
    report_path = output_dir / "training_report.json"

    run_config = _run_config(
        segment_mel_frames=segment_mel_frames,
        train_items=train_items,
        val_items=val_items,
        batch_size=batch_size,
        max_updates=max_updates,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        data_seed=data_seed,
        model_seed=model_seed,
        order_seed=order_seed,
        noise_seed=noise_seed,
    )
    val_segments, _ = collect_owned_vocoder_segments(
        root,
        "val",
        segment_mel_frames=segment_mel_frames,
        max_items=val_items,
        seed=data_seed + 50000000,
    )
    objective = OwnedMinimumPhaseObjectiveV2().cpu()

    if last_path.exists():
        model, payload = load_minimum_phase_checkpoint(
            last_path,
            expected_run_config=run_config,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        progress = dict(payload["progress"])
        history = list(payload["history"])
        epoch = int(progress["epoch"])
        next_item_offset = int(progress["next_item_offset"])
        global_step = int(progress["global_step"])
        best_val_total = float(progress["best_val_total"])
        best_step = int(progress["best_step"])
        initial_validation = dict(progress["initial_validation"])
        clipped_update_count = int(progress.get("clipped_update_count", 0))
    else:
        torch.manual_seed(model_seed)
        model = LykenoxFrameRateCepstralPredictorV1().cpu().train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        initial_validation = _evaluate(
            model,
            objective,
            val_segments,
            batch_size=batch_size,
            noise_seed=noise_seed,
        )
        epoch = 1
        next_item_offset = 0
        global_step = 0
        best_val_total = float(initial_validation["total"])
        best_step = 0
        history: list[dict[str, object]] = []
        clipped_update_count = 0

    updates_this_run = 0

    def progress_payload() -> dict[str, object]:
        return {
            "epoch": epoch,
            "next_item_offset": next_item_offset,
            "global_step": global_step,
            "best_val_total": best_val_total,
            "best_step": best_step,
            "initial_validation": initial_validation,
            "clipped_update_count": clipped_update_count,
        }

    def save_last() -> None:
        save_minimum_phase_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            run_config=run_config,
            progress=progress_payload(),
            history=history,
        )

    while global_step < max_updates:
        train_segments, _ = collect_owned_vocoder_segments(
            root,
            "train",
            segment_mel_frames=segment_mel_frames,
            max_items=train_items,
            seed=data_seed + epoch,
        )
        order = _epoch_order(len(train_segments), order_seed=order_seed, epoch=epoch)
        if next_item_offset < 0 or next_item_offset > len(order):
            raise RuntimeError("minimum-phase checkpoint has invalid item offset")

        while next_item_offset < len(order) and global_step < max_updates:
            if max_updates_this_run is not None and updates_this_run >= max_updates_this_run:
                save_last()
                report = {
                    "status": "incomplete",
                    "stop_reason": "max_updates_this_run_reached",
                    "trainer_version": TRAINER_VERSION,
                    "global_step": global_step,
                    "epoch": epoch,
                    "next_item_offset": next_item_offset,
                    "updates_this_run": updates_this_run,
                    "initial_validation": initial_validation,
                    "best_val_total": best_val_total,
                    "best_step": best_step,
                    "last_checkpoint": str(last_path),
                    "best_checkpoint": str(best_path) if best_path.exists() else None,
                    "active_weights": active_weights(),
                }
                _atomic_json(progress_path, report)
                return report

            batch_indices = order[
                next_item_offset : min(next_item_offset + batch_size, len(order))
            ]
            batch = _stack_batch(train_segments, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            losses, prediction = _forward(
                model,
                objective,
                batch,
                noise_seed=noise_seed,
            )
            if not torch.isfinite(losses.total):
                raise RuntimeError(f"non-finite minimum-phase loss at step {global_step}")
            losses.total.backward()
            raw_grad_norm = float(
                torch.sqrt(
                    sum(
                        parameter.grad.detach().square().sum()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                )
            )
            if not math.isfinite(raw_grad_norm) or raw_grad_norm <= 0.0:
                raise RuntimeError(f"invalid minimum-phase gradient at step {global_step}")
            if raw_grad_norm > grad_clip:
                clipped_update_count += 1
            clip_return = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            if not math.isfinite(clip_return):
                raise RuntimeError(f"non-finite clipped gradient at step {global_step}")
            optimizer.step()

            global_step += 1
            updates_this_run += 1
            next_item_offset += len(batch_indices)
            train_row = _metric_row(losses)
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "train": train_row,
                    "raw_gradient_norm": raw_grad_norm,
                    "gradient_clipped": raw_grad_norm > grad_clip,
                    "prediction_samples": int(prediction.shape[-1]),
                }
            )

            if global_step % eval_every == 0 or global_step == max_updates:
                validation = _evaluate(
                    model,
                    objective,
                    val_segments,
                    batch_size=batch_size,
                    noise_seed=noise_seed,
                )
                improved = validation["total"] < best_val_total
                history[-1]["validation"] = validation
                history[-1]["validation_improved"] = bool(improved)
                if improved:
                    best_val_total = float(validation["total"])
                    best_step = global_step
                    save_minimum_phase_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        run_config=run_config,
                        progress=progress_payload(),
                        history=history,
                    )

            if global_step % checkpoint_every == 0:
                save_last()

        if next_item_offset >= len(order):
            epoch += 1
            next_item_offset = 0

    save_last()
    final_validation = _evaluate(
        model,
        objective,
        val_segments,
        batch_size=batch_size,
        noise_seed=noise_seed,
    )
    validation_improved = best_step > 0 and best_val_total < float(initial_validation["total"])
    report = {
        "status": "pass" if validation_improved and best_path.exists() else "needs_review",
        "stop_reason": "max_updates_reached",
        "trainer_version": TRAINER_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "objective_version": ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
        "loss_weight_contract_version": ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION,
        "active_weights": active_weights(),
        "global_step": global_step,
        "epochs_reached": epoch,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "best_val_total": best_val_total,
        "best_step": best_step,
        "validation_improved": validation_improved,
        "clipped_update_count": clipped_update_count,
        "clipped_update_fraction": clipped_update_count / max(global_step, 1),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "metrics_accept_voice_quality": False,
        "full_held_out_audio_required": True,
        "next_action": "render_complete_val_utterances_from_best_checkpoint",
    }
    _atomic_json(report_path, report)
    _atomic_json(progress_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=DEFAULT_SEGMENT_MEL_FRAMES)
    parser.add_argument("--train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--val-items", type=int, default=DEFAULT_VAL_ITEMS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    args = parser.parse_args()
    print(
        json.dumps(
            run_minimum_phase_training(
                args.root,
                segment_mel_frames=args.segment_mel_frames,
                train_items=args.train_items,
                val_items=args.val_items,
                batch_size=args.batch_size,
                max_updates=args.max_updates,
                max_updates_this_run=args.max_updates_this_run,
                learning_rate=args.learning_rate,
                grad_clip=args.grad_clip,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
