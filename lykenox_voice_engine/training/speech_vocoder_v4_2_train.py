"""Bounded, exactly resumable persistent trainer for LYKENOX vocoder v4.2.

This trainer is intentionally separate from the historical v4.1 run.  It is designed for
the target Windows CPU and a command runner with a short wall-clock ceiling:

* each invocation stops below a conservative time budget and writes ``last.pt`` atomically;
* resume reconstructs the exact epoch crop/order and restores generator, discriminator,
  both optimizers and torch RNG state;
* held-out validation uses a fixed segment set and selects checkpoints with
  reconstruction + direct log-mel-envelope + target-relative spectral balance;
* adversarial/feature-matching training starts only after a reconstruction warmup;
* v4.1 artifacts are never loaded, mutated, or overwritten;
* persistent completion does not imply perceptual acceptance: the next gate is a full
  held-out oracle-utterance listening audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderGeneratorV42,
    VOCODER_GENERATOR_V4_2_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import PitchFrames, extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import (
    VocoderSegment,
    collect_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import (
    LogMelEnvelopeLoss,
    VOCODER_ENVELOPE_LOSS_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_losses import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import (
    build_v4_2_training_provenance,
    load_v4_2_checkpoint,
    save_v4_2_checkpoint,
)


TRAINER_CONTRACT_VERSION = "v4-2-bounded-resumable-v1"
TRAIN_SEGMENT_SCHEDULE_VERSION = "epoch-crop-shuffle-v1"
DEFAULT_TIME_BUDGET_SECONDS = 70.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 9.0
DEFAULT_VALIDATION_RESERVE_SECONDS = 16.0
DEFAULT_VALIDATION_SEED_OFFSET = 100_003


@dataclass(frozen=True)
class ConditionedSegment:
    segment: VocoderSegment
    pitch: PitchFrames


@dataclass
class EpochAccumulator:
    reconstruction_sum: float = 0.0
    envelope_sum: float = 0.0
    balance_sum: float = 0.0
    generator_total_sum: float = 0.0
    update_count: int = 0
    discriminator_sum: float = 0.0
    adversarial_sum: float = 0.0
    feature_matching_sum: float = 0.0
    adversarial_count: int = 0

    @classmethod
    def from_payload(cls, payload: object, *, epoch: int) -> "EpochAccumulator":
        if not isinstance(payload, dict) or int(payload.get("epoch", -1)) != int(epoch):
            return cls()
        return cls(
            reconstruction_sum=float(payload.get("reconstruction_sum", 0.0)),
            envelope_sum=float(payload.get("envelope_sum", 0.0)),
            balance_sum=float(payload.get("balance_sum", 0.0)),
            generator_total_sum=float(payload.get("generator_total_sum", 0.0)),
            update_count=int(payload.get("update_count", 0)),
            discriminator_sum=float(payload.get("discriminator_sum", 0.0)),
            adversarial_sum=float(payload.get("adversarial_sum", 0.0)),
            feature_matching_sum=float(payload.get("feature_matching_sum", 0.0)),
            adversarial_count=int(payload.get("adversarial_count", 0)),
        )

    def to_payload(self, *, epoch: int) -> dict[str, object]:
        return {
            "epoch": int(epoch),
            "reconstruction_sum": self.reconstruction_sum,
            "envelope_sum": self.envelope_sum,
            "balance_sum": self.balance_sum,
            "generator_total_sum": self.generator_total_sum,
            "update_count": self.update_count,
            "discriminator_sum": self.discriminator_sum,
            "adversarial_sum": self.adversarial_sum,
            "feature_matching_sum": self.feature_matching_sum,
            "adversarial_count": self.adversarial_count,
        }


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _condition_segments(segments: list[VocoderSegment]) -> list[ConditionedSegment]:
    return [
        ConditionedSegment(
            segment=segment,
            pitch=extract_pitch_frames(
                segment.waveform,
                frame_count=segment.mel_frames,
            ),
        )
        for segment in segments
    ]


def _generate(
    generator: LykenoxVocoderGeneratorV42,
    item: ConditionedSegment,
) -> torch.Tensor:
    return generator(
        item.segment.mel.unsqueeze(0),
        item.pitch.f0_hz.unsqueeze(0),
        item.pitch.voiced.unsqueeze(0),
    )


def _segment_set_sha256(segments: list[VocoderSegment]) -> str:
    rows = [
        {
            "split": segment.split,
            "utterance_id": segment.utterance_id,
            "wav_path": segment.wav_path,
            "start_frame": segment.start_frame,
            "mel_frames": segment.mel_frames,
        }
        for segment in segments
    ]
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _epoch_segments(
    root: Path,
    *,
    epoch: int,
    seed: int,
    segment_mel_frames: int,
    train_items: int,
) -> tuple[list[ConditionedSegment], int]:
    crop_seed = seed + epoch
    segments, skipped = collect_vocoder_segments(
        root,
        "train",
        segment_mel_frames=segment_mel_frames,
        max_items=train_items,
        seed=crop_seed,
    )
    order = list(range(len(segments)))
    random.Random(seed + 1_000_003 + epoch).shuffle(order)
    ordered = [segments[index] for index in order]
    return _condition_segments(ordered), len(skipped)


def _base_losses(
    generator: LykenoxVocoderGeneratorV42,
    envelope_loss: LogMelEnvelopeLoss,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    envelope_weight: float,
    balance_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction = multi_resolution_reconstruction_loss(prediction, target).total
    envelope = envelope_loss(prediction, target).total
    balance = target_relative_spectral_balance_loss(
        prediction,
        target,
        sample_rate=generator.config.sample_rate,
    ).loss
    total = reconstruction + envelope_weight * envelope + balance_weight * balance
    return total, {
        "reconstruction": reconstruction,
        "envelope": envelope,
        "balance": balance,
    }


def _validation_metrics(
    generator: LykenoxVocoderGeneratorV42,
    envelope_loss: LogMelEnvelopeLoss,
    items: list[ConditionedSegment],
    *,
    envelope_weight: float,
    balance_weight: float,
) -> tuple[float, float, float, float]:
    generator.eval()
    reconstruction_values: list[float] = []
    envelope_values: list[float] = []
    balance_values: list[float] = []
    with torch.no_grad():
        for item in items:
            prediction = _generate(generator, item)
            target = item.segment.waveform.unsqueeze(0)
            _total, losses = _base_losses(
                generator,
                envelope_loss,
                prediction,
                target,
                envelope_weight=envelope_weight,
                balance_weight=balance_weight,
            )
            values = [
                float(losses["reconstruction"].detach()),
                float(losses["envelope"].detach()),
                float(losses["balance"].detach()),
            ]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("Non-finite held-out v4.2 validation metric")
            reconstruction_values.append(values[0])
            envelope_values.append(values[1])
            balance_values.append(values[2])
    generator.train()
    reconstruction_mean = statistics.fmean(reconstruction_values)
    envelope_mean = statistics.fmean(envelope_values)
    balance_mean = statistics.fmean(balance_values)
    score = (
        reconstruction_mean
        + envelope_weight * envelope_mean
        + balance_weight * balance_mean
    )
    return reconstruction_mean, envelope_mean, balance_mean, score


def _run_config(
    *,
    segment_mel_frames: int,
    train_items: int,
    val_items: int,
    max_epochs: int,
    warmup_epochs: int,
    patience: int,
    seed: int,
    validation_seed: int,
    generator_lr: float,
    discriminator_lr: float,
    envelope_weight: float,
    balance_weight: float,
    adversarial_weight: float,
    feature_matching_weight: float,
    gradient_clip_norm: float,
    min_delta: float,
    checkpoint_every_updates: int,
    val_segment_set_sha256: str,
) -> dict[str, object]:
    return {
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "generator_architecture": VOCODER_GENERATOR_V4_2_ARCHITECTURE,
        "envelope_loss_version": VOCODER_ENVELOPE_LOSS_VERSION,
        "train_segment_schedule_version": TRAIN_SEGMENT_SCHEDULE_VERSION,
        "segment_mel_frames": int(segment_mel_frames),
        "train_items": int(train_items),
        "val_items": int(val_items),
        "max_epochs": int(max_epochs),
        "warmup_epochs": int(warmup_epochs),
        "patience": int(patience),
        "seed": int(seed),
        "validation_seed": int(validation_seed),
        "generator_lr": float(generator_lr),
        "discriminator_lr": float(discriminator_lr),
        "envelope_weight": float(envelope_weight),
        "balance_weight": float(balance_weight),
        "adversarial_weight": float(adversarial_weight),
        "feature_matching_weight": float(feature_matching_weight),
        "gradient_clip_norm": float(gradient_clip_norm),
        "min_delta": float(min_delta),
        "checkpoint_every_updates": int(checkpoint_every_updates),
        "validation_segment_set_sha256": val_segment_set_sha256,
    }


def _checkpoint_metadata(
    *,
    run_config: dict[str, object],
    history: list[dict[str, object]],
    initial_metrics: tuple[float, float, float, float],
    best_metrics: tuple[float, float, float, float],
    best_epoch: int,
    epochs_without_improvement: int,
    partial_epoch_state: dict[str, object] | None,
    resumed_invocations: int,
) -> dict[str, object]:
    initial_reconstruction, initial_envelope, initial_balance, initial_score = initial_metrics
    best_reconstruction, best_envelope, best_balance, best_score = best_metrics
    return {
        "purpose": "persistent_v4_2_envelope_first_training",
        "run_config": run_config,
        "history": history,
        "initial_validation_reconstruction": initial_reconstruction,
        "initial_validation_envelope": initial_envelope,
        "initial_validation_spectral_balance": initial_balance,
        "initial_validation_selection_score": initial_score,
        "best_validation_reconstruction": best_reconstruction,
        "best_validation_envelope": best_envelope,
        "best_validation_spectral_balance": best_balance,
        "best_validation_selection_score": best_score,
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "partial_epoch_state": partial_epoch_state,
        "resumed_invocations": int(resumed_invocations),
        "checkpoint_semantics": (
            "epoch is the 1-based epoch currently/next processed; next_item_offset is the "
            "next deterministic shuffled item within that epoch; checkpoints occur only "
            "between complete generator/discriminator updates"
        ),
    }


def _metrics_from_metadata(metadata: dict[str, object], prefix: str) -> tuple[float, float, float, float]:
    return (
        float(metadata[f"{prefix}_validation_reconstruction"]),
        float(metadata[f"{prefix}_validation_envelope"]),
        float(metadata[f"{prefix}_validation_spectral_balance"]),
        float(metadata[f"{prefix}_validation_selection_score"]),
    )


def _atomic_checkpoint(
    path: Path,
    generator: LykenoxVocoderGeneratorV42,
    discriminator: LykenoxMultiScaleWaveformDiscriminator,
    **kwargs: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_v4_2_checkpoint(temporary, generator, discriminator, **kwargs)
    os.replace(temporary, path)


def _save_state(
    path: Path,
    *,
    generator: LykenoxVocoderGeneratorV42,
    discriminator: LykenoxMultiScaleWaveformDiscriminator,
    generator_optimizer: torch.optim.Optimizer | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    next_item_offset: int,
    provenance: dict[str, object],
    metadata: dict[str, object],
    validation_metrics: tuple[float, float, float, float],
) -> None:
    reconstruction, envelope, balance, score = validation_metrics
    _atomic_checkpoint(
        path,
        generator,
        discriminator,
        epoch=epoch,
        global_step=global_step,
        next_item_offset=next_item_offset,
        validation_reconstruction_loss=reconstruction,
        validation_envelope_loss=envelope,
        validation_spectral_balance_loss=balance,
        validation_selection_score=score,
        training_provenance=provenance,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        training_metadata=metadata,
    )


def _latest_validation(
    history: list[dict[str, object]],
    initial_metrics: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not history:
        return initial_metrics
    row = history[-1]
    return (
        float(row["validation_reconstruction"]),
        float(row["validation_envelope"]),
        float(row["validation_spectral_balance"]),
        float(row["validation_selection_score"]),
    )


def run_bounded_resumable_v4_2_training(
    root: Path,
    *,
    segment_mel_frames: int = 64,
    train_items: int = 118,
    val_items: int = 14,
    max_epochs: int = 28,
    warmup_epochs: int = 4,
    patience: int = 6,
    seed: int = 2420,
    generator_lr: float = 2e-4,
    discriminator_lr: float = 1e-4,
    envelope_weight: float = 0.50,
    balance_weight: float = 0.25,
    adversarial_weight: float = 0.03,
    feature_matching_weight: float = 0.50,
    gradient_clip_norm: float = 5.0,
    min_delta: float = 1e-4,
    checkpoint_every_updates: int = 8,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    checkpoint_reserve_seconds: float = DEFAULT_CHECKPOINT_RESERVE_SECONDS,
    validation_reserve_seconds: float = DEFAULT_VALIDATION_RESERVE_SECONDS,
    max_updates_this_run: int | None = None,
    artifact_dir_override: Path | None = None,
) -> dict[str, object]:
    if segment_mel_frames < 32:
        raise ValueError("segment_mel_frames must be >= 32")
    if train_items < 2 or val_items < 2:
        raise ValueError("train_items and val_items must both be >= 2")
    if max_epochs < 1 or not 0 <= warmup_epochs <= max_epochs:
        raise ValueError("invalid epoch configuration")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if generator_lr <= 0.0 or discriminator_lr <= 0.0:
        raise ValueError("learning rates must be positive")
    if envelope_weight <= 0.0 or balance_weight <= 0.0:
        raise ValueError("envelope/balance weights must be positive")
    if adversarial_weight < 0.0 or feature_matching_weight < 0.0:
        raise ValueError("adversarial weights must be non-negative")
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive")
    if checkpoint_every_updates < 1:
        raise ValueError("checkpoint_every_updates must be >= 1")
    if time_budget_seconds <= checkpoint_reserve_seconds + 5.0:
        raise ValueError("time budget is too small for safe checkpointing")
    if max_updates_this_run is not None and max_updates_this_run < 1:
        raise ValueError("max_updates_this_run must be positive when supplied")

    root = Path(root).resolve()
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    validation_seed = seed + DEFAULT_VALIDATION_SEED_OFFSET
    val_segments, val_skipped = collect_vocoder_segments(
        root,
        "val",
        segment_mel_frames=segment_mel_frames,
        max_items=val_items,
        seed=validation_seed,
    )
    val_conditioned = _condition_segments(val_segments)
    val_set_sha = _segment_set_sha256(val_segments)

    provenance = build_v4_2_training_provenance(
        root,
        segment_mel_frames=segment_mel_frames,
        seed=seed,
    )
    run_config = _run_config(
        segment_mel_frames=segment_mel_frames,
        train_items=train_items,
        val_items=val_items,
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        patience=patience,
        seed=seed,
        validation_seed=validation_seed,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        envelope_weight=envelope_weight,
        balance_weight=balance_weight,
        adversarial_weight=adversarial_weight,
        feature_matching_weight=feature_matching_weight,
        gradient_clip_norm=gradient_clip_norm,
        min_delta=min_delta,
        checkpoint_every_updates=checkpoint_every_updates,
        val_segment_set_sha256=val_set_sha,
    )

    artifact_dir = (
        Path(artifact_dir_override).resolve()
        if artifact_dir_override is not None
        else root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_source_filter_v4_2"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    last_path = artifact_dir / "last.pt"
    best_path = artifact_dir / "best.pt"
    progress_path = artifact_dir / "training_progress.json"
    report_path = artifact_dir / "training_report.json"

    envelope_loss = LogMelEnvelopeLoss().cpu()
    resumed_payload: dict[str, object] | None = None
    if last_path.exists():
        generator, discriminator, resumed_payload = load_v4_2_checkpoint(last_path)
        if resumed_payload.get("training_provenance") != provenance:
            raise RuntimeError(
                "Existing v4.2 last.pt provenance differs from the active dataset/features"
            )
        metadata = resumed_payload.get("training_metadata")
        if not isinstance(metadata, dict) or metadata.get("run_config") != run_config:
            raise RuntimeError(
                "Existing v4.2 last.pt training configuration differs from this command"
            )
        epoch = int(resumed_payload.get("epoch", 1))
        next_item_offset = int(resumed_payload.get("next_item_offset", 0))
        global_step = int(resumed_payload.get("global_step", 0))
        history = list(metadata.get("history", []))
        initial_metrics = _metrics_from_metadata(metadata, "initial")
        best_metrics = _metrics_from_metadata(metadata, "best")
        best_epoch = int(metadata.get("best_epoch", 0))
        epochs_without_improvement = int(metadata.get("epochs_without_improvement", 0))
        accumulator = EpochAccumulator.from_payload(
            metadata.get("partial_epoch_state"),
            epoch=epoch,
        )
        resumed_invocations = int(metadata.get("resumed_invocations", 0)) + 1
        rng_state = resumed_payload.get("torch_rng_state")
        if isinstance(rng_state, torch.Tensor):
            torch.set_rng_state(rng_state)
        generator.train()
        discriminator.train()
    else:
        torch.manual_seed(seed)
        generator = LykenoxVocoderGeneratorV42().cpu().train()
        discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
        initial_metrics = _validation_metrics(
            generator,
            envelope_loss,
            val_conditioned,
            envelope_weight=envelope_weight,
            balance_weight=balance_weight,
        )
        best_metrics = initial_metrics
        epoch = 1
        next_item_offset = 0
        global_step = 0
        history: list[dict[str, object]] = []
        best_epoch = 0
        epochs_without_improvement = 0
        accumulator = EpochAccumulator()
        resumed_invocations = 0

    generator_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=generator_lr,
        weight_decay=1e-5,
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=discriminator_lr,
        weight_decay=1e-5,
    )
    if resumed_payload is not None:
        generator_state = resumed_payload.get("generator_optimizer_state")
        discriminator_state = resumed_payload.get("discriminator_optimizer_state")
        if not isinstance(generator_state, dict) or not isinstance(discriminator_state, dict):
            raise RuntimeError("Cannot exactly resume v4.2 without both optimizer states")
        generator_optimizer.load_state_dict(generator_state)
        discriminator_optimizer.load_state_dict(discriminator_state)

    update_times: list[float] = []
    updates_this_run = 0
    train_skipped_latest = 0

    def current_metadata(partial: dict[str, object] | None) -> dict[str, object]:
        return _checkpoint_metadata(
            run_config=run_config,
            history=history,
            initial_metrics=initial_metrics,
            best_metrics=best_metrics,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            partial_epoch_state=partial,
            resumed_invocations=resumed_invocations,
        )

    def save_interruption(reason: str) -> dict[str, object]:
        partial = accumulator.to_payload(epoch=epoch) if next_item_offset > 0 else None
        metadata = current_metadata(partial)
        latest = _latest_validation(history, initial_metrics)
        _save_state(
            last_path,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            epoch=epoch,
            global_step=global_step,
            next_item_offset=next_item_offset,
            provenance=provenance,
            metadata=metadata,
            validation_metrics=latest,
        )
        progress = {
            "status": "incomplete",
            "device": "cpu",
            "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "architecture": VOCODER_GENERATOR_V4_2_ARCHITECTURE,
            "stop_reason": reason,
            "epochs_completed": len(history),
            "current_epoch": epoch,
            "next_item_offset": next_item_offset,
            "global_step": global_step,
            "updates_this_run": updates_this_run,
            "best_epoch": best_epoch,
            "best_validation_selection_score": round(best_metrics[3], 6),
            "best_validation_envelope": round(best_metrics[1], 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "mean_update_seconds_this_run": (
                round(statistics.fmean(update_times), 4) if update_times else None
            ),
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
            "persistent_training_complete": False,
            "v4_1_checkpoint_mutated": False,
            "next_gate": "rerun_same_command_to_resume",
        }
        _atomic_json(progress_path, progress)
        return {**progress, "progress_report": str(progress_path)}

    while epoch <= max_epochs:
        train_conditioned, train_skipped_latest = _epoch_segments(
            root,
            epoch=epoch,
            seed=seed,
            segment_mel_frames=segment_mel_frames,
            train_items=train_items,
        )
        if next_item_offset > len(train_conditioned):
            raise RuntimeError("v4.2 resume offset exceeds deterministic epoch item count")

        while next_item_offset < len(train_conditioned):
            elapsed = time.perf_counter() - started
            estimated_update = max(update_times[-4:] or [2.0])
            is_last_item = next_item_offset == len(train_conditioned) - 1
            required = checkpoint_reserve_seconds + estimated_update
            if is_last_item:
                required += validation_reserve_seconds
            if elapsed + required >= time_budget_seconds:
                return save_interruption("time_budget")
            if max_updates_this_run is not None and updates_this_run >= max_updates_this_run:
                return save_interruption("max_updates_this_run")

            item = train_conditioned[next_item_offset]
            mel = item.segment.mel.unsqueeze(0)
            f0_hz = item.pitch.f0_hz.unsqueeze(0)
            voiced = item.pitch.voiced.unsqueeze(0)
            target = item.segment.waveform.unsqueeze(0)

            update_started = time.perf_counter()
            generator_optimizer.zero_grad(set_to_none=True)
            prediction = generator(mel, f0_hz, voiced)
            generator_total, base = _base_losses(
                generator,
                envelope_loss,
                prediction,
                target,
                envelope_weight=envelope_weight,
                balance_weight=balance_weight,
            )

            adversarial_value = 0.0
            feature_matching_value = 0.0
            discriminator_value = 0.0
            adversarial_active = epoch > warmup_epochs and (
                adversarial_weight > 0.0 or feature_matching_weight > 0.0
            )
            if adversarial_active:
                _set_requires_grad(discriminator, False)
                with torch.no_grad():
                    real_features = discriminator(target)
                fake_features = discriminator(prediction)
                adversarial = generator_adversarial_loss(fake_features)
                matching = feature_matching_loss(real_features, fake_features)
                generator_total = (
                    generator_total
                    + adversarial_weight * adversarial
                    + feature_matching_weight * matching
                )
                adversarial_value = float(adversarial.detach())
                feature_matching_value = float(matching.detach())

            if not bool(torch.isfinite(generator_total)):
                raise RuntimeError("Non-finite v4.2 generator loss")
            generator_total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=gradient_clip_norm,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Non-finite v4.2 generator gradient norm")
            generator_optimizer.step()
            _set_requires_grad(discriminator, True)

            if adversarial_active:
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_output = discriminator(target)
                fake_output = discriminator(prediction.detach())
                discriminator_loss = discriminator_hinge_loss(real_output, fake_output)
                if not bool(torch.isfinite(discriminator_loss)):
                    raise RuntimeError("Non-finite v4.2 discriminator loss")
                discriminator_loss.backward()
                discriminator_norm = torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    max_norm=gradient_clip_norm,
                )
                if not bool(torch.isfinite(discriminator_norm)):
                    raise RuntimeError("Non-finite v4.2 discriminator gradient norm")
                discriminator_optimizer.step()
                discriminator_value = float(discriminator_loss.detach())

            reconstruction_value = float(base["reconstruction"].detach())
            envelope_value = float(base["envelope"].detach())
            balance_value = float(base["balance"].detach())
            total_value = float(generator_total.detach())
            if not all(
                math.isfinite(value)
                for value in (
                    reconstruction_value,
                    envelope_value,
                    balance_value,
                    total_value,
                    adversarial_value,
                    feature_matching_value,
                    discriminator_value,
                )
            ):
                raise RuntimeError("Non-finite v4.2 training metric")

            accumulator.reconstruction_sum += reconstruction_value
            accumulator.envelope_sum += envelope_value
            accumulator.balance_sum += balance_value
            accumulator.generator_total_sum += total_value
            accumulator.update_count += 1
            if adversarial_active:
                accumulator.adversarial_sum += adversarial_value
                accumulator.feature_matching_sum += feature_matching_value
                accumulator.discriminator_sum += discriminator_value
                accumulator.adversarial_count += 1

            global_step += 1
            updates_this_run += 1
            next_item_offset += 1
            update_times.append(time.perf_counter() - update_started)

            if global_step % checkpoint_every_updates == 0:
                metadata = current_metadata(accumulator.to_payload(epoch=epoch))
                latest = _latest_validation(history, initial_metrics)
                _save_state(
                    last_path,
                    generator=generator,
                    discriminator=discriminator,
                    generator_optimizer=generator_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    next_item_offset=next_item_offset,
                    provenance=provenance,
                    metadata=metadata,
                    validation_metrics=latest,
                )

        validation = _validation_metrics(
            generator,
            envelope_loss,
            val_conditioned,
            envelope_weight=envelope_weight,
            balance_weight=balance_weight,
        )
        update_count = max(1, accumulator.update_count)
        adversarial_count = max(1, accumulator.adversarial_count)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_reconstruction": accumulator.reconstruction_sum / update_count,
            "train_envelope": accumulator.envelope_sum / update_count,
            "train_spectral_balance": accumulator.balance_sum / update_count,
            "train_generator_total": accumulator.generator_total_sum / update_count,
            "train_adversarial": (
                accumulator.adversarial_sum / adversarial_count
                if accumulator.adversarial_count
                else 0.0
            ),
            "train_feature_matching": (
                accumulator.feature_matching_sum / adversarial_count
                if accumulator.adversarial_count
                else 0.0
            ),
            "train_discriminator": (
                accumulator.discriminator_sum / adversarial_count
                if accumulator.adversarial_count
                else 0.0
            ),
            "adversarial_active": epoch > warmup_epochs,
            "validation_reconstruction": validation[0],
            "validation_envelope": validation[1],
            "validation_spectral_balance": validation[2],
            "validation_selection_score": validation[3],
        }
        history.append(row)

        if validation[3] < best_metrics[3] - min_delta:
            best_metrics = validation
            best_epoch = epoch
            epochs_without_improvement = 0
            best_metadata = current_metadata(None)
            _save_state(
                best_path,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                next_item_offset=0,
                provenance=provenance,
                metadata=best_metadata,
                validation_metrics=validation,
            )
        else:
            epochs_without_improvement += 1

        epoch += 1
        next_item_offset = 0
        accumulator = EpochAccumulator()
        latest_metadata = current_metadata(None)
        _save_state(
            last_path,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            epoch=epoch,
            global_step=global_step,
            next_item_offset=0,
            provenance=provenance,
            metadata=latest_metadata,
            validation_metrics=validation,
        )

        if len(history) >= warmup_epochs and epochs_without_improvement >= patience:
            stop_reason = "early_stopping"
            break
    else:
        stop_reason = "max_epochs"

    training_improved = best_metrics[3] < initial_metrics[3]
    envelope_improved = best_metrics[1] < initial_metrics[1]
    completed_pass = training_improved and envelope_improved and best_path.exists()
    report = {
        "status": "pass" if completed_pass else "fail",
        "device": "cpu",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "architecture": VOCODER_GENERATOR_V4_2_ARCHITECTURE,
        "envelope_loss_version": VOCODER_ENVELOPE_LOSS_VERSION,
        "stop_reason": stop_reason,
        "epochs_completed": len(history),
        "global_step": global_step,
        "best_epoch": best_epoch,
        "initial_validation": {
            "reconstruction": round(initial_metrics[0], 6),
            "envelope": round(initial_metrics[1], 6),
            "spectral_balance": round(initial_metrics[2], 6),
            "selection_score": round(initial_metrics[3], 6),
        },
        "best_validation": {
            "reconstruction": round(best_metrics[0], 6),
            "envelope": round(best_metrics[1], 6),
            "spectral_balance": round(best_metrics[2], 6),
            "selection_score": round(best_metrics[3], 6),
        },
        "training_improved": training_improved,
        "envelope_improved": envelope_improved,
        "train_items": train_items,
        "val_items": len(val_conditioned),
        "val_skipped": len(val_skipped),
        "train_skipped_latest_epoch": train_skipped_latest,
        "segment_mel_frames": segment_mel_frames,
        "resumed_invocations": resumed_invocations,
        "elapsed_seconds_this_invocation": round(time.perf_counter() - started, 3),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "last_checkpoint": str(last_path),
        "persistent_training_complete": True,
        "full_utterance_perceptual_acceptance": False,
        "v4_1_checkpoint_mutated": False,
        "reference_audio_required_for_product_inference": False,
        "next_gate": (
            "run_v4_2_full_utterance_oracle_acceptance"
            if completed_pass
            else "inspect_v4_2_training_failure_before_any_more_training"
        ),
    }
    _atomic_json(report_path, report)
    _atomic_json(progress_path, report)
    return {**report, "training_report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_bounded_resumable_v4_2_training(
                args.root,
                time_budget_seconds=args.time_budget_seconds,
                max_updates_this_run=args.max_updates_this_run,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
