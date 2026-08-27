"""Bounded, exactly resumable persistent training for LYKENOX vocoder v4.1.

This is the first real training loop for the accepted source-filter candidate.  It is
designed for the target Windows CPU and for a command runner that may impose a short wall
clock limit:

- every invocation has a conservative wall-clock budget;
- ``last.pt`` records the exact next epoch/item position plus both optimizers and torch RNG;
- the per-epoch train segment start changes deterministically, so repeated epochs cover
  more of the owned recordings instead of memorizing one fixed crop;
- held-out validation uses a fixed segment set across the whole run;
- best-checkpoint selection uses reconstruction + target-relative spectral balance;
- periodic checkpointing limits lost work even if the process is interrupted unexpectedly;
- completed runs write three held-out listening pairs and re-run the known artifact gates.

Rerun the same command to continue.  Training configuration is part of the checkpoint and
a mismatched command is rejected instead of silently resuming a different experiment.
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

import numpy as np
import soundfile as sf
import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderGeneratorV41,
    VOCODER_GENERATOR_V4_1_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import PitchFrames, extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import (
    VocoderSegment,
    collect_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_losses import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_periodicity_probe import (
    _generated_specific_frame_lock,
)
from lykenox_voice_engine.training.speech_vocoder_polyphase_forensic import (
    _pitch_metrics,
    _spectral_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    spectral_band_fractions,
    target_relative_spectral_balance_loss,
)
from lykenox_voice_engine.training.speech_vocoder_source_filter_artifact import (
    build_source_filter_training_provenance,
    load_source_filter_checkpoint,
    save_source_filter_checkpoint,
)


TRAINER_CONTRACT_VERSION = "source-filter-bounded-resumable-v1"
TRAIN_SEGMENT_SCHEDULE_VERSION = "epoch-seed-v1"
DEFAULT_TIME_BUDGET_SECONDS = 70.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 8.0
DEFAULT_VALIDATION_SEED_OFFSET = 100_003


@dataclass(frozen=True)
class ConditionedSegment:
    segment: VocoderSegment
    pitch: PitchFrames


@dataclass
class EpochAccumulator:
    reconstruction_sum: float = 0.0
    balance_sum: float = 0.0
    update_count: int = 0
    discriminator_sum: float = 0.0
    adversarial_sum: float = 0.0
    feature_matching_sum: float = 0.0
    adversarial_count: int = 0

    @classmethod
    def from_payload(cls, payload: object, *, epoch: int) -> "EpochAccumulator":
        if not isinstance(payload, dict):
            return cls()
        if int(payload.get("epoch", -1)) != int(epoch):
            return cls()
        return cls(
            reconstruction_sum=float(payload.get("reconstruction_sum", 0.0)),
            balance_sum=float(payload.get("balance_sum", 0.0)),
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
            "balance_sum": self.balance_sum,
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
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
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
    generator: LykenoxVocoderGeneratorV41,
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


def _validation_metrics(
    generator: LykenoxVocoderGeneratorV41,
    items: list[ConditionedSegment],
    *,
    balance_weight: float,
) -> tuple[float, float, float]:
    generator.eval()
    reconstruction_values: list[float] = []
    balance_values: list[float] = []
    with torch.no_grad():
        for item in items:
            prediction = _generate(generator, item)
            target = item.segment.waveform.unsqueeze(0)
            reconstruction = multi_resolution_reconstruction_loss(
                prediction,
                target,
            ).total
            balance = target_relative_spectral_balance_loss(
                prediction,
                target,
                sample_rate=generator.config.sample_rate,
            ).loss
            reconstruction_value = float(reconstruction.detach().cpu())
            balance_value = float(balance.detach().cpu())
            if not math.isfinite(reconstruction_value) or not math.isfinite(balance_value):
                raise RuntimeError("Non-finite held-out source-filter validation metric")
            reconstruction_values.append(reconstruction_value)
            balance_values.append(balance_value)
    generator.train()
    reconstruction_mean = statistics.fmean(reconstruction_values)
    balance_mean = statistics.fmean(balance_values)
    return (
        reconstruction_mean,
        balance_mean,
        reconstruction_mean + balance_weight * balance_mean,
    )


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
    balance_weight: float,
    adversarial_weight: float,
    feature_matching_weight: float,
    min_delta: float,
    checkpoint_every_updates: int,
    val_segment_set_sha256: str,
) -> dict[str, object]:
    """Stable training identity.

    Execution controls such as wall-clock budget and ``max_updates_this_run`` are
    deliberately excluded so a later invocation can continue the same experiment.
    """

    return {
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "generator_architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
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
        "balance_weight": float(balance_weight),
        "adversarial_weight": float(adversarial_weight),
        "feature_matching_weight": float(feature_matching_weight),
        "min_delta": float(min_delta),
        "checkpoint_every_updates": int(checkpoint_every_updates),
        "validation_segment_set_sha256": val_segment_set_sha256,
    }


def _latest_validation(
    history: list[dict[str, object]],
    *,
    initial_reconstruction: float,
    initial_balance: float,
    initial_score: float,
) -> tuple[float, float, float]:
    if not history:
        return initial_reconstruction, initial_balance, initial_score
    row = history[-1]
    return (
        float(row["validation_reconstruction"]),
        float(row["validation_spectral_balance"]),
        float(row["validation_selection_score"]),
    )


def _checkpoint_metadata(
    *,
    run_config: dict[str, object],
    history: list[dict[str, object]],
    initial_reconstruction: float,
    initial_balance: float,
    initial_score: float,
    best_reconstruction: float,
    best_balance: float,
    best_score: float,
    best_epoch: int,
    epochs_without_improvement: int,
    partial_epoch_state: dict[str, object] | None,
    resumed_invocations: int,
) -> dict[str, object]:
    return {
        "purpose": "persistent_source_filter_v4_1_training",
        "run_config": run_config,
        "history": history,
        "initial_validation_reconstruction": initial_reconstruction,
        "initial_validation_spectral_balance": initial_balance,
        "initial_validation_selection_score": initial_score,
        "best_validation_reconstruction": best_reconstruction,
        "best_validation_spectral_balance": best_balance,
        "best_validation_selection_score": best_score,
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "partial_epoch_state": partial_epoch_state,
        "resumed_invocations": int(resumed_invocations),
        "checkpoint_semantics": (
            "epoch is the 1-based epoch to process next/currently; "
            "next_item_offset is the next shuffled item within that epoch"
        ),
    }


def _restore_or_initialize(
    last_path: Path,
    *,
    provenance: dict[str, object],
    run_config: dict[str, object],
    val_items_conditioned: list[ConditionedSegment],
    balance_weight: float,
    seed: int,
) -> tuple[
    LykenoxVocoderGeneratorV41,
    LykenoxMultiScaleWaveformDiscriminator,
    dict[str, object] | None,
    int,
    int,
    int,
    list[dict[str, object]],
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    int,
    EpochAccumulator,
]:
    if last_path.exists():
        generator, discriminator, payload = load_source_filter_checkpoint(last_path)
        if payload.get("training_provenance") != provenance:
            raise RuntimeError(
                "Existing v4.1 last.pt provenance differs from the active dataset. "
                "Do not resume after changing manifests or feature contracts."
            )
        metadata = payload.get("training_metadata")
        if not isinstance(metadata, dict) or metadata.get("run_config") != run_config:
            raise RuntimeError(
                "Existing v4.1 last.pt training configuration differs from this command."
            )
        epoch = int(payload.get("epoch", 1))
        next_item_offset = int(payload.get("next_item_offset", 0))
        global_step = int(payload.get("global_step", 0))
        history = list(metadata.get("history", []))
        initial_reconstruction = float(metadata["initial_validation_reconstruction"])
        initial_balance = float(metadata["initial_validation_spectral_balance"])
        initial_score = float(metadata["initial_validation_selection_score"])
        best_reconstruction = float(metadata["best_validation_reconstruction"])
        best_balance = float(metadata["best_validation_spectral_balance"])
        best_score = float(metadata["best_validation_selection_score"])
        best_epoch = int(metadata["best_epoch"])
        epochs_without_improvement = int(metadata.get("epochs_without_improvement", 0))
        accumulator = EpochAccumulator.from_payload(
            metadata.get("partial_epoch_state"),
            epoch=epoch,
        )
        rng_state = payload.get("torch_rng_state")
        if isinstance(rng_state, torch.Tensor):
            torch.set_rng_state(rng_state)
        generator.train()
        discriminator.train()
        return (
            generator,
            discriminator,
            payload,
            epoch,
            next_item_offset,
            global_step,
            history,
            initial_reconstruction,
            initial_balance,
            initial_score,
            best_reconstruction,
            best_balance,
            best_score,
            best_epoch,
            epochs_without_improvement,
            accumulator,
        )

    torch.manual_seed(seed)
    generator = LykenoxVocoderGeneratorV41().cpu().train()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
    initial_reconstruction, initial_balance, initial_score = _validation_metrics(
        generator,
        val_items_conditioned,
        balance_weight=balance_weight,
    )
    return (
        generator,
        discriminator,
        None,
        1,
        0,
        0,
        [],
        initial_reconstruction,
        initial_balance,
        initial_score,
        initial_reconstruction,
        initial_balance,
        initial_score,
        0,
        0,
        EpochAccumulator(),
    )


def _atomic_checkpoint(
    path: Path,
    generator: LykenoxVocoderGeneratorV41,
    discriminator: LykenoxMultiScaleWaveformDiscriminator,
    **kwargs: Any,
) -> None:
    """Write a checkpoint through a sibling temporary file then replace atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_source_filter_checkpoint(
        temporary,
        generator,
        discriminator,
        **kwargs,
    )
    os.replace(temporary, path)


def _save_last(
    *,
    last_path: Path,
    generator: LykenoxVocoderGeneratorV41,
    discriminator: LykenoxMultiScaleWaveformDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    next_item_offset: int,
    provenance: dict[str, object],
    metadata: dict[str, object],
    history: list[dict[str, object]],
    initial_reconstruction: float,
    initial_balance: float,
    initial_score: float,
) -> None:
    latest_reconstruction, latest_balance, latest_score = _latest_validation(
        history,
        initial_reconstruction=initial_reconstruction,
        initial_balance=initial_balance,
        initial_score=initial_score,
    )
    _atomic_checkpoint(
        last_path,
        generator,
        discriminator,
        epoch=epoch,
        global_step=global_step,
        next_item_offset=next_item_offset,
        validation_reconstruction_loss=latest_reconstruction,
        validation_spectral_balance_loss=latest_balance,
        validation_selection_score=latest_score,
        training_provenance=provenance,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        training_metadata=metadata,
    )


def _known_artifact_evaluation(
    generator: LykenoxVocoderGeneratorV41,
    items: list[ConditionedSegment],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    rows: list[dict[str, object]] = []
    confirmed_locks = 0
    collapse_count = 0
    upper_voice_missing_count = 0

    with torch.no_grad():
        for index, item in enumerate(items[:3], start=1):
            generated_tensor = _generate(generator, item).squeeze(0).detach().cpu()
            reference_tensor = item.segment.waveform.detach().cpu()
            generated = generated_tensor.numpy().astype(np.float64, copy=False)
            reference = reference_tensor.numpy().astype(np.float64, copy=False)

            confirmed, forensic = _generated_specific_frame_lock(
                generated,
                reference,
                generator.config.sample_rate,
            )
            generated_spectral = _spectral_metrics(
                generated,
                generator.config.sample_rate,
            )
            reference_spectral = _spectral_metrics(
                reference,
                generator.config.sample_rate,
            )
            generated_pitch = _pitch_metrics(generated, generator.config.sample_rate)
            reference_pitch = _pitch_metrics(reference, generator.config.sample_rate)
            generated_bands = spectral_band_fractions(
                generated_tensor.unsqueeze(0),
                sample_rate=generator.config.sample_rate,
            ).squeeze(0)
            reference_bands = spectral_band_fractions(
                reference_tensor.unsqueeze(0),
                sample_rate=generator.config.sample_rate,
            ).squeeze(0)
            generated_above_300 = float(generated_bands[2:].sum())
            reference_above_300 = float(reference_bands[2:].sum())
            upper_voice_missing = generated_above_300 < max(
                0.03,
                0.15 * reference_above_300,
            )

            confirmed_locks += int(confirmed)
            collapse_count += int(
                bool(generated_spectral["subbass_or_silence_collapsed"])
            )
            upper_voice_missing_count += int(upper_voice_missing)

            generated_path = (
                output_dir
                / f"val_{index:02d}_{item.segment.utterance_id}_generated.wav"
            )
            reference_path = (
                output_dir
                / f"val_{index:02d}_{item.segment.utterance_id}_reference.wav"
            )
            sf.write(
                str(generated_path),
                generated_tensor.numpy(),
                generator.config.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                str(reference_path),
                reference_tensor.numpy(),
                generator.config.sample_rate,
                subtype="PCM_16",
            )

            rows.append(
                {
                    "utterance_id": item.segment.utterance_id,
                    "generated": str(generated_path),
                    "reference": str(reference_path),
                    "generated_pitch": generated_pitch,
                    "reference_pitch": reference_pitch,
                    "generated_spectral": generated_spectral,
                    "reference_spectral": reference_spectral,
                    "generated_above_300hz_fraction": round(
                        generated_above_300,
                        6,
                    ),
                    "reference_above_300hz_fraction": round(
                        reference_above_300,
                        6,
                    ),
                    "upper_voice_band_missing": bool(upper_voice_missing),
                    **forensic,
                }
            )
    generator.train()
    return rows, confirmed_locks, collapse_count, upper_voice_missing_count


def run_bounded_resumable_source_filter_training(
    root: Path,
    *,
    segment_mel_frames: int = 64,
    train_items: int = 118,
    val_items: int = 14,
    max_epochs: int = 24,
    warmup_epochs: int = 8,
    patience: int = 6,
    seed: int = 1337,
    generator_lr: float = 2e-4,
    discriminator_lr: float = 2e-4,
    balance_weight: float = 0.50,
    adversarial_weight: float = 0.05,
    feature_matching_weight: float = 1.0,
    min_delta: float = 1e-4,
    checkpoint_every_updates: int = 16,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    checkpoint_reserve_seconds: float = DEFAULT_CHECKPOINT_RESERVE_SECONDS,
    max_updates_this_run: int | None = None,
    artifact_dir_override: Path | None = None,
) -> dict[str, object]:
    if segment_mel_frames < 32:
        raise ValueError("segment_mel_frames must be >= 32")
    if train_items < 2 or val_items < 2:
        raise ValueError("train_items and val_items must both be >= 2")
    if max_epochs < 1 or warmup_epochs < 0 or warmup_epochs > max_epochs:
        raise ValueError("Invalid epoch configuration")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if generator_lr <= 0.0 or discriminator_lr <= 0.0:
        raise ValueError("learning rates must be positive")
    if balance_weight <= 0.0:
        raise ValueError("balance_weight must be positive")
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
    val_items_conditioned = _condition_segments(val_segments)
    val_set_sha = _segment_set_sha256(val_segments)

    provenance = build_source_filter_training_provenance(
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
        balance_weight=balance_weight,
        adversarial_weight=adversarial_weight,
        feature_matching_weight=feature_matching_weight,
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
        / "vocoder_source_filter_v4_1"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    last_path = artifact_dir / "last.pt"
    best_path = artifact_dir / "best.pt"
    progress_path = artifact_dir / "training_progress.json"
    report_path = artifact_dir / "training_report.json"

    (
        generator,
        discriminator,
        resumed_payload,
        epoch,
        next_item_offset,
        global_step,
        history,
        initial_reconstruction,
        initial_balance,
        initial_score,
        best_reconstruction,
        best_balance,
        best_score,
        best_epoch,
        epochs_without_improvement,
        accumulator,
    ) = _restore_or_initialize(
        last_path,
        provenance=provenance,
        run_config=run_config,
        val_items_conditioned=val_items_conditioned,
        balance_weight=balance_weight,
        seed=seed,
    )

    resumed_invocations = 0
    if resumed_payload is not None:
        metadata = resumed_payload.get("training_metadata")
        if isinstance(metadata, dict):
            resumed_invocations = int(metadata.get("resumed_invocations", 0)) + 1

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
        generator_optimizer_state = resumed_payload.get("generator_optimizer_state")
        discriminator_optimizer_state = resumed_payload.get(
            "discriminator_optimizer_state"
        )
        if not isinstance(generator_optimizer_state, dict) or not isinstance(
            discriminator_optimizer_state,
            dict,
        ):
            raise RuntimeError("Cannot resume v4.1 training without both optimizer states")
        generator_optimizer.load_state_dict(generator_optimizer_state)
        discriminator_optimizer.load_state_dict(discriminator_optimizer_state)

    invocation_update_times: list[float] = []
    updates_this_run = 0
    stop_reason = "max_epochs_reached"
    completed = False
    early_stopped = False

    def current_metadata(
        partial: dict[str, object] | None,
    ) -> dict[str, object]:
        return _checkpoint_metadata(
            run_config=run_config,
            history=history,
            initial_reconstruction=initial_reconstruction,
            initial_balance=initial_balance,
            initial_score=initial_score,
            best_reconstruction=best_reconstruction,
            best_balance=best_balance,
            best_score=best_score,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            partial_epoch_state=partial,
            resumed_invocations=resumed_invocations,
        )

    def save_interruption(reason: str) -> dict[str, object]:
        partial = (
            accumulator.to_payload(epoch=epoch)
            if next_item_offset > 0
            else None
        )
        metadata = current_metadata(partial)
        _save_last(
            last_path=last_path,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            epoch=epoch,
            global_step=global_step,
            next_item_offset=next_item_offset,
            provenance=provenance,
            metadata=metadata,
            history=history,
            initial_reconstruction=initial_reconstruction,
            initial_balance=initial_balance,
            initial_score=initial_score,
        )
        progress = {
            "status": "incomplete",
            "device": "cpu",
            "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
            "stop_reason": reason,
            "epochs_completed": len(history),
            "current_epoch": epoch,
            "next_item_offset": next_item_offset,
            "global_step": global_step,
            "updates_this_run": updates_this_run,
            "best_epoch": best_epoch,
            "initial_validation_reconstruction": round(initial_reconstruction, 6),
            "best_validation_reconstruction": round(best_reconstruction, 6),
            "initial_validation_spectral_balance": round(initial_balance, 6),
            "best_validation_spectral_balance": round(best_balance, 6),
            "initial_validation_selection_score": round(initial_score, 6),
            "best_validation_selection_score": round(best_score, 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
            "next_gate": "rerun_same_command_to_resume",
        }
        _atomic_json(progress_path, progress)
        return {**progress, "progress_report": str(progress_path)}

    while epoch <= max_epochs:
        # Each epoch receives a new deterministic crop location per utterance.  The epoch
        # number is part of the seed, so a mid-epoch resume reconstructs the exact same
        # list and order.
        epoch_seed = seed + epoch
        train_segments, train_skipped = collect_vocoder_segments(
            root,
            "train",
            segment_mel_frames=segment_mel_frames,
            max_items=train_items,
            seed=epoch_seed,
        )
        train_conditioned = _condition_segments(train_segments)
        order = list(range(len(train_conditioned)))
        random.Random(seed * 1009 + epoch).shuffle(order)

        if next_item_offset > len(order):
            raise RuntimeError(
                "Checkpoint next_item_offset exceeds the regenerated epoch item count"
            )
        if next_item_offset == 0:
            accumulator = EpochAccumulator()

        for position in range(next_item_offset, len(order)):
            elapsed = time.perf_counter() - started
            if elapsed >= time_budget_seconds - checkpoint_reserve_seconds:
                stop_reason = "time_budget_reached"
                return save_interruption(stop_reason)
            if (
                max_updates_this_run is not None
                and updates_this_run >= max_updates_this_run
            ):
                stop_reason = "max_updates_this_run_reached"
                return save_interruption(stop_reason)

            item = train_conditioned[order[position]]
            mel = item.segment.mel.unsqueeze(0)
            target = item.segment.waveform.unsqueeze(0)
            f0 = item.pitch.f0_hz.unsqueeze(0)
            voiced = item.pitch.voiced.unsqueeze(0)
            update_started = time.perf_counter()

            if epoch > warmup_epochs:
                _set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    detached = generator(mel, f0, voiced)
                real_output = discriminator(target)
                fake_output = discriminator(detached)
                discriminator_loss = discriminator_hinge_loss(
                    real_output,
                    fake_output,
                )
                if not torch.isfinite(discriminator_loss):
                    raise RuntimeError(
                        f"Non-finite discriminator loss at epoch {epoch} item {position}"
                    )
                discriminator_loss.backward()
                discriminator_grad = torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    10.0,
                )
                if not math.isfinite(float(discriminator_grad)):
                    raise RuntimeError("Non-finite discriminator gradient")
                discriminator_optimizer.step()
            else:
                discriminator_loss = None

            _set_requires_grad(discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)
            prediction = generator(mel, f0, voiced)
            reconstruction = multi_resolution_reconstruction_loss(
                prediction,
                target,
            )
            balance = target_relative_spectral_balance_loss(
                prediction,
                target,
                sample_rate=generator.config.sample_rate,
            )
            generator_loss = (
                reconstruction.total
                + balance_weight * balance.loss
            )
            adversarial = None
            feature_match = None
            if epoch > warmup_epochs:
                with torch.no_grad():
                    real_features = discriminator(target)
                fake_features = discriminator(prediction)
                adversarial = generator_adversarial_loss(fake_features)
                feature_match = feature_matching_loss(
                    real_features,
                    fake_features,
                )
                generator_loss = (
                    generator_loss
                    + adversarial_weight * adversarial
                    + feature_matching_weight * feature_match
                )

            if not torch.isfinite(generator_loss):
                raise RuntimeError(
                    f"Non-finite generator loss at epoch {epoch} item {position}"
                )
            generator_loss.backward()
            generator_grad = torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                10.0,
            )
            if not math.isfinite(float(generator_grad)):
                raise RuntimeError("Non-finite generator gradient")
            generator_optimizer.step()
            _set_requires_grad(discriminator, True)

            reconstruction_value = float(reconstruction.total.detach().cpu())
            balance_value = float(balance.loss.detach().cpu())
            accumulator.reconstruction_sum += reconstruction_value
            accumulator.balance_sum += balance_value
            accumulator.update_count += 1
            if discriminator_loss is not None:
                assert adversarial is not None and feature_match is not None
                accumulator.discriminator_sum += float(
                    discriminator_loss.detach().cpu()
                )
                accumulator.adversarial_sum += float(adversarial.detach().cpu())
                accumulator.feature_matching_sum += float(
                    feature_match.detach().cpu()
                )
                accumulator.adversarial_count += 1

            invocation_update_times.append(time.perf_counter() - update_started)
            global_step += 1
            updates_this_run += 1
            next_item_offset = position + 1

            if global_step % checkpoint_every_updates == 0:
                metadata = current_metadata(
                    accumulator.to_payload(epoch=epoch)
                )
                _save_last(
                    last_path=last_path,
                    generator=generator,
                    discriminator=discriminator,
                    generator_optimizer=generator_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    next_item_offset=next_item_offset,
                    provenance=provenance,
                    metadata=metadata,
                    history=history,
                    initial_reconstruction=initial_reconstruction,
                    initial_balance=initial_balance,
                    initial_score=initial_score,
                )

        # Full epoch only: validation is comparable and may update best/early stopping.
        if accumulator.update_count != len(order):
            raise RuntimeError(
                "Epoch accumulator count does not match completed deterministic order"
            )

        validation_reconstruction, validation_balance, validation_score = (
            _validation_metrics(
                generator,
                val_items_conditioned,
                balance_weight=balance_weight,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "phase": (
                    "reconstruction_source_balance_warmup"
                    if epoch <= warmup_epochs
                    else "mild_adversarial"
                ),
                "train_items": len(order),
                "epoch_segment_seed": epoch_seed,
                "train_reconstruction": (
                    accumulator.reconstruction_sum / accumulator.update_count
                ),
                "train_spectral_balance": (
                    accumulator.balance_sum / accumulator.update_count
                ),
                "validation_reconstruction": validation_reconstruction,
                "validation_spectral_balance": validation_balance,
                "validation_selection_score": validation_score,
                "discriminator_loss": (
                    accumulator.discriminator_sum
                    / accumulator.adversarial_count
                    if accumulator.adversarial_count
                    else None
                ),
                "generator_adversarial_loss": (
                    accumulator.adversarial_sum
                    / accumulator.adversarial_count
                    if accumulator.adversarial_count
                    else None
                ),
                "feature_matching_loss": (
                    accumulator.feature_matching_sum
                    / accumulator.adversarial_count
                    if accumulator.adversarial_count
                    else None
                ),
                "global_step": global_step,
                "train_skipped_count": len(train_skipped),
            }
        )

        improved = validation_score < best_score - min_delta
        if improved:
            best_reconstruction = validation_reconstruction
            best_balance = validation_balance
            best_score = validation_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_metadata = current_metadata(None)
            _atomic_checkpoint(
                best_path,
                generator,
                discriminator,
                epoch=epoch + 1,
                global_step=global_step,
                next_item_offset=0,
                validation_reconstruction_loss=validation_reconstruction,
                validation_spectral_balance_loss=validation_balance,
                validation_selection_score=validation_score,
                training_provenance=provenance,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                training_metadata=best_metadata,
            )
        else:
            epochs_without_improvement += 1

        epoch += 1
        next_item_offset = 0
        accumulator = EpochAccumulator()
        metadata = current_metadata(None)
        _save_last(
            last_path=last_path,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            epoch=epoch,
            global_step=global_step,
            next_item_offset=0,
            provenance=provenance,
            metadata=metadata,
            history=history,
            initial_reconstruction=initial_reconstruction,
            initial_balance=initial_balance,
            initial_score=initial_score,
        )

        progress = {
            "status": "running",
            "device": "cpu",
            "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
            "epochs_completed": len(history),
            "next_epoch": epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_validation_reconstruction": round(best_reconstruction, 6),
            "best_validation_spectral_balance": round(best_balance, 6),
            "best_validation_selection_score": round(best_score, 6),
            "epochs_without_improvement": epochs_without_improvement,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
        }
        _atomic_json(progress_path, progress)

        if epochs_without_improvement >= patience:
            early_stopped = True
            stop_reason = "early_stopping"
            break

    if epoch > max_epochs:
        completed = True
        stop_reason = "max_epochs_reached"
    if early_stopped:
        completed = True

    if not completed:
        return save_interruption(stop_reason)

    if not best_path.exists():
        report = {
            "status": "needs_review",
            "device": "cpu",
            "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
            "stop_reason": stop_reason,
            "epochs_completed": len(history),
            "global_step": global_step,
            "initial_validation_selection_score": round(initial_score, 6),
            "best_validation_selection_score": round(best_score, 6),
            "reason": "no_trained_epoch_improved_over_initial_model",
            "last_checkpoint": str(last_path),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "next_gate": "review_persistent_v4_1_optimization",
        }
        _atomic_json(report_path, report)
        return {**report, "report_path": str(report_path)}

    best_generator, _, _ = load_source_filter_checkpoint(best_path)
    best_generator.train()
    listening_pairs, frame_locks, collapse_count, upper_missing_count = (
        _known_artifact_evaluation(
            best_generator,
            val_items_conditioned,
            artifact_dir / "listening",
        )
    )
    automatic_gate = (
        frame_locks == 0
        and collapse_count == 0
        and upper_missing_count == 0
        and best_balance < initial_balance
    )

    report = {
        "status": "pass" if automatic_gate else "needs_review",
        "device": "cpu",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
        "parameters": best_generator.parameter_count(),
        "stop_reason": stop_reason,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "global_step": global_step,
        "initial_validation_reconstruction": round(initial_reconstruction, 6),
        "best_validation_reconstruction": round(best_reconstruction, 6),
        "initial_validation_spectral_balance": round(initial_balance, 6),
        "best_validation_spectral_balance": round(best_balance, 6),
        "initial_validation_selection_score": round(initial_score, 6),
        "best_validation_selection_score": round(best_score, 6),
        "validation_selection_improved": best_score < initial_score,
        "validation_spectral_balance_improved": best_balance < initial_balance,
        "confirmed_generated_specific_frame_locks": frame_locks,
        "subbass_or_silence_collapse_count": collapse_count,
        "upper_voice_band_missing_count": upper_missing_count,
        "automatic_artifact_gate_pass": automatic_gate,
        "resumed_invocations": resumed_invocations,
        "mean_seconds_per_update_this_run": (
            round(statistics.fmean(invocation_update_times), 4)
            if invocation_update_times
            else None
        ),
        "elapsed_seconds_this_run": round(time.perf_counter() - started, 3),
        "validation_items": len(val_segments),
        "validation_skipped_count": len(val_skipped),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "listening_pairs": listening_pairs,
        "history": history,
        "next_gate": (
            "listen_persistent_v4_1_validation_wavs"
            if automatic_gate
            else "review_persistent_v4_1_artifacts_before_more_training"
        ),
        "warning": (
            "A persistent-training pass clears known numerical/artifact gates only. "
            "Human listening remains mandatory before runtime export."
        ),
    }
    _atomic_json(report_path, report)
    _atomic_json(progress_path, report)
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=64)
    parser.add_argument("--train-items", type=int, default=118)
    parser.add_argument("--val-items", type=int, default=14)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--generator-lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--balance-weight", type=float, default=0.50)
    parser.add_argument("--adversarial-weight", type=float, default=0.05)
    parser.add_argument("--feature-matching-weight", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every-updates", type=int, default=16)
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_bounded_resumable_source_filter_training(
                args.root,
                segment_mel_frames=args.segment_mel_frames,
                train_items=args.train_items,
                val_items=args.val_items,
                max_epochs=args.max_epochs,
                warmup_epochs=args.warmup_epochs,
                patience=args.patience,
                seed=args.seed,
                generator_lr=args.generator_lr,
                discriminator_lr=args.discriminator_lr,
                balance_weight=args.balance_weight,
                adversarial_weight=args.adversarial_weight,
                feature_matching_weight=args.feature_matching_weight,
                min_delta=args.min_delta,
                checkpoint_every_updates=args.checkpoint_every_updates,
                time_budget_seconds=args.time_budget_seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
