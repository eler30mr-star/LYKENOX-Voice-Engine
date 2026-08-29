"""Bounded, exactly resumable persistent trainer for LYKENOX vocoder v5.

V5 deliberately removes the v4.x harmonic-exposure objective from checkpoint selection.
The stable target-referenced selection score is reconstruction + envelope + spectral balance
+ local spectral contrast. Adversarial and feature-matching terms are training-only after
warmup. Each invocation checkpoints and exits safely below the local two-minute ceiling.
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
    LykenoxVocoderGeneratorV5,
    VOCODER_GENERATOR_V5_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import PitchFrames, extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import VocoderSegment, collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import (
    LogMelEnvelopeLoss,
    VOCODER_ENVELOPE_LOSS_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
    VOCODER_LOCAL_SPECTRAL_CONTRAST_VERSION,
    target_relative_local_spectral_contrast_loss,
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
from lykenox_voice_engine.training.speech_vocoder_v5_artifact import (
    build_v5_training_provenance,
    load_v5_checkpoint,
    save_v5_checkpoint,
)


TRAINER_CONTRACT_VERSION = "v5-bounded-resumable-v1"
TRAIN_SEGMENT_SCHEDULE_VERSION = "epoch-crop-shuffle-v1"
DEFAULT_TIME_BUDGET_SECONDS = 80.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 10.0
DEFAULT_VALIDATION_RESERVE_SECONDS = 20.0
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
    contrast_sum: float = 0.0
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
        fields = {name: payload.get(name, 0) for name in cls.__dataclass_fields__}
        return cls(**fields)

    def to_payload(self, *, epoch: int) -> dict[str, object]:
        return {"epoch": int(epoch), **{name: getattr(self, name) for name in self.__dataclass_fields__}}


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _condition_segments(segments: list[VocoderSegment]) -> list[ConditionedSegment]:
    return [
        ConditionedSegment(
            segment=segment,
            pitch=extract_pitch_frames(segment.waveform, frame_count=segment.mel_frames),
        )
        for segment in segments
    ]


def _segment_set_sha256(segments: list[VocoderSegment]) -> str:
    rows = [
        {
            "split": s.split,
            "utterance_id": s.utterance_id,
            "wav_path": s.wav_path,
            "start_frame": s.start_frame,
            "mel_frames": s.mel_frames,
        }
        for s in segments
    ]
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _epoch_segments(root: Path, *, epoch: int, seed: int, segment_mel_frames: int, train_items: int) -> tuple[list[ConditionedSegment], int]:
    segments, skipped = collect_vocoder_segments(
        root,
        "train",
        segment_mel_frames=segment_mel_frames,
        max_items=train_items,
        seed=seed + epoch,
    )
    order = list(range(len(segments)))
    random.Random(seed + 1_000_003 + epoch).shuffle(order)
    return _condition_segments([segments[index] for index in order]), len(skipped)


def _generate(generator: LykenoxVocoderGeneratorV5, item: ConditionedSegment) -> torch.Tensor:
    return generator(
        item.segment.mel.unsqueeze(0),
        item.pitch.f0_hz.unsqueeze(0),
        item.pitch.voiced.unsqueeze(0),
    )


def _base_losses(
    generator: LykenoxVocoderGeneratorV5,
    envelope_loss: LogMelEnvelopeLoss,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    envelope_weight: float,
    balance_weight: float,
    contrast_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction = multi_resolution_reconstruction_loss(prediction, target).total
    envelope = envelope_loss(prediction, target).total
    balance = target_relative_spectral_balance_loss(
        prediction,
        target,
        sample_rate=generator.config.sample_rate,
    ).loss
    contrast = target_relative_local_spectral_contrast_loss(
        prediction,
        target,
        hop_length=generator.config.hop_length,
    ).loss
    total = reconstruction + envelope_weight * envelope + balance_weight * balance + contrast_weight * contrast
    return total, {
        "reconstruction": reconstruction,
        "envelope": envelope,
        "balance": balance,
        "contrast": contrast,
    }


def _validation_metrics(
    generator: LykenoxVocoderGeneratorV5,
    envelope_loss: LogMelEnvelopeLoss,
    items: list[ConditionedSegment],
    *,
    envelope_weight: float,
    balance_weight: float,
    contrast_weight: float,
) -> tuple[float, float, float, float, float]:
    generator.eval()
    buckets: list[list[float]] = [[], [], [], []]
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
                contrast_weight=contrast_weight,
            )
            row = [float(losses[key].detach()) for key in ("reconstruction", "envelope", "balance", "contrast")]
            if not all(math.isfinite(value) for value in row):
                raise RuntimeError("Non-finite held-out v5 validation metric")
            for bucket, value in zip(buckets, row, strict=True):
                bucket.append(value)
    generator.train()
    means = [statistics.fmean(bucket) for bucket in buckets]
    score = means[0] + envelope_weight * means[1] + balance_weight * means[2] + contrast_weight * means[3]
    return means[0], means[1], means[2], means[3], score


def _run_config(**kwargs: object) -> dict[str, object]:
    return {
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "generator_architecture": VOCODER_GENERATOR_V5_ARCHITECTURE,
        "source_family": LykenoxVocoderGeneratorV5.source_family,
        "explicit_sinusoidal_carrier": False,
        "deterministic_harmonics": 0,
        "envelope_loss_version": VOCODER_ENVELOPE_LOSS_VERSION,
        "local_spectral_contrast_version": VOCODER_LOCAL_SPECTRAL_CONTRAST_VERSION,
        "train_segment_schedule_version": TRAIN_SEGMENT_SCHEDULE_VERSION,
        **kwargs,
    }


def _checkpoint_metadata(
    *,
    run_config: dict[str, object],
    history: list[dict[str, object]],
    initial_metrics: tuple[float, float, float, float, float],
    best_metrics: tuple[float, float, float, float, float],
    best_epoch: int,
    epochs_without_improvement: int,
    partial_epoch_state: dict[str, object] | None,
    resumed_invocations: int,
) -> dict[str, object]:
    names = ("reconstruction", "envelope", "spectral_balance", "local_spectral_contrast", "selection_score")
    metadata: dict[str, object] = {
        "purpose": "persistent_v5_stochastic_glottal_filter_training",
        "run_config": run_config,
        "history": history,
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "partial_epoch_state": partial_epoch_state,
        "resumed_invocations": int(resumed_invocations),
    }
    for name, value in zip(names, initial_metrics, strict=True):
        metadata[f"initial_validation_{name}"] = value
    for name, value in zip(names, best_metrics, strict=True):
        metadata[f"best_validation_{name}"] = value
    return metadata


def _metrics_from_metadata(metadata: dict[str, object], prefix: str) -> tuple[float, float, float, float, float]:
    names = ("reconstruction", "envelope", "spectral_balance", "local_spectral_contrast", "selection_score")
    return tuple(float(metadata[f"{prefix}_validation_{name}"]) for name in names)  # type: ignore[return-value]


def _save_state(
    path: Path,
    *,
    generator: LykenoxVocoderGeneratorV5,
    discriminator: LykenoxMultiScaleWaveformDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    next_item_offset: int,
    provenance: dict[str, object],
    metadata: dict[str, object],
    validation_metrics: tuple[float, float, float, float, float],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    reconstruction, envelope, balance, contrast, score = validation_metrics
    save_v5_checkpoint(
        temporary,
        generator,
        discriminator,
        epoch=epoch,
        global_step=global_step,
        next_item_offset=next_item_offset,
        validation_reconstruction_loss=reconstruction,
        validation_envelope_loss=envelope,
        validation_spectral_balance_loss=balance,
        validation_local_spectral_contrast_loss=contrast,
        validation_selection_score=score,
        training_provenance=provenance,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        training_metadata=metadata,
    )
    os.replace(temporary, path)


def _latest_validation(history: list[dict[str, object]], initial: tuple[float, float, float, float, float]) -> tuple[float, float, float, float, float]:
    if not history:
        return initial
    row = history[-1]
    return (
        float(row["validation_reconstruction"]),
        float(row["validation_envelope"]),
        float(row["validation_spectral_balance"]),
        float(row["validation_local_spectral_contrast"]),
        float(row["validation_selection_score"]),
    )


def run_bounded_resumable_v5_training(
    root: Path,
    *,
    segment_mel_frames: int = 48,
    train_items: int = 118,
    val_items: int = 14,
    max_epochs: int = 28,
    warmup_epochs: int = 4,
    patience: int = 6,
    seed: int = 2500,
    generator_lr: float = 2e-4,
    discriminator_lr: float = 1e-4,
    envelope_weight: float = 0.50,
    balance_weight: float = 0.25,
    contrast_weight: float = 0.15,
    adversarial_weight: float = 0.03,
    feature_matching_weight: float = 0.50,
    gradient_clip_norm: float = 5.0,
    min_delta: float = 1e-4,
    checkpoint_every_updates: int = 6,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    checkpoint_reserve_seconds: float = DEFAULT_CHECKPOINT_RESERVE_SECONDS,
    validation_reserve_seconds: float = DEFAULT_VALIDATION_RESERVE_SECONDS,
    max_updates_this_run: int | None = None,
    artifact_dir_override: Path | None = None,
) -> dict[str, object]:
    if segment_mel_frames < 32 or train_items < 2 or val_items < 2:
        raise ValueError("invalid v5 data bounds")
    if max_epochs < 1 or not 0 <= warmup_epochs <= max_epochs or patience < 1:
        raise ValueError("invalid v5 epoch configuration")
    if min(generator_lr, discriminator_lr, envelope_weight, balance_weight, contrast_weight, gradient_clip_norm) <= 0.0:
        raise ValueError("positive v5 hyperparameter required")
    if adversarial_weight < 0.0 or feature_matching_weight < 0.0 or checkpoint_every_updates < 1:
        raise ValueError("invalid v5 training weights/checkpoint cadence")
    if max_updates_this_run is not None and max_updates_this_run < 1:
        raise ValueError("max_updates_this_run must be positive")

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
    provenance = build_v5_training_provenance(root, segment_mel_frames=segment_mel_frames, seed=seed)
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
        contrast_weight=contrast_weight,
        adversarial_weight=adversarial_weight,
        feature_matching_weight=feature_matching_weight,
        gradient_clip_norm=gradient_clip_norm,
        min_delta=min_delta,
        checkpoint_every_updates=checkpoint_every_updates,
        validation_segment_set_sha256=_segment_set_sha256(val_segments),
    )

    artifact_dir = Path(artifact_dir_override).resolve() if artifact_dir_override is not None else root / "models" / "lykenox_identity" / "training" / "vocoder_stochastic_glottal_filter_v5"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    last_path = artifact_dir / "last.pt"
    best_path = artifact_dir / "best.pt"
    progress_path = artifact_dir / "training_progress.json"
    report_path = artifact_dir / "training_report.json"
    envelope_loss = LogMelEnvelopeLoss().cpu()

    resumed_payload: dict[str, object] | None = None
    if last_path.exists():
        generator, discriminator, resumed_payload = load_v5_checkpoint(last_path)
        if resumed_payload.get("training_provenance") != provenance:
            raise RuntimeError("Existing v5 last.pt provenance differs from active dataset/features")
        metadata = resumed_payload.get("training_metadata")
        if not isinstance(metadata, dict) or metadata.get("run_config") != run_config:
            raise RuntimeError("Existing v5 last.pt configuration differs from this command")
        epoch = int(resumed_payload.get("epoch", 1))
        next_item_offset = int(resumed_payload.get("next_item_offset", 0))
        global_step = int(resumed_payload.get("global_step", 0))
        history = list(metadata.get("history", []))
        initial_metrics = _metrics_from_metadata(metadata, "initial")
        best_metrics = _metrics_from_metadata(metadata, "best")
        best_epoch = int(metadata.get("best_epoch", 0))
        epochs_without_improvement = int(metadata.get("epochs_without_improvement", 0))
        accumulator = EpochAccumulator.from_payload(metadata.get("partial_epoch_state"), epoch=epoch)
        resumed_invocations = int(metadata.get("resumed_invocations", 0)) + 1
        rng_state = resumed_payload.get("torch_rng_state")
        if isinstance(rng_state, torch.Tensor):
            torch.set_rng_state(rng_state)
        generator.train(); discriminator.train()
    else:
        torch.manual_seed(seed)
        generator = LykenoxVocoderGeneratorV5().cpu().train()
        discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
        initial_metrics = _validation_metrics(
            generator,
            envelope_loss,
            val_conditioned,
            envelope_weight=envelope_weight,
            balance_weight=balance_weight,
            contrast_weight=contrast_weight,
        )
        best_metrics = initial_metrics
        epoch = 1; next_item_offset = 0; global_step = 0
        history: list[dict[str, object]] = []
        best_epoch = 0; epochs_without_improvement = 0
        accumulator = EpochAccumulator(); resumed_invocations = 0

    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=generator_lr, weight_decay=1e-5)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=discriminator_lr, weight_decay=1e-5)
    if resumed_payload is not None:
        gs = resumed_payload.get("generator_optimizer_state")
        ds = resumed_payload.get("discriminator_optimizer_state")
        if not isinstance(gs, dict) or not isinstance(ds, dict):
            raise RuntimeError("Cannot exactly resume v5 without both optimizer states")
        generator_optimizer.load_state_dict(gs); discriminator_optimizer.load_state_dict(ds)

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
            metadata=current_metadata(partial),
            validation_metrics=_latest_validation(history, initial_metrics),
        )
        progress = {
            "status": "incomplete",
            "device": "cpu",
            "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "architecture": VOCODER_GENERATOR_V5_ARCHITECTURE,
            "source_family": generator.source_family,
            "stop_reason": reason,
            "epochs_completed": len(history),
            "current_epoch": epoch,
            "next_item_offset": next_item_offset,
            "global_step": global_step,
            "updates_this_run": updates_this_run,
            "best_epoch": best_epoch,
            "best_validation_selection_score": round(best_metrics[4], 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "mean_update_seconds_this_run": round(statistics.fmean(update_times), 4) if update_times else None,
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
            "persistent_training_complete": False,
            "historical_checkpoints_mutated": False,
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
            raise RuntimeError("v5 resume offset exceeds deterministic epoch item count")

        while next_item_offset < len(train_conditioned):
            elapsed = time.perf_counter() - started
            estimated_update = max(update_times[-4:] or [2.0])
            is_last_item = next_item_offset == len(train_conditioned) - 1
            required = checkpoint_reserve_seconds + estimated_update + (validation_reserve_seconds if is_last_item else 0.0)
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
                contrast_weight=contrast_weight,
            )
            adversarial_value = feature_matching_value = discriminator_value = 0.0
            adversarial_active = epoch > warmup_epochs and (adversarial_weight > 0.0 or feature_matching_weight > 0.0)
            if adversarial_active:
                _set_requires_grad(discriminator, False)
                with torch.no_grad():
                    real_features = discriminator(target)
                fake_features = discriminator(prediction)
                adversarial = generator_adversarial_loss(fake_features)
                matching = feature_matching_loss(real_features, fake_features)
                generator_total = generator_total + adversarial_weight * adversarial + feature_matching_weight * matching
                adversarial_value = float(adversarial.detach()); feature_matching_value = float(matching.detach())

            if not bool(torch.isfinite(generator_total)):
                raise RuntimeError("Non-finite v5 generator loss")
            generator_total.backward()
            norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=gradient_clip_norm)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite v5 generator gradient norm")
            generator_optimizer.step(); _set_requires_grad(discriminator, True)

            if adversarial_active:
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_output = discriminator(target)
                fake_output = discriminator(prediction.detach())
                discriminator_loss = discriminator_hinge_loss(real_output, fake_output)
                if not bool(torch.isfinite(discriminator_loss)):
                    raise RuntimeError("Non-finite v5 discriminator loss")
                discriminator_loss.backward()
                dnorm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=gradient_clip_norm)
                if not bool(torch.isfinite(dnorm)):
                    raise RuntimeError("Non-finite v5 discriminator gradient norm")
                discriminator_optimizer.step(); discriminator_value = float(discriminator_loss.detach())

            values = {
                "reconstruction": float(base["reconstruction"].detach()),
                "envelope": float(base["envelope"].detach()),
                "balance": float(base["balance"].detach()),
                "contrast": float(base["contrast"].detach()),
                "total": float(generator_total.detach()),
            }
            if not all(math.isfinite(v) for v in (*values.values(), adversarial_value, feature_matching_value, discriminator_value)):
                raise RuntimeError("Non-finite v5 training metric")
            accumulator.reconstruction_sum += values["reconstruction"]
            accumulator.envelope_sum += values["envelope"]
            accumulator.balance_sum += values["balance"]
            accumulator.contrast_sum += values["contrast"]
            accumulator.generator_total_sum += values["total"]
            accumulator.update_count += 1
            if adversarial_active:
                accumulator.adversarial_sum += adversarial_value
                accumulator.feature_matching_sum += feature_matching_value
                accumulator.discriminator_sum += discriminator_value
                accumulator.adversarial_count += 1

            global_step += 1; updates_this_run += 1; next_item_offset += 1
            update_times.append(time.perf_counter() - update_started)
            if global_step % checkpoint_every_updates == 0:
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
                    metadata=current_metadata(accumulator.to_payload(epoch=epoch)),
                    validation_metrics=_latest_validation(history, initial_metrics),
                )

        validation = _validation_metrics(
            generator,
            envelope_loss,
            val_conditioned,
            envelope_weight=envelope_weight,
            balance_weight=balance_weight,
            contrast_weight=contrast_weight,
        )
        count = max(1, accumulator.update_count)
        adv_count = max(1, accumulator.adversarial_count)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_reconstruction": accumulator.reconstruction_sum / count,
            "train_envelope": accumulator.envelope_sum / count,
            "train_spectral_balance": accumulator.balance_sum / count,
            "train_local_spectral_contrast": accumulator.contrast_sum / count,
            "train_generator_total": accumulator.generator_total_sum / count,
            "train_adversarial": accumulator.adversarial_sum / adv_count if accumulator.adversarial_count else 0.0,
            "train_feature_matching": accumulator.feature_matching_sum / adv_count if accumulator.adversarial_count else 0.0,
            "train_discriminator": accumulator.discriminator_sum / adv_count if accumulator.adversarial_count else 0.0,
            "adversarial_active": epoch > warmup_epochs,
            "validation_reconstruction": validation[0],
            "validation_envelope": validation[1],
            "validation_spectral_balance": validation[2],
            "validation_local_spectral_contrast": validation[3],
            "validation_selection_score": validation[4],
        }
        history.append(row)
        if validation[4] < best_metrics[4] - min_delta:
            best_metrics = validation; best_epoch = epoch; epochs_without_improvement = 0
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
                metadata=current_metadata(None),
                validation_metrics=validation,
            )
        else:
            epochs_without_improvement += 1

        epoch += 1; next_item_offset = 0; accumulator = EpochAccumulator()
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
            metadata=current_metadata(None),
            validation_metrics=validation,
        )
        if len(history) >= warmup_epochs and epochs_without_improvement >= patience:
            stop_reason = "early_stopping"; break
    else:
        stop_reason = "max_epochs"

    training_improved = best_metrics[4] < initial_metrics[4]
    envelope_improved = best_metrics[1] < initial_metrics[1]
    completed_pass = training_improved and envelope_improved and best_path.exists()
    report = {
        "status": "pass" if completed_pass else "fail",
        "device": "cpu",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "architecture": VOCODER_GENERATOR_V5_ARCHITECTURE,
        "source_family": generator.source_family,
        "explicit_sinusoidal_carrier": False,
        "deterministic_harmonics": 0,
        "envelope_loss_version": VOCODER_ENVELOPE_LOSS_VERSION,
        "local_spectral_contrast_version": VOCODER_LOCAL_SPECTRAL_CONTRAST_VERSION,
        "stop_reason": stop_reason,
        "epochs_completed": len(history),
        "global_step": global_step,
        "best_epoch": best_epoch,
        "initial_validation": {
            "reconstruction": round(initial_metrics[0], 6),
            "envelope": round(initial_metrics[1], 6),
            "spectral_balance": round(initial_metrics[2], 6),
            "local_spectral_contrast": round(initial_metrics[3], 6),
            "selection_score": round(initial_metrics[4], 6),
        },
        "best_validation": {
            "reconstruction": round(best_metrics[0], 6),
            "envelope": round(best_metrics[1], 6),
            "spectral_balance": round(best_metrics[2], 6),
            "local_spectral_contrast": round(best_metrics[3], 6),
            "selection_score": round(best_metrics[4], 6),
        },
        "training_improved": training_improved,
        "envelope_improved": envelope_improved,
        "train_items": train_items,
        "val_items": len(val_conditioned),
        "val_skipped": len(val_skipped),
        "train_skipped_latest_epoch": train_skipped_latest,
        "segment_mel_frames": segment_mel_frames,
        "resumed_invocations": resumed_invocations,
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "last_checkpoint": str(last_path),
        "persistent_training_complete": True,
        "full_utterance_perceptual_acceptance": False,
        "historical_checkpoints_mutated": False,
        "reference_audio_required_for_product_inference": False,
        "next_gate": "run_v5_full_utterance_oracle_acceptance" if completed_pass else "inspect_v5_training_failure_before_any_more_training",
    }
    _atomic_json(report_path, report); _atomic_json(progress_path, report)
    return {**report, "training_report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_bounded_resumable_v5_training(
        args.root,
        time_budget_seconds=args.time_budget_seconds,
        max_updates_this_run=args.max_updates_this_run,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
