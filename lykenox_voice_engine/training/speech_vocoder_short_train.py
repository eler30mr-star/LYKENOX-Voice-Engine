"""Bounded persistent LYKENOX vocoder training with held-out validation.

This is the first checkpointed vocoder run intended to produce audio worth listening to.
It is still deliberately short and CPU-bounded: the goal is to decide whether the current
LYKENOX-owned generator/discriminator recipe deserves a longer run, not to declare final
vocoder quality.

Properties:
- deterministic train/validation segment sets
- spectral reconstruction warm-up before adversarial pressure
- held-out validation after every epoch
- best/last resumable checkpoints
- early stopping and a wall-clock budget
- progress JSON written every epoch
- best-checkpoint validation WAV pairs for human listening

Rerunning the same command resumes a compatible interrupted run from ``last.pt``. If the
wall-clock budget is reached, the command returns cleanly with ``status: incomplete``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import soundfile as sf
import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderConfig,
    LykenoxVocoderGenerator,
)
from lykenox_voice_engine.training.speech_vocoder_artifact import (
    build_vocoder_training_provenance,
    load_vocoder_checkpoint,
    save_vocoder_checkpoint,
)
from lykenox_voice_engine.training.speech_vocoder_data import VocoderSegment, collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_losses import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)


SHORT_TRAIN_CONTRACT_VERSION = "vocoder-short-train-v1"
DEFAULT_TIME_BUDGET_SECONDS = 85.0


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _segment_set_sha256(segments: list[VocoderSegment]) -> str:
    rows = [
        {
            "split": segment.split,
            "utterance_id": segment.utterance_id,
            "start_frame": segment.start_frame,
            "mel_frames": segment.mel_frames,
            "wav_path": segment.wav_path,
        }
        for segment in segments
    ]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reconstruction_mean(
    generator: LykenoxVocoderGenerator,
    segments: list[VocoderSegment],
) -> float:
    generator.eval()
    values: list[float] = []
    with torch.no_grad():
        for segment in segments:
            prediction = generator(segment.mel.unsqueeze(0))
            target = segment.waveform.unsqueeze(0)
            loss = multi_resolution_reconstruction_loss(prediction, target).total
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError("Non-finite held-out vocoder reconstruction loss")
            values.append(value)
    generator.train()
    return statistics.fmean(values)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_listening_pairs(
    generator: LykenoxVocoderGenerator,
    segments: list[VocoderSegment],
    output_dir: Path,
    *,
    sample_rate: int,
    limit: int = 3,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    written: list[dict[str, object]] = []
    with torch.no_grad():
        for index, segment in enumerate(segments[:limit], start=1):
            generated = generator(segment.mel.unsqueeze(0)).squeeze(0).detach().cpu()
            target = segment.waveform.detach().cpu()
            generated_path = output_dir / f"val_{index:02d}_{segment.utterance_id}_generated.wav"
            reference_path = output_dir / f"val_{index:02d}_{segment.utterance_id}_reference.wav"
            sf.write(str(generated_path), generated.numpy(), sample_rate, subtype="PCM_16")
            sf.write(str(reference_path), target.numpy(), sample_rate, subtype="PCM_16")
            written.append(
                {
                    "utterance_id": segment.utterance_id,
                    "start_frame": segment.start_frame,
                    "mel_frames": segment.mel_frames,
                    "generated": str(generated_path),
                    "reference": str(reference_path),
                }
            )
    generator.train()
    return written


def _resume_if_compatible(
    last_path: Path,
    *,
    expected_provenance: dict[str, object],
    expected_run_config: dict[str, object],
) -> tuple[
    LykenoxVocoderGenerator,
    LykenoxMultiScaleWaveformDiscriminator,
    dict[str, object],
] | None:
    if not last_path.exists():
        return None
    generator, discriminator, payload = load_vocoder_checkpoint(last_path)
    if payload.get("training_provenance") != expected_provenance:
        raise RuntimeError(
            "Existing last.pt provenance does not match this short-training run. "
            "Do not resume with different data/configuration."
        )
    metadata = payload.get("training_metadata")
    if not isinstance(metadata, dict) or metadata.get("run_config") != expected_run_config:
        raise RuntimeError(
            "Existing last.pt short-training configuration does not match this command."
        )
    return generator, discriminator, payload


def run_persistent_vocoder_short_training(
    root: Path,
    *,
    segment_mel_frames: int = 96,
    train_items: int = 16,
    val_items: int = 6,
    max_epochs: int = 8,
    warmup_epochs: int = 2,
    patience: int = 3,
    seed: int = 1337,
    generator_lr: float = 2e-4,
    discriminator_lr: float = 2e-4,
    adversarial_weight: float = 0.10,
    feature_matching_weight: float = 2.0,
    min_delta: float = 1e-4,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    if segment_mel_frames < 32:
        raise ValueError("segment_mel_frames must be >= 32")
    if train_items < 2 or val_items < 2:
        raise ValueError("train_items and val_items must both be >= 2")
    if max_epochs < 1 or warmup_epochs < 0 or warmup_epochs > max_epochs:
        raise ValueError("Invalid epoch configuration")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if time_budget_seconds <= 5:
        raise ValueError("time_budget_seconds must be > 5")

    root = Path(root).resolve()
    started = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_segments, train_skipped = collect_vocoder_segments(
        root,
        "train",
        segment_mel_frames=segment_mel_frames,
        max_items=train_items,
        seed=seed,
    )
    val_segments, val_skipped = collect_vocoder_segments(
        root,
        "val",
        segment_mel_frames=segment_mel_frames,
        max_items=val_items,
        seed=seed,
    )

    provenance = build_vocoder_training_provenance(
        root,
        segment_mel_frames=segment_mel_frames,
        seed=seed,
    )
    train_set_sha = _segment_set_sha256(train_segments)
    val_set_sha = _segment_set_sha256(val_segments)
    run_config: dict[str, object] = {
        "contract_version": SHORT_TRAIN_CONTRACT_VERSION,
        "segment_mel_frames": segment_mel_frames,
        "train_items": train_items,
        "val_items": val_items,
        "max_epochs": max_epochs,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "seed": seed,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "adversarial_weight": adversarial_weight,
        "feature_matching_weight": feature_matching_weight,
        "min_delta": min_delta,
        "train_segment_set_sha256": train_set_sha,
        "val_segment_set_sha256": val_set_sha,
    }

    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_short_training"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = artifact_dir / "best.pt"
    last_path = artifact_dir / "last.pt"
    progress_path = artifact_dir / "training_progress.json"
    report_path = artifact_dir / "training_report.json"

    resumed = _resume_if_compatible(
        last_path,
        expected_provenance=provenance,
        expected_run_config=run_config,
    )
    if resumed is None:
        config = LykenoxVocoderConfig()
        generator = LykenoxVocoderGenerator(config).cpu().train()
        discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
        start_epoch = 1
        global_step = 0
        history: list[dict[str, object]] = []
        initial_validation = _reconstruction_mean(generator, val_segments)
        best_validation = initial_validation
        best_epoch = 0
        epochs_without_improvement = 0
    else:
        generator, discriminator, payload = resumed
        generator.train()
        discriminator.train()
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        metadata = dict(payload.get("training_metadata", {}))
        history = list(metadata.get("history", []))
        initial_validation = float(metadata["initial_validation_reconstruction"])
        best_validation = float(metadata["best_validation_reconstruction"])
        best_epoch = int(metadata["best_epoch"])
        epochs_without_improvement = int(metadata.get("epochs_without_improvement", 0))

    generator_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=generator_lr, weight_decay=1e-5
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=discriminator_lr, weight_decay=1e-5
    )
    if resumed is not None:
        payload = resumed[2]
        generator_state = payload.get("generator_optimizer_state")
        discriminator_state = payload.get("discriminator_optimizer_state")
        if not isinstance(generator_state, dict) or not isinstance(discriminator_state, dict):
            raise RuntimeError("Cannot resume vocoder short training without optimizer states")
        generator_optimizer.load_state_dict(generator_state)
        discriminator_optimizer.load_state_dict(discriminator_state)

    update_timings: list[float] = []
    stop_reason = "max_epochs_reached"
    incomplete = False

    for epoch in range(start_epoch, max_epochs + 1):
        order = list(range(len(train_segments)))
        random.Random(seed + epoch).shuffle(order)
        epoch_reconstruction: list[float] = []
        epoch_discriminator: list[float] = []
        epoch_adversarial: list[float] = []
        epoch_feature_matching: list[float] = []

        for position, segment_index in enumerate(order):
            if time.perf_counter() - started >= time_budget_seconds:
                incomplete = True
                stop_reason = "time_budget_reached"
                break

            segment = train_segments[segment_index]
            mel = segment.mel.unsqueeze(0)
            target = segment.waveform.unsqueeze(0)
            update_started = time.perf_counter()

            if epoch <= warmup_epochs:
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = generator(mel)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                if not torch.isfinite(reconstruction.total):
                    raise RuntimeError(
                        f"Non-finite reconstruction loss at epoch {epoch} item {position}"
                    )
                reconstruction.total.backward()
                generator_grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(generator_grad)):
                    raise RuntimeError("Non-finite generator gradient during warm-up")
                generator_optimizer.step()
                epoch_reconstruction.append(float(reconstruction.total.detach().cpu()))
            else:
                _set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_detached = generator(mel)
                real_output = discriminator(target)
                fake_output = discriminator(fake_detached)
                discriminator_loss = discriminator_hinge_loss(real_output, fake_output)
                if not torch.isfinite(discriminator_loss):
                    raise RuntimeError("Non-finite discriminator loss")
                discriminator_loss.backward()
                discriminator_grad = torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), 10.0
                )
                if not math.isfinite(float(discriminator_grad)):
                    raise RuntimeError("Non-finite discriminator gradient")
                discriminator_optimizer.step()

                _set_requires_grad(discriminator, False)
                generator_optimizer.zero_grad(set_to_none=True)
                fake = generator(mel)
                reconstruction = multi_resolution_reconstruction_loss(fake, target)
                with torch.no_grad():
                    real_features = discriminator(target)
                fake_features = discriminator(fake)
                adversarial = generator_adversarial_loss(fake_features)
                feature_match = feature_matching_loss(real_features, fake_features)
                generator_loss = (
                    reconstruction.total
                    + adversarial_weight * adversarial
                    + feature_matching_weight * feature_match
                )
                if not torch.isfinite(generator_loss):
                    raise RuntimeError("Non-finite adversarial generator loss")
                generator_loss.backward()
                generator_grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(generator_grad)):
                    raise RuntimeError("Non-finite adversarial generator gradient")
                generator_optimizer.step()
                _set_requires_grad(discriminator, True)

                epoch_reconstruction.append(float(reconstruction.total.detach().cpu()))
                epoch_discriminator.append(float(discriminator_loss.detach().cpu()))
                epoch_adversarial.append(float(adversarial.detach().cpu()))
                epoch_feature_matching.append(float(feature_match.detach().cpu()))

            update_timings.append(time.perf_counter() - update_started)
            global_step += 1

        # Never pretend a partial epoch is a comparable validation epoch.
        if incomplete:
            metadata = {
                "purpose": "persistent_vocoder_short_training",
                "run_config": run_config,
                "history": history,
                "initial_validation_reconstruction": initial_validation,
                "best_validation_reconstruction": best_validation,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "partial_epoch": epoch,
            }
            save_vocoder_checkpoint(
                last_path,
                generator,
                discriminator,
                epoch=epoch - 1,
                global_step=global_step,
                validation_reconstruction_loss=(history[-1]["validation_reconstruction"] if history else None),
                training_provenance=provenance,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                training_metadata=metadata,
            )
            progress = {
                "status": "incomplete",
                "stop_reason": stop_reason,
                "epochs_completed": epoch - 1,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "initial_validation_reconstruction": initial_validation,
                "best_validation_reconstruction": best_validation,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "next_gate": "rerun_same_command_to_resume",
            }
            _write_json(progress_path, progress)
            return {**progress, "progress_report": str(progress_path), "last_checkpoint": str(last_path)}

        validation = _reconstruction_mean(generator, val_segments)
        epoch_row: dict[str, object] = {
            "epoch": epoch,
            "phase": "reconstruction_warmup" if epoch <= warmup_epochs else "adversarial",
            "train_reconstruction": statistics.fmean(epoch_reconstruction),
            "validation_reconstruction": validation,
            "discriminator_loss": (
                statistics.fmean(epoch_discriminator) if epoch_discriminator else None
            ),
            "generator_adversarial_loss": (
                statistics.fmean(epoch_adversarial) if epoch_adversarial else None
            ),
            "feature_matching_loss": (
                statistics.fmean(epoch_feature_matching)
                if epoch_feature_matching
                else None
            ),
            "global_step": global_step,
        }
        history.append(epoch_row)

        improved = validation < best_validation - min_delta
        if improved:
            best_validation = validation
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        metadata = {
            "purpose": "persistent_vocoder_short_training",
            "run_config": run_config,
            "history": history,
            "initial_validation_reconstruction": initial_validation,
            "best_validation_reconstruction": best_validation,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        save_vocoder_checkpoint(
            last_path,
            generator,
            discriminator,
            epoch=epoch,
            global_step=global_step,
            validation_reconstruction_loss=validation,
            training_provenance=provenance,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            training_metadata=metadata,
        )
        if improved:
            save_vocoder_checkpoint(
                best_path,
                generator,
                discriminator,
                epoch=epoch,
                global_step=global_step,
                validation_reconstruction_loss=validation,
                training_provenance=provenance,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                training_metadata=metadata,
            )

        _write_json(
            progress_path,
            {
                "status": "running",
                "epoch": epoch,
                "max_epochs": max_epochs,
                "global_step": global_step,
                "initial_validation_reconstruction": initial_validation,
                "current_validation_reconstruction": validation,
                "best_validation_reconstruction": best_validation,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "last_epoch_metrics": epoch_row,
            },
        )

        if epochs_without_improvement >= patience:
            stop_reason = "early_stopping"
            break

    if not best_path.exists():
        # A short run can theoretically fail to beat its random initial validation.
        # Keep last.pt for diagnosis, but do not bless it as best.
        final_report = {
            "status": "needs_review",
            "stop_reason": stop_reason,
            "epochs_completed": int(history[-1]["epoch"]) if history else 0,
            "global_step": global_step,
            "initial_validation_reconstruction": initial_validation,
            "best_validation_reconstruction": best_validation,
            "validation_improved_over_initial": False,
            "history": history,
            "next_gate": "adjust_vocoder_short_training_recipe",
            "last_checkpoint": str(last_path),
        }
        _write_json(report_path, final_report)
        return {**final_report, "report_path": str(report_path)}

    best_generator, _, best_payload = load_vocoder_checkpoint(best_path)
    best_generator.train()
    recomputed_best_validation = _reconstruction_mean(best_generator, val_segments)
    stored_best_validation = float(best_payload["validation_reconstruction_loss"])
    validation_roundtrip_delta = abs(recomputed_best_validation - stored_best_validation)
    checkpoint_validation_exact = validation_roundtrip_delta < 1e-6
    validation_improved = recomputed_best_validation < initial_validation - min_delta

    listening_dir = artifact_dir / "listening"
    listening_pairs = _save_listening_pairs(
        best_generator,
        val_segments,
        listening_dir,
        sample_rate=best_generator.config.sample_rate,
        limit=3,
    )

    gate_pass = validation_improved and checkpoint_validation_exact
    final_status = "pass" if gate_pass else "needs_review"
    final_report = {
        "status": final_status,
        "device": "cpu",
        "contract_version": SHORT_TRAIN_CONTRACT_VERSION,
        "generator_architecture": "lykenox_compact_transposed_conv_v0",
        "segment_mel_frames": segment_mel_frames,
        "segment_audio_seconds": round(
            segment_mel_frames * best_generator.config.hop_length / best_generator.config.sample_rate,
            4,
        ),
        "train_segments": len(train_segments),
        "val_segments": len(val_segments),
        "train_skipped_count": len(train_skipped),
        "val_skipped_count": len(val_skipped),
        "train_segment_set_sha256": train_set_sha,
        "val_segment_set_sha256": val_set_sha,
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": int(best_payload["epoch"]),
        "global_step": global_step,
        "stop_reason": stop_reason,
        "initial_validation_reconstruction": round(initial_validation, 6),
        "best_validation_reconstruction": round(recomputed_best_validation, 6),
        "validation_improved_over_initial": validation_improved,
        "checkpoint_validation_exact": checkpoint_validation_exact,
        "checkpoint_validation_abs_delta": validation_roundtrip_delta,
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "listening_pairs": listening_pairs,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mean_seconds_per_update": (
            round(statistics.fmean(update_timings), 4) if update_timings else None
        ),
        "min_seconds_per_update": round(min(update_timings), 4) if update_timings else None,
        "max_seconds_per_update": round(max(update_timings), 4) if update_timings else None,
        "next_gate": (
            "listen_vocoder_validation_wavs"
            if gate_pass
            else "adjust_vocoder_short_training_recipe"
        ),
        "warning": (
            "A numerical pass does not establish perceptual vocoder quality. Compare the "
            "generated/reference validation WAV pairs before any longer training run."
        ),
    }
    _write_json(report_path, final_report)
    return {**final_report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=96)
    parser.add_argument("--train-items", type=int, default=16)
    parser.add_argument("--val-items", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    result = run_persistent_vocoder_short_training(
        args.root,
        segment_mel_frames=args.segment_mel_frames,
        train_items=args.train_items,
        val_items=args.val_items,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        seed=args.seed,
        time_budget_seconds=args.time_budget_seconds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
