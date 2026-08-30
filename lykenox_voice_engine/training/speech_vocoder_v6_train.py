"""Bounded, exactly resumable trainer for the LYKENOX v6 direct-waveform vocoder."""
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

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderGeneratorV6,
    VOCODER_GENERATOR_V6_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import LogMelEnvelopeLoss
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
    target_relative_level_loss,
    target_relative_presence_loss,
)
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
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
from lykenox_voice_engine.training.speech_vocoder_v6_artifact import (
    build_v6_training_provenance,
    load_v6_checkpoint,
    save_v6_checkpoint,
)

TRAINER_CONTRACT_VERSION = "v6-bounded-resumable-v1"
TRAIN_SEGMENT_SCHEDULE_VERSION = "epoch-crop-shuffle-v1"
DEFAULT_TIME_BUDGET_SECONDS = 80.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 10.0
DEFAULT_VALIDATION_RESERVE_SECONDS = 20.0
DEFAULT_VALIDATION_SEED_OFFSET = 100_003

METRIC_NAMES = (
    "reconstruction", "envelope", "spectral_balance", "local_spectral_contrast",
    "level", "rms_error_db", "presence", "presence_1k_8k_error_db",
    "prediction_rms_db", "target_rms_db",
    "prediction_band_80_300", "prediction_band_300_1000",
    "prediction_band_1k_3k", "prediction_band_3k_8k",
    "target_band_80_300", "target_band_300_1000",
    "target_band_1k_3k", "target_band_3k_8k", "selection_score",
)
ACCUM_NAMES = (
    "reconstruction", "envelope", "spectral_balance", "local_spectral_contrast",
    "level", "presence", "generator_total", "adversarial", "feature_matching",
    "discriminator",
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _segment_set_sha256(segments: list[object]) -> str:
    rows = [
        {k: getattr(s, k) for k in ("split", "utterance_id", "wav_path", "start_frame", "mel_frames")}
        for s in segments
    ]
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _condition(segments: list[object]) -> list[tuple[object, object]]:
    return [
        (s, extract_pitch_frames(s.waveform, frame_count=s.mel_frames))
        for s in segments
    ]


def _epoch_items(root: Path, epoch: int, seed: int, frames: int, count: int):
    segments, skipped = collect_vocoder_segments(
        root, "train", segment_mel_frames=frames, max_items=count, seed=seed + epoch
    )
    order = list(range(len(segments)))
    random.Random(seed + 1_000_003 + epoch).shuffle(order)
    return _condition([segments[i] for i in order]), len(skipped)


def _generate(generator: LykenoxVocoderGeneratorV6, item: tuple[object, object]) -> torch.Tensor:
    segment, pitch = item
    return generator(
        segment.mel.unsqueeze(0), pitch.f0_hz.unsqueeze(0), pitch.voiced.unsqueeze(0)
    )


def _losses(
    generator: LykenoxVocoderGeneratorV6,
    envelope_loss: LogMelEnvelopeLoss,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *, envelope_weight: float, balance_weight: float, contrast_weight: float,
    level_weight: float, presence_weight: float,
):
    reconstruction = multi_resolution_reconstruction_loss(prediction, target).total
    envelope = envelope_loss(prediction, target).total
    balance = target_relative_spectral_balance_loss(
        prediction, target, sample_rate=generator.config.sample_rate
    ).loss
    contrast = target_relative_local_spectral_contrast_loss(
        prediction, target, hop_length=generator.config.hop_length
    ).loss
    level = target_relative_level_loss(prediction, target)
    presence = target_relative_presence_loss(
        prediction, target, sample_rate=generator.config.sample_rate,
        hop_length=generator.config.hop_length,
    )
    total = (
        reconstruction + envelope_weight * envelope + balance_weight * balance
        + contrast_weight * contrast + level_weight * level.loss
        + presence_weight * presence.loss
    )
    m = {
        "reconstruction": reconstruction, "envelope": envelope,
        "spectral_balance": balance, "local_spectral_contrast": contrast,
        "level": level.loss, "rms_error_db": level.rms_error_db,
        "presence": presence.loss,
        "presence_1k_8k_error_db": presence.presence_1k_8k_error_db,
        "prediction_rms_db": level.prediction_rms_db,
        "target_rms_db": level.target_rms_db,
    }
    for prefix, values in (
        ("prediction", presence.prediction_band_fractions),
        ("target", presence.target_band_fractions),
    ):
        for name, value in zip(("80_300", "300_1000", "1k_3k", "3k_8k"), values, strict=True):
            m[f"{prefix}_band_{name}"] = value
    return total, m


def _selection(m: dict[str, float], ew: float, bw: float, cw: float, lw: float, pw: float) -> float:
    return (
        m["reconstruction"] + ew * m["envelope"] + bw * m["spectral_balance"]
        + cw * m["local_spectral_contrast"] + lw * m["level"] + pw * m["presence"]
    )


def _validate(generator, envelope_loss, items, *, ew, bw, cw, lw, pw) -> dict[str, float]:
    generator.eval()
    buckets = {name: [] for name in METRIC_NAMES if name != "selection_score"}
    with torch.no_grad():
        for item in items:
            pred = _generate(generator, item)
            target = item[0].waveform.unsqueeze(0)
            _, metrics = _losses(
                generator, envelope_loss, pred, target,
                envelope_weight=ew, balance_weight=bw, contrast_weight=cw,
                level_weight=lw, presence_weight=pw,
            )
            for name, tensor in metrics.items():
                value = float(tensor.detach())
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite held-out v6 metric: {name}")
                buckets[name].append(value)
    generator.train()
    means = {name: statistics.fmean(values) for name, values in buckets.items()}
    means["selection_score"] = _selection(means, ew, bw, cw, lw, pw)
    return means


def _optimizer(generator: LykenoxVocoderGeneratorV6, lr: float, level_mult: float):
    level = list(generator.level_parameters())
    level_ids = {id(p) for p in level}
    shape = [p for p in generator.parameters() if id(p) not in level_ids]
    if not shape or not level:
        raise RuntimeError("v6 optimizer grouping is incomplete")
    return torch.optim.AdamW([
        {"params": shape, "lr": lr, "weight_decay": 1e-5},
        {"params": level, "lr": lr * level_mult, "weight_decay": 0.0},
    ])


def _empty_accum(epoch: int) -> dict[str, object]:
    return {"epoch": epoch, "updates": 0, "adversarial_updates": 0, **{f"{n}_sum": 0.0 for n in ACCUM_NAMES}}


def _run_config(**kwargs: object) -> dict[str, object]:
    return {
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "generator_architecture": VOCODER_GENERATOR_V6_ARCHITECTURE,
        "source_family": LykenoxVocoderGeneratorV6.source_family,
        "explicit_source": False, "voiced_noise_source": False,
        "raw_source_bypass": False, "waveform_shape_level_decoupled": True,
        "level_control_family": LykenoxVocoderGeneratorV6.level_control_family,
        "level_presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
        "train_segment_schedule_version": TRAIN_SEGMENT_SCHEDULE_VERSION,
        **kwargs,
    }


def _metadata(run_config, history, initial, best, best_epoch, no_improve, partial, resumed):
    return {
        "purpose": "persistent_v6_direct_conditional_waveform_training",
        "run_config": run_config, "history": history,
        "initial_validation": initial, "best_validation": best,
        "best_epoch": best_epoch, "epochs_without_improvement": no_improve,
        "partial_epoch_state": partial, "resumed_invocations": resumed,
    }


def _save(path, generator, discriminator, gopt, dopt, epoch, step, offset, provenance, metadata, metrics):
    tmp = path.with_name(path.name + ".tmp")
    save_v6_checkpoint(
        tmp, generator, discriminator, epoch=epoch, global_step=step,
        next_item_offset=offset, validation_metrics=metrics,
        training_provenance=provenance, generator_optimizer=gopt,
        discriminator_optimizer=dopt, training_metadata=metadata,
    )
    os.replace(tmp, path)


def run_bounded_resumable_v6_training(
    root: Path, *, segment_mel_frames: int = 48, train_items: int = 118,
    val_items: int = 14, max_epochs: int = 28, warmup_epochs: int = 4,
    patience: int = 6, seed: int = 66000, generator_lr: float = 2e-4,
    level_lr_multiplier: float = 4.0, discriminator_lr: float = 1e-4,
    envelope_weight: float = 0.50, balance_weight: float = 0.25,
    contrast_weight: float = 0.15, level_weight: float = 0.75,
    presence_weight: float = 0.35, adversarial_weight: float = 0.03,
    feature_matching_weight: float = 0.50, gradient_clip_norm: float = 5.0,
    min_delta: float = 1e-4, checkpoint_every_updates: int = 6,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    checkpoint_reserve_seconds: float = DEFAULT_CHECKPOINT_RESERVE_SECONDS,
    validation_reserve_seconds: float = DEFAULT_VALIDATION_RESERVE_SECONDS,
    max_updates_this_run: int | None = None, artifact_dir_override: Path | None = None,
) -> dict[str, object]:
    if segment_mel_frames < 32 or train_items < 2 or val_items < 2:
        raise ValueError("invalid v6 data bounds")
    if max_epochs < 1 or patience < 1 or not 0 <= warmup_epochs <= max_epochs:
        raise ValueError("invalid v6 epoch bounds")
    if not 1.0 <= level_lr_multiplier <= 16.0:
        raise ValueError("level_lr_multiplier must be between 1 and 16")
    positive = (generator_lr, discriminator_lr, envelope_weight, balance_weight,
                contrast_weight, level_weight, presence_weight, gradient_clip_norm)
    if min(positive) <= 0 or checkpoint_every_updates < 1:
        raise ValueError("positive v6 optimizer/loss parameters required")
    if adversarial_weight < 0 or feature_matching_weight < 0:
        raise ValueError("invalid adversarial weights")
    if max_updates_this_run is not None and max_updates_this_run < 1:
        raise ValueError("max_updates_this_run must be positive")

    root = Path(root).resolve()
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    val_seed = seed + DEFAULT_VALIDATION_SEED_OFFSET
    val_segments, val_skipped = collect_vocoder_segments(
        root, "val", segment_mel_frames=segment_mel_frames, max_items=val_items, seed=val_seed
    )
    val = _condition(val_segments)
    provenance = build_v6_training_provenance(root, segment_mel_frames=segment_mel_frames, seed=seed)
    run_config = _run_config(
        segment_mel_frames=segment_mel_frames, train_items=train_items, val_items=val_items,
        max_epochs=max_epochs, warmup_epochs=warmup_epochs, patience=patience, seed=seed,
        validation_seed=val_seed, generator_lr=generator_lr,
        level_lr_multiplier=level_lr_multiplier, discriminator_lr=discriminator_lr,
        envelope_weight=envelope_weight, balance_weight=balance_weight,
        contrast_weight=contrast_weight, level_weight=level_weight,
        presence_weight=presence_weight, adversarial_weight=adversarial_weight,
        feature_matching_weight=feature_matching_weight, gradient_clip_norm=gradient_clip_norm,
        min_delta=min_delta, checkpoint_every_updates=checkpoint_every_updates,
        validation_segment_set_sha256=_segment_set_sha256(val_segments),
    )
    artifact_dir = Path(artifact_dir_override).resolve() if artifact_dir_override else (
        root / "models" / "lykenox_identity" / "training" / "vocoder_direct_waveform_v6"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = artifact_dir / "last.pt", artifact_dir / "best.pt"
    progress_path, report_path = artifact_dir / "training_progress.json", artifact_dir / "training_report.json"
    envelope_loss = LogMelEnvelopeLoss().cpu()

    resumed_payload = None
    if last_path.exists():
        generator, discriminator, resumed_payload = load_v6_checkpoint(last_path)
        if resumed_payload.get("training_provenance") != provenance:
            raise RuntimeError("Existing v6 checkpoint provenance differs from active data")
        meta = resumed_payload.get("training_metadata")
        if not isinstance(meta, dict) or meta.get("run_config") != run_config:
            raise RuntimeError("Existing v6 checkpoint configuration differs from this command")
        epoch = int(resumed_payload["epoch"]); step = int(resumed_payload["global_step"])
        offset = int(resumed_payload["next_item_offset"])
        history = list(meta.get("history", []))
        initial = dict(meta["initial_validation"]); best = dict(meta["best_validation"])
        best_epoch = int(meta.get("best_epoch", 0)); no_improve = int(meta.get("epochs_without_improvement", 0))
        partial = meta.get("partial_epoch_state")
        accum = dict(partial) if isinstance(partial, dict) and int(partial.get("epoch", -1)) == epoch else _empty_accum(epoch)
        resumed = int(meta.get("resumed_invocations", 0)) + 1
        rng = resumed_payload.get("torch_rng_state")
        if not isinstance(rng, torch.Tensor):
            raise RuntimeError("Cannot exactly resume v6 without torch RNG state")
        torch.set_rng_state(rng)
        generator.train(); discriminator.train()
    else:
        torch.manual_seed(seed)
        generator = LykenoxVocoderGeneratorV6().cpu().train()
        discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
        initial = _validate(generator, envelope_loss, val, ew=envelope_weight, bw=balance_weight,
                            cw=contrast_weight, lw=level_weight, pw=presence_weight)
        best = dict(initial); epoch = 1; step = 0; offset = 0; history = []
        best_epoch = 0; no_improve = 0; accum = _empty_accum(epoch); resumed = 0

    gopt = _optimizer(generator, generator_lr, level_lr_multiplier)
    dopt = torch.optim.AdamW(discriminator.parameters(), lr=discriminator_lr, weight_decay=1e-5)
    if resumed_payload is not None:
        gs, ds = resumed_payload.get("generator_optimizer_state"), resumed_payload.get("discriminator_optimizer_state")
        if not isinstance(gs, dict) or not isinstance(ds, dict):
            raise RuntimeError("Cannot exactly resume v6 without both optimizer states")
        gopt.load_state_dict(gs); dopt.load_state_dict(ds)

    update_times: list[float] = []
    updates_this_run = 0
    train_skipped = 0

    def meta(partial):
        return _metadata(run_config, history, initial, best, best_epoch, no_improve, partial, resumed)

    def interrupt(reason: str):
        partial = accum if offset > 0 else None
        current = dict(history[-1]["validation"]) if history else dict(initial)
        _save(last_path, generator, discriminator, gopt, dopt, epoch, step, offset, provenance, meta(partial), current)
        out = {
            "status": "incomplete", "device": "cpu", "trainer_contract_version": TRAINER_CONTRACT_VERSION,
            "architecture": VOCODER_GENERATOR_V6_ARCHITECTURE, "stop_reason": reason,
            "epochs_completed": len(history), "current_epoch": epoch, "next_item_offset": offset,
            "global_step": step, "updates_this_run": updates_this_run, "best_epoch": best_epoch,
            "best_validation_selection_score": round(float(best["selection_score"]), 6),
            "best_validation_rms_error_db": round(float(best["rms_error_db"]), 6),
            "best_validation_presence_error_db": round(float(best["presence_1k_8k_error_db"]), 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "last_checkpoint": str(last_path), "best_checkpoint": str(best_path) if best_path.exists() else None,
            "persistent_training_complete": False, "historical_checkpoints_mutated": False,
            "next_gate": "rerun_same_command_to_resume",
        }
        _atomic_json(progress_path, out)
        return {**out, "progress_report": str(progress_path)}

    while epoch <= max_epochs:
        items, train_skipped = _epoch_items(root, epoch, seed, segment_mel_frames, train_items)
        if offset > len(items):
            raise RuntimeError("v6 resume offset exceeds deterministic epoch length")
        while offset < len(items):
            estimated = max(update_times[-4:] or [2.0])
            last_item = offset == len(items) - 1
            reserve = checkpoint_reserve_seconds + estimated + (validation_reserve_seconds if last_item else 0)
            if time.perf_counter() - started + reserve >= time_budget_seconds:
                return interrupt("time_budget")
            if max_updates_this_run is not None and updates_this_run >= max_updates_this_run:
                return interrupt("max_updates_this_run")

            item = items[offset]; target = item[0].waveform.unsqueeze(0)
            update_started = time.perf_counter()
            gopt.zero_grad(set_to_none=True)
            pred = _generate(generator, item)
            gtotal, base = _losses(
                generator, envelope_loss, pred, target, envelope_weight=envelope_weight,
                balance_weight=balance_weight, contrast_weight=contrast_weight,
                level_weight=level_weight, presence_weight=presence_weight,
            )
            adv = fm = dval = 0.0
            adv_active = epoch > warmup_epochs and (adversarial_weight > 0 or feature_matching_weight > 0)
            if adv_active:
                for p in discriminator.parameters(): p.requires_grad_(False)
                with torch.no_grad(): real_features = discriminator(target)
                fake_features = discriminator(pred)
                adv_t = generator_adversarial_loss(fake_features)
                fm_t = feature_matching_loss(real_features, fake_features)
                gtotal = gtotal + adversarial_weight * adv_t + feature_matching_weight * fm_t
                adv, fm = float(adv_t.detach()), float(fm_t.detach())
            if not bool(torch.isfinite(gtotal)):
                raise RuntimeError("Non-finite v6 generator loss")
            gtotal.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip_norm)
            if not bool(torch.isfinite(gnorm)): raise RuntimeError("Non-finite v6 generator gradient")
            gopt.step()
            for p in discriminator.parameters(): p.requires_grad_(True)
            if adv_active:
                dopt.zero_grad(set_to_none=True)
                dloss = discriminator_hinge_loss(discriminator(target), discriminator(pred.detach()))
                if not bool(torch.isfinite(dloss)): raise RuntimeError("Non-finite v6 discriminator loss")
                dloss.backward()
                dnorm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), gradient_clip_norm)
                if not bool(torch.isfinite(dnorm)): raise RuntimeError("Non-finite v6 discriminator gradient")
                dopt.step(); dval = float(dloss.detach())

            train_values = {
                "reconstruction": float(base["reconstruction"].detach()),
                "envelope": float(base["envelope"].detach()),
                "spectral_balance": float(base["spectral_balance"].detach()),
                "local_spectral_contrast": float(base["local_spectral_contrast"].detach()),
                "level": float(base["level"].detach()), "presence": float(base["presence"].detach()),
                "generator_total": float(gtotal.detach()), "adversarial": adv,
                "feature_matching": fm, "discriminator": dval,
            }
            if not all(math.isfinite(v) for v in train_values.values()):
                raise RuntimeError("Non-finite v6 training metric")
            for name, value in train_values.items(): accum[f"{name}_sum"] = float(accum[f"{name}_sum"]) + value
            accum["updates"] = int(accum["updates"]) + 1
            if adv_active: accum["adversarial_updates"] = int(accum["adversarial_updates"]) + 1
            step += 1; offset += 1; updates_this_run += 1
            update_times.append(time.perf_counter() - update_started)
            if step % checkpoint_every_updates == 0:
                current = dict(history[-1]["validation"]) if history else dict(initial)
                _save(last_path, generator, discriminator, gopt, dopt, epoch, step, offset, provenance, meta(accum), current)

        validation = _validate(generator, envelope_loss, val, ew=envelope_weight, bw=balance_weight,
                               cw=contrast_weight, lw=level_weight, pw=presence_weight)
        u = max(1, int(accum["updates"])); au = max(1, int(accum["adversarial_updates"]))
        train_metrics = {
            name: float(accum[f"{name}_sum"]) / (au if name in ("adversarial", "feature_matching", "discriminator") else u)
            for name in ACCUM_NAMES
        }
        history.append({"epoch": epoch, "global_step": step, "train": train_metrics, "validation": dict(validation)})
        if validation["selection_score"] < best["selection_score"] - min_delta:
            best = dict(validation); best_epoch = epoch; no_improve = 0
            _save(best_path, generator, discriminator, gopt, dopt, epoch + 1, step, 0, provenance, meta(None), validation)
        else:
            no_improve += 1
        epoch += 1; offset = 0; accum = _empty_accum(epoch)
        _save(last_path, generator, discriminator, gopt, dopt, epoch, step, 0, provenance, meta(None), validation)
        if len(history) >= warmup_epochs and no_improve >= patience:
            stop_reason = "early_stopping"; break
    else:
        stop_reason = "max_epochs"

    checks = {
        "training_improved": best["selection_score"] < initial["selection_score"],
        "envelope_improved": best["envelope"] < initial["envelope"],
        "spectral_balance_improved": best["spectral_balance"] < initial["spectral_balance"],
        "level_improved": best["level"] < initial["level"],
        "rms_error_improved": best["rms_error_db"] < initial["rms_error_db"],
        "presence_improved": best["presence"] < initial["presence"],
        "presence_error_improved": best["presence_1k_8k_error_db"] < initial["presence_1k_8k_error_db"],
    }
    passed = all(checks.values()) and best_path.exists()
    rounded = lambda m: {name: round(float(m[name]), 6) for name in METRIC_NAMES}
    report = {
        "status": "pass" if passed else "fail", "device": "cpu",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "architecture": VOCODER_GENERATOR_V6_ARCHITECTURE, "source_family": generator.source_family,
        "explicit_source": False, "voiced_noise_source": False, "raw_source_bypass": False,
        "waveform_shape_level_decoupled": True, "level_control_family": generator.level_control_family,
        "level_presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
        "stop_reason": stop_reason, "epochs_completed": len(history), "global_step": step,
        "best_epoch": best_epoch, "initial_validation": rounded(initial), "best_validation": rounded(best),
        **checks, "train_items": train_items, "val_items": len(val), "val_skipped": len(val_skipped),
        "train_skipped_latest_epoch": train_skipped, "segment_mel_frames": segment_mel_frames,
        "generator_lr": generator_lr, "level_lr_multiplier": level_lr_multiplier,
        "resumed_invocations": resumed, "best_checkpoint": str(best_path) if best_path.exists() else None,
        "last_checkpoint": str(last_path), "persistent_training_complete": True,
        "full_utterance_perceptual_acceptance": False, "historical_checkpoints_mutated": False,
        "reference_audio_required_for_product_inference": False,
        "next_gate": "run_v6_full_utterance_oracle_acceptance" if passed else "inspect_v6_training_failure_before_any_more_training",
    }
    _atomic_json(report_path, report); _atomic_json(progress_path, report)
    return {**report, "training_report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_bounded_resumable_v6_training(
        args.root, time_budget_seconds=args.time_budget_seconds,
        max_updates_this_run=args.max_updates_this_run,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
