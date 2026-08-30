"""Bounded, exactly resumable V7 trainer hard-gated after the first epoch."""
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

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV7, VOCODER_GENERATOR_V7_ARCHITECTURE
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import target_relative_level_loss, target_relative_presence_loss
from lykenox_voice_engine.training.speech_vocoder_losses import multi_resolution_reconstruction_loss
from lykenox_voice_engine.training.speech_vocoder_v7_artifact import build_v7_training_provenance, load_v7_checkpoint, save_v7_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v7_content_loss import VOCODER_V7_CONTENT_LOSS_VERSION, V7MelContentConsistencyLoss

TRAINER_CONTRACT_VERSION = "v7-first-epoch-bounded-resumable-v1"
TRAIN_SEGMENT_SCHEDULE_VERSION = "epoch-crop-shuffle-v1"
ARTIFACT_DIR_NAME = "vocoder_source_free_v7_first_epoch"
DEFAULT_TIME_BUDGET_SECONDS = 80.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 10.0
DEFAULT_VALIDATION_RESERVE_SECONDS = 20.0
DEFAULT_VALIDATION_SEED_OFFSET = 100_003


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, path)


def _segment_set_sha256(segments: list[object]) -> str:
    rows = [{k: getattr(s, k) for k in ("split", "utterance_id", "wav_path", "start_frame", "mel_frames")} for s in segments]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _condition(segments: list[object]) -> list[tuple[object, object]]:
    return [(s, extract_pitch_frames(s.waveform, frame_count=s.mel_frames)) for s in segments]


def _epoch_items(root: Path, epoch: int, seed: int, frames: int, count: int):
    segments, skipped = collect_vocoder_segments(root, "train", segment_mel_frames=frames, max_items=count, seed=seed + epoch)
    order = list(range(len(segments))); random.Random(seed + 1_000_003 + epoch).shuffle(order)
    return _condition([segments[i] for i in order]), len(skipped)


def _generate(generator: LykenoxVocoderGeneratorV7, item: tuple[object, object]) -> torch.Tensor:
    segment, pitch = item
    return generator(segment.mel.unsqueeze(0), pitch.f0_hz.unsqueeze(0), pitch.voiced.unsqueeze(0))


def _train_losses(generator: LykenoxVocoderGeneratorV7, content_loss: V7MelContentConsistencyLoss, prediction: torch.Tensor, target: torch.Tensor, mel: torch.Tensor, *, content_weight: float, level_weight: float):
    reconstruction = multi_resolution_reconstruction_loss(prediction, target).total
    content = content_loss(prediction, mel)
    level = target_relative_level_loss(prediction, target)
    total = reconstruction + content_weight * content.total + level_weight * level.loss
    return total, reconstruction, content, level


def _validate(generator, content_loss, items, *, content_weight: float, level_weight: float) -> dict[str, float]:
    names = ("reconstruction", "content", "content_log_mel", "content_centered_shape", "content_spectral_delta", "content_temporal_delta", "content_temporal_acceleration", "level", "rms_error_db", "prediction_rms_db", "target_rms_db", "presence_1k_8k_error_db", "prediction_band_80_300", "prediction_band_300_1000", "prediction_band_1k_3k", "prediction_band_3k_8k", "target_band_80_300", "target_band_300_1000", "target_band_1k_3k", "target_band_3k_8k")
    buckets = {name: [] for name in names}; generator.eval()
    with torch.no_grad():
        for item in items:
            segment = item[0]; prediction = _generate(generator, item); target = segment.waveform.unsqueeze(0); mel = segment.mel.unsqueeze(0)
            _, reconstruction, content, level = _train_losses(generator, content_loss, prediction, target, mel, content_weight=content_weight, level_weight=level_weight)
            presence = target_relative_presence_loss(prediction, target, sample_rate=generator.config.sample_rate, hop_length=generator.config.hop_length)
            values = {"reconstruction": reconstruction, "content": content.total, "content_log_mel": content.log_mel_l1, "content_centered_shape": content.centered_shape_l1, "content_spectral_delta": content.spectral_delta_l1, "content_temporal_delta": content.temporal_delta_l1, "content_temporal_acceleration": content.temporal_acceleration_l1, "level": level.loss, "rms_error_db": level.rms_error_db, "prediction_rms_db": level.prediction_rms_db, "target_rms_db": level.target_rms_db, "presence_1k_8k_error_db": presence.presence_1k_8k_error_db}
            for prefix, band_values in (("prediction", presence.prediction_band_fractions), ("target", presence.target_band_fractions)):
                for band, value in zip(("80_300", "300_1000", "1k_3k", "3k_8k"), band_values, strict=True): values[f"{prefix}_band_{band}"] = value
            for name, tensor in values.items():
                value = float(tensor.detach())
                if not math.isfinite(value): raise RuntimeError(f"Non-finite held-out v7 metric: {name}")
                buckets[name].append(value)
    generator.train(); means = {name: statistics.fmean(values) for name, values in buckets.items()}
    means["selection_score"] = means["reconstruction"] + content_weight * means["content"] + level_weight * means["level"]
    return means


def _run_config(**kwargs: object) -> dict[str, object]:
    return {"trainer_contract_version": TRAINER_CONTRACT_VERSION, "generator_architecture": VOCODER_GENERATOR_V7_ARCHITECTURE, "source_free": True, "sample_phase_conditioning": False, "sample_rate_pitch_features": False, "pitch_conditioning_scope": "frame_latent_only", "deterministic_noise_conditioning": False, "level_rescue_branch": False, "v7_content_loss_version": VOCODER_V7_CONTENT_LOSS_VERSION, "train_segment_schedule_version": TRAIN_SEGMENT_SCHEDULE_VERSION, "hard_epoch_limit": 1, **kwargs}


def _metadata(run_config, history, initial, best, best_epoch, partial, resumed):
    return {"purpose": "v7_first_epoch_pre_oracle_training", "run_config": run_config, "history": history, "initial_validation": initial, "best_validation": best, "best_epoch": best_epoch, "partial_epoch_state": partial, "resumed_invocations": resumed, "full_utterance_perceptual_acceptance": False}


def _save(path, generator, optimizer, epoch, step, offset, provenance, metadata, metrics):
    tmp = path.with_name(path.name + ".tmp")
    save_v7_checkpoint(tmp, generator, epoch=epoch, global_step=step, next_item_offset=offset, validation_metrics=metrics, training_provenance=provenance, generator_optimizer=optimizer, training_metadata=metadata)
    os.replace(tmp, path)


def run_bounded_resumable_v7_first_epoch(root: Path, *, segment_mel_frames: int = 48, train_items: int = 118, val_items: int = 14, seed: int = 77000, generator_lr: float = 2e-4, content_weight: float = 0.75, level_weight: float = 0.25, gradient_clip_norm: float = 5.0, min_delta: float = 1e-4, checkpoint_every_updates: int = 6, time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS, checkpoint_reserve_seconds: float = DEFAULT_CHECKPOINT_RESERVE_SECONDS, validation_reserve_seconds: float = DEFAULT_VALIDATION_RESERVE_SECONDS, max_updates_this_run: int | None = None, artifact_dir_override: Path | None = None) -> dict[str, object]:
    if segment_mel_frames < 32 or train_items < 2 or val_items < 2: raise ValueError("invalid v7 data bounds")
    if min(generator_lr, content_weight, level_weight, gradient_clip_norm) <= 0 or checkpoint_every_updates < 1: raise ValueError("positive v7 optimizer/loss parameters required")
    if max_updates_this_run is not None and max_updates_this_run < 1: raise ValueError("max_updates_this_run must be positive")
    root = Path(root).resolve(); started = time.perf_counter(); torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    val_seed = seed + DEFAULT_VALIDATION_SEED_OFFSET
    val_segments, val_skipped = collect_vocoder_segments(root, "val", segment_mel_frames=segment_mel_frames, max_items=val_items, seed=val_seed)
    val = _condition(val_segments); provenance = build_v7_training_provenance(root, segment_mel_frames=segment_mel_frames, seed=seed)
    run_config = _run_config(segment_mel_frames=segment_mel_frames, train_items=train_items, val_items=val_items, seed=seed, validation_seed=val_seed, generator_lr=generator_lr, content_weight=content_weight, level_weight=level_weight, gradient_clip_norm=gradient_clip_norm, min_delta=min_delta, checkpoint_every_updates=checkpoint_every_updates, validation_segment_set_sha256=_segment_set_sha256(val_segments))
    artifact_dir = Path(artifact_dir_override).resolve() if artifact_dir_override else root / "models" / "lykenox_identity" / "training" / ARTIFACT_DIR_NAME
    artifact_dir.mkdir(parents=True, exist_ok=True); last_path, best_path = artifact_dir / "last.pt", artifact_dir / "best.pt"; progress_path = artifact_dir / "training_progress.json"
    content_loss = V7MelContentConsistencyLoss().cpu(); resumed_payload = None
    if last_path.exists():
        generator, resumed_payload = load_v7_checkpoint(last_path); meta = resumed_payload.get("training_metadata")
        if resumed_payload.get("training_provenance") != provenance: raise RuntimeError("Existing v7 checkpoint provenance differs from active data")
        if not isinstance(meta, dict) or meta.get("run_config") != run_config: raise RuntimeError("Existing v7 checkpoint configuration differs from this command")
        history = list(meta.get("history", []))
        if history:
            return {"status": "gate_reached", "trainer_contract_version": TRAINER_CONTRACT_VERSION, "architecture": VOCODER_GENERATOR_V7_ARCHITECTURE, "epochs_completed": 1, "current_epoch": 2, "next_item_offset": 0, "global_step": int(resumed_payload["global_step"]), "best_epoch": int(meta.get("best_epoch", 1)), "persistent_training_complete": False, "full_utterance_perceptual_acceptance": False, "next_gate": "run_v7_full_utterance_oracle_vs_v4_2_before_any_epoch2"}
        epoch = int(resumed_payload["epoch"]); step = int(resumed_payload["global_step"]); offset = int(resumed_payload["next_item_offset"]); initial = dict(meta["initial_validation"]); best = dict(meta["best_validation"]); best_epoch = int(meta.get("best_epoch", 0)); partial = meta.get("partial_epoch_state"); accum = dict(partial) if isinstance(partial, dict) else {"epoch": 1, "updates": 0, "total_sum": 0.0}; resumed = int(meta.get("resumed_invocations", 0)) + 1
        rng = resumed_payload.get("torch_rng_state");
        if not isinstance(rng, torch.Tensor): raise RuntimeError("Cannot exactly resume v7 without torch RNG state")
        torch.set_rng_state(rng); generator.train()
    else:
        torch.manual_seed(seed); generator = LykenoxVocoderGeneratorV7().cpu().train(); initial = _validate(generator, content_loss, val, content_weight=content_weight, level_weight=level_weight); best = dict(initial); best_epoch = 0; epoch = 1; step = 0; offset = 0; history = []; accum = {"epoch": 1, "updates": 0, "total_sum": 0.0}; resumed = 0
    optimizer = torch.optim.AdamW(generator.parameters(), lr=generator_lr, weight_decay=1e-5)
    if resumed_payload is not None:
        state = resumed_payload.get("generator_optimizer_state")
        if not isinstance(state, dict): raise RuntimeError("Cannot exactly resume v7 without optimizer state")
        optimizer.load_state_dict(state)
    update_times: list[float] = []; updates_this_run = 0
    def meta(partial): return _metadata(run_config, history, initial, best, best_epoch, partial, resumed)
    def interrupt(reason: str):
        current = dict(history[-1]["validation"]) if history else dict(initial); _save(last_path, generator, optimizer, epoch, step, offset, provenance, meta(accum if offset > 0 else None), current)
        out = {"status": "incomplete", "stop_reason": reason, "trainer_contract_version": TRAINER_CONTRACT_VERSION, "architecture": VOCODER_GENERATOR_V7_ARCHITECTURE, "epochs_completed": len(history), "current_epoch": epoch, "next_item_offset": offset, "global_step": step, "updates_this_run": updates_this_run, "best_epoch": best_epoch, "best_validation_selection_score": round(float(best["selection_score"]), 6), "best_validation_rms_error_db": round(float(best["rms_error_db"]), 6), "best_validation_presence_error_db": round(float(best["presence_1k_8k_error_db"]), 6), "elapsed_seconds": round(time.perf_counter() - started, 3), "last_checkpoint": str(last_path), "best_checkpoint": str(best_path) if best_path.exists() else None, "persistent_training_complete": False, "full_utterance_perceptual_acceptance": False, "next_gate": "rerun_same_command_to_resume_first_epoch"}
        _atomic_json(progress_path, out); return out
    items, train_skipped = _epoch_items(root, 1, seed, segment_mel_frames, train_items)
    if offset > len(items): raise RuntimeError("v7 resume offset exceeds deterministic epoch length")
    while offset < len(items):
        estimated = max(update_times[-4:] or [2.0]); last_item = offset == len(items) - 1; reserve = checkpoint_reserve_seconds + estimated + (validation_reserve_seconds if last_item else 0)
        if time.perf_counter() - started + reserve >= time_budget_seconds: return interrupt("time_budget")
        if max_updates_this_run is not None and updates_this_run >= max_updates_this_run: return interrupt("max_updates_this_run")
        item = items[offset]; segment = item[0]; target = segment.waveform.unsqueeze(0); mel = segment.mel.unsqueeze(0); update_started = time.perf_counter(); optimizer.zero_grad(set_to_none=True); prediction = _generate(generator, item)
        total, _, _, _ = _train_losses(generator, content_loss, prediction, target, mel, content_weight=content_weight, level_weight=level_weight)
        if not bool(torch.isfinite(total)): raise RuntimeError("Non-finite v7 generator loss")
        total.backward(); gnorm = torch.nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip_norm)
        if not bool(torch.isfinite(gnorm)): raise RuntimeError("Non-finite v7 generator gradient")
        optimizer.step(); accum["updates"] = int(accum["updates"]) + 1; accum["total_sum"] = float(accum["total_sum"]) + float(total.detach()); step += 1; offset += 1; updates_this_run += 1; update_times.append(time.perf_counter() - update_started)
        if step % checkpoint_every_updates == 0: _save(last_path, generator, optimizer, 1, step, offset, provenance, meta(accum), dict(initial))
    validation = _validate(generator, content_loss, val, content_weight=content_weight, level_weight=level_weight); train_mean = float(accum["total_sum"]) / max(1, int(accum["updates"])); history.append({"epoch": 1, "updates": int(accum["updates"]), "train_total": train_mean, "validation": dict(validation)})
    if validation["selection_score"] < best["selection_score"] - min_delta: best = dict(validation); best_epoch = 1
    epoch = 2; offset = 0; final_meta = meta(None); _save(last_path, generator, optimizer, epoch, step, offset, provenance, final_meta, validation)
    if best_epoch == 1: _save(best_path, generator, optimizer, epoch, step, offset, provenance, final_meta, validation)
    out = {"status": "gate_reached", "trainer_contract_version": TRAINER_CONTRACT_VERSION, "architecture": VOCODER_GENERATOR_V7_ARCHITECTURE, "epochs_completed": 1, "current_epoch": 2, "next_item_offset": 0, "global_step": step, "updates_this_run": updates_this_run, "best_epoch": best_epoch, "initial_validation": initial, "epoch1_validation": validation, "train_total": train_mean, "train_skipped": train_skipped, "val_skipped": val_skipped, "last_checkpoint": str(last_path), "best_checkpoint": str(best_path) if best_path.exists() else None, "persistent_training_complete": False, "full_utterance_perceptual_acceptance": False, "epoch2_training_blocked": True, "predicted_duration_modified": False, "posthoc_gain_normalization_used": False, "posthoc_eq_used": False, "posthoc_denoising_used": False, "next_gate": "run_v7_full_utterance_oracle_vs_v4_2_before_any_epoch2"}
    _atomic_json(progress_path, out); return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS); parser.add_argument("--max-updates-this-run", type=int, default=None); args = parser.parse_args()
    print(json.dumps(run_bounded_resumable_v7_first_epoch(args.root, time_budget_seconds=args.time_budget_seconds, max_updates_this_run=args.max_updates_this_run), indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
