"""CPU-first trainer for the LYKENOX pitch-synchronous real residual cycle source.

The target is the owned Step-3f real residual, but voiced learning is performed in F0-relative phase
coordinates rather than absolute 512/256 frame-grid sample phase.  This is the root representation
change after V2 produced good pronunciation/level without gangoso but retained robotic timbre.

The acoustic context encoder is warm-started from the owned V2 checkpoint when available.  Cycle
recurrent dynamics and cycle heads are learned from TRAIN real residual cycles.  There is no teacher
forcing, codebook, external model/weight/service, post-hoc enhancement, or duration modification.
Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lykenox_voice_engine.models.vocoder.network_pitch_synchronous_residual_cycle_source_v1 import (
    CYCLE_PHASE_BINS,
    PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE,
    LykenoxPitchSynchronousResidualCycleSourceV1,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import (
    _load_or_build_target,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v2 import (
    CHECKPOINT_SCHEMA_VERSION as V2_CHECKPOINT_SCHEMA_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    OwnedVocoderUtterance,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    SAMPLE_RATE,
    fixed_linear_frame_to_sample,
)


TRAINER_VERSION = "owned-pitch-synchronous-residual-cycle-source-trainer-v1"
CHECKPOINT_SCHEMA_VERSION = "owned-pitch-synchronous-residual-cycle-source-checkpoint-v1"
TARGET_CACHE_VERSION = "owned-pitch-synchronous-real-residual-cycle-target-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_TRAIN_ITEMS = 48
DEFAULT_VAL_ITEMS = 3
DEFAULT_MAX_UPDATES = 600
DEFAULT_CYCLES_PER_UPDATE = 64
DEFAULT_CONTEXT_MARGIN_FRAMES = 16
DEFAULT_LEARNING_RATE = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_EVAL_EVERY = 50
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_SEED = 20260917
MIN_CYCLE_SAMPLES = 24
MAX_CYCLE_SAMPLES = 800
EPSILON = 1.0e-7


@dataclass(frozen=True)
class PitchSynchronousCycleTarget:
    start_sample: int
    end_sample: int
    frame_index: int
    periodicity: float
    canonical_cycle: torch.Tensor


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _canonicalize_cycle(cycle: torch.Tensor) -> torch.Tensor:
    if cycle.ndim != 1 or cycle.numel() < 2:
        raise ValueError("cycle must be a one-dimensional waveform")
    value = F.interpolate(
        cycle.view(1, 1, -1).to(torch.float32),
        size=CYCLE_PHASE_BINS,
        mode="linear",
        align_corners=True,
    )[0, 0]
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("canonical cycle contains non-finite values")
    return value.contiguous()


def conditioning_cycle_boundaries(
    utterance: OwnedVocoderUtterance,
) -> list[tuple[int, int, int, float]]:
    """Return voiced F0-cycle boundaries using conditioning only.

    Each tuple is ``(start_sample, end_sample, nearest_frame, mean_periodicity)``.  No target residual
    or held-out waveform sample is used to determine generation boundaries.
    """
    frame_f0 = utterance.f0_hz.unsqueeze(0).to(torch.float32)
    frame_voiced = utterance.voiced.unsqueeze(0).to(torch.float32)
    frame_periodicity = utterance.periodicity.unsqueeze(0).to(torch.float32)
    sample_f0 = fixed_linear_frame_to_sample(frame_f0, hop_length=HOP_LENGTH).squeeze(0)
    sample_voiced = fixed_linear_frame_to_sample(frame_voiced, hop_length=HOP_LENGTH).squeeze(0)
    sample_periodicity = fixed_linear_frame_to_sample(frame_periodicity, hop_length=HOP_LENGTH).squeeze(0)
    phase_increment = torch.where(
        (sample_f0 > 0.0) & (sample_voiced >= 0.5),
        sample_f0 / float(SAMPLE_RATE),
        torch.zeros_like(sample_f0),
    )
    phase = torch.cumsum(phase_increment, dim=0)
    previous = torch.cat((torch.zeros(1, dtype=phase.dtype), phase[:-1]), dim=0)
    crossings = torch.nonzero(torch.floor(phase) > torch.floor(previous), as_tuple=False).flatten()
    result: list[tuple[int, int, int, float]] = []
    if crossings.numel() < 2:
        return result
    frame_count = int(utterance.mel_frames)
    for left_tensor, right_tensor in zip(crossings[:-1], crossings[1:]):
        left = int(left_tensor)
        right = int(right_tensor)
        period = right - left
        if period < MIN_CYCLE_SAMPLES or period > MAX_CYCLE_SAMPLES:
            continue
        if float(sample_voiced[left:right].mean()) < 0.75:
            continue
        midpoint = (left + right - 1) // 2
        frame_index = min(frame_count - 1, max(0, int(round(midpoint / float(HOP_LENGTH)))))
        periodicity = float(sample_periodicity[left:right].mean().clamp(0.0, 1.0))
        result.append((left, right, frame_index, periodicity))
    return result


def _cycle_cache_path(root: Path, split: str, utterance_id: str) -> Path:
    return (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "pitch_synchronous_cycle_source_v1"
        / "targets"
        / split
        / f"{utterance_id}.pt"
    )


def load_or_build_cycle_targets(
    root: Path,
    utterance: OwnedVocoderUtterance,
) -> list[PitchSynchronousCycleTarget]:
    path = _cycle_cache_path(root, utterance.split, utterance.utterance_id)
    base_target = _load_or_build_target(root, utterance)
    boundaries = conditioning_cycle_boundaries(utterance)
    if path.exists():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if (
            isinstance(payload, dict)
            and payload.get("target_cache_version") == TARGET_CACHE_VERSION
            and payload.get("utterance_id") == utterance.utterance_id
            and payload.get("frame_count") == int(utterance.mel_frames)
            and payload.get("boundary_count") == len(boundaries)
        ):
            items = payload.get("cycles")
            if isinstance(items, list):
                return [
                    PitchSynchronousCycleTarget(
                        start_sample=int(item["start_sample"]),
                        end_sample=int(item["end_sample"]),
                        frame_index=int(item["frame_index"]),
                        periodicity=float(item["periodicity"]),
                        canonical_cycle=item["canonical_cycle"].to(torch.float32).contiguous(),
                    )
                    for item in items
                ]

    residual = base_target["residual"].to(torch.float32).contiguous()
    targets: list[PitchSynchronousCycleTarget] = []
    for left, right, frame_index, periodicity in boundaries:
        cycle = residual[left:right]
        if cycle.numel() < 2 or float(cycle.abs().max()) <= 1.0e-7:
            continue
        targets.append(
            PitchSynchronousCycleTarget(
                start_sample=left,
                end_sample=right,
                frame_index=frame_index,
                periodicity=periodicity,
                canonical_cycle=_canonicalize_cycle(cycle),
            )
        )
    payload: dict[str, Any] = {
        "target_cache_version": TARGET_CACHE_VERSION,
        "policy_id": POLICY_ID,
        "split": utterance.split,
        "utterance_id": utterance.utterance_id,
        "frame_count": int(utterance.mel_frames),
        "boundary_count": len(boundaries),
        "cycles": [
            {
                "start_sample": item.start_sample,
                "end_sample": item.end_sample,
                "frame_index": item.frame_index,
                "periodicity": item.periodicity,
                "canonical_cycle": item.canonical_cycle,
            }
            for item in targets
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return targets


def _target_log_rms(cycles: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(cycles.square().mean(dim=-1).clamp_min(EPSILON * EPSILON))
    return torch.log(rms)


def _cycle_loss_terms(
    prediction: torch.Tensor,
    predicted_log_rms: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_norm = torch.sqrt(prediction.square().sum(dim=-1).clamp_min(1.0e-10))
    target_norm = torch.sqrt(target.square().sum(dim=-1).clamp_min(1.0e-10))
    cosine = (prediction * target).sum(dim=-1) / (pred_norm * target_norm)
    shape = (1.0 - cosine.clamp(-1.0, 1.0)).mean()
    target_log_rms = _target_log_rms(target)
    level = F.smooth_l1_loss(predicted_log_rms, target_log_rms)
    scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-4)
    pointwise = ((prediction - target).abs() / scale).mean()
    derivative_scale = (target[:, 1:] - target[:, :-1]).abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-4)
    derivative = (
        ((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
        / derivative_scale
    ).mean()
    pred_spec = torch.fft.rfft(prediction, dim=-1).abs().clamp_min(1.0e-5)
    target_spec = torch.fft.rfft(target, dim=-1).abs().clamp_min(1.0e-5)
    spectral = F.l1_loss(torch.log(pred_spec), torch.log(target_spec))
    total = 0.75 * shape + 1.25 * level + 1.0 * pointwise + 0.5 * derivative + 0.35 * spectral
    public = {
        "total": float(total.detach()),
        "cycle_shape": float(shape.detach()),
        "cycle_log_rms": float(level.detach()),
        "cycle_relative_l1": float(pointwise.detach()),
        "cycle_derivative": float(derivative.detach()),
        "cycle_log_spectrum": float(spectral.detach()),
    }
    return total, public


def _copy_v2_context_encoder(root: Path, model: LykenoxPitchSynchronousResidualCycleSourceV1) -> str | None:
    checkpoint_path = root / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2" / "best.pt"
    if not checkpoint_path.exists():
        return None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("checkpoint_schema_version") != V2_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("V2 warm-start checkpoint schema mismatch")
    state = payload["model_state"]
    own = model.state_dict()
    copied = 0
    for key, value in state.items():
        if not (key.startswith("conditioning_projection.") or key.startswith("context_blocks.")):
            continue
        if key in own and own[key].shape == value.shape:
            own[key] = value.detach().clone()
            copied += 1
    if copied < 2:
        raise RuntimeError("V2 context warm start copied no meaningful parameters")
    model.load_state_dict(own)
    return str(checkpoint_path)


def _crop_cycle_sequence(
    cycles: list[PitchSynchronousCycleTarget],
    *,
    max_cycles: int,
    update: int,
    seed: int,
) -> list[PitchSynchronousCycleTarget]:
    if len(cycles) <= max_cycles:
        return cycles
    span = len(cycles) - max_cycles
    start = (int(seed) * 1000003 + int(update) * 9176 + 53) % (span + 1)
    return cycles[start : start + max_cycles]


def _training_tensors(
    utterance: OwnedVocoderUtterance,
    cycles: list[PitchSynchronousCycleTarget],
    *,
    context_margin_frames: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    if not cycles:
        raise ValueError("empty pitch-synchronous cycle sequence")
    min_frame = max(0, min(item.frame_index for item in cycles) - context_margin_frames)
    max_frame = min(int(utterance.mel_frames) - 1, max(item.frame_index for item in cycles) + context_margin_frames)
    end = max_frame + 1
    indices = torch.tensor([item.frame_index - min_frame for item in cycles], dtype=torch.long)
    target = torch.stack([item.canonical_cycle for item in cycles], dim=0).to(torch.float32)
    tensors = (
        utterance.mel[min_frame:end].unsqueeze(0).cpu(),
        utterance.f0_hz[min_frame:end].unsqueeze(0).cpu(),
        utterance.voiced[min_frame:end].unsqueeze(0).cpu(),
        utterance.periodicity[min_frame:end].unsqueeze(0).cpu(),
        indices,
    )
    return tensors, target


@torch.no_grad()
def _evaluate_complete(
    model: LykenoxPitchSynchronousResidualCycleSourceV1,
    root: Path,
    utterances: list[OwnedVocoderUtterance],
) -> dict[str, float]:
    model.eval()
    totals: list[dict[str, float]] = []
    for utterance in utterances:
        cycles = load_or_build_cycle_targets(root, utterance)
        if not cycles:
            continue
        tensors, target = _training_tensors(utterance, cycles, context_margin_frames=0)
        prediction, log_rms = model(*tensors)
        _, public = _cycle_loss_terms(prediction, log_rms, target)
        totals.append(public)
    if not totals:
        raise RuntimeError("no held-out pitch-synchronous cycles available")
    keys = tuple(totals[0])
    return {key: sum(item[key] for item in totals) / float(len(totals)) for key in keys}


def _save_checkpoint(
    path: Path,
    model: LykenoxPitchSynchronousResidualCycleSourceV1,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    best_val: float,
    config: dict[str, object],
) -> None:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE,
        "update": int(update),
        "best_val_total": float(best_val),
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_pitch_synchronous_cycle_source_v1(
    root: Path,
    *,
    train_items: int = DEFAULT_TRAIN_ITEMS,
    val_items: int = DEFAULT_VAL_ITEMS,
    max_updates: int = DEFAULT_MAX_UPDATES,
    cycles_per_update: int = DEFAULT_CYCLES_PER_UPDATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    resume: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if train_items < 1 or val_items < 1 or max_updates < 1 or cycles_per_update < 8:
        raise ValueError("invalid pitch-synchronous training limits")
    root = Path(root).resolve()
    torch.manual_seed(int(seed))
    run_dir = root / "models" / "lykenox_identity" / "training" / "pitch_synchronous_cycle_source_v1"
    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    train_set = collect_owned_vocoder_utterances(root, "train", max_items=train_items)
    val_set = collect_owned_vocoder_utterances(root, "val", max_items=val_items)
    cycle_counts = [len(load_or_build_cycle_targets(root, item)) for item in train_set]
    usable_indices = [index for index, count in enumerate(cycle_counts) if count >= 8]
    if not usable_indices:
        raise RuntimeError("no usable TRAIN pitch-synchronous cycle sequences")

    model = LykenoxPitchSynchronousResidualCycleSourceV1().cpu()
    warm_start = _copy_v2_context_encoder(root, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=DEFAULT_WEIGHT_DECAY)
    config: dict[str, object] = {
        "train_items": train_items,
        "val_items": val_items,
        "max_updates": max_updates,
        "cycles_per_update": cycles_per_update,
        "context_margin_frames": DEFAULT_CONTEXT_MARGIN_FRAMES,
        "learning_rate": learning_rate,
        "seed": seed,
        "cycle_phase_bins": CYCLE_PHASE_BINS,
        "v2_context_warm_start": warm_start,
        "teacher_forcing": False,
        "train_split_only_for_optimizer_updates": True,
    }
    start_update = 0
    best_val = math.inf
    if resume and latest.exists():
        try:
            payload = torch.load(latest, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(latest, map_location="cpu")
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("pitch-synchronous checkpoint schema mismatch")
        if payload.get("architecture") != PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE:
            raise RuntimeError("pitch-synchronous architecture mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_update = int(payload["update"])
        best_val = float(payload.get("best_val_total", math.inf))

    history = run_dir / "history.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    for update in range(start_update + 1, max_updates + 1):
        model.train()
        selected_index = usable_indices[(int(seed) + update * 7919) % len(usable_indices)]
        utterance = train_set[selected_index]
        all_cycles = load_or_build_cycle_targets(root, utterance)
        cycles = _crop_cycle_sequence(all_cycles, max_cycles=cycles_per_update, update=update, seed=seed + selected_index)
        tensors, target = _training_tensors(
            utterance,
            cycles,
            context_margin_frames=DEFAULT_CONTEXT_MARGIN_FRAMES,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction, log_rms = model(*tensors)
        loss, terms = _cycle_loss_terms(prediction, log_rms, target)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise RuntimeError("pitch-synchronous gradient norm became non-finite")
        optimizer.step()
        record: dict[str, object] = {
            "update": update,
            "utterance_id": utterance.utterance_id,
            "cycle_count": len(cycles),
            "teacher_forcing": False,
            "grad_norm": grad_norm,
            **terms,
        }
        if update % DEFAULT_EVAL_EVERY == 0 or update == max_updates:
            validation = _evaluate_complete(model, root, val_set)
            record["validation"] = validation
            if validation["total"] < best_val:
                best_val = validation["total"]
                _save_checkpoint(best, model, optimizer, update=update, best_val=best_val, config=config)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if update % DEFAULT_CHECKPOINT_EVERY == 0 or update == max_updates:
            _save_checkpoint(latest, model, optimizer, update=update, best_val=best_val, config=config)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report: dict[str, object] = {
        "status": "pitch_synchronous_cycle_source_training_complete",
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE,
        "updates": max_updates,
        "best_val_total": best_val,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "v2_context_warm_start": warm_start,
        "codebook_used": False,
        "teacher_forcing_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "next_action": "render_complete_heldout_hybrid_source_and_listen_for_robotic_timbre",
    }
    _atomic_json(run_dir / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--val-items", type=int, default=DEFAULT_VAL_ITEMS)
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--cycles-per-update", type=int, default=DEFAULT_CYCLES_PER_UPDATE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_pitch_synchronous_cycle_source_v1(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        max_updates=args.max_updates,
        cycles_per_update=args.cycles_per_update,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
