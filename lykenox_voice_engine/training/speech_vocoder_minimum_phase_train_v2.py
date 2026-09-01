"""CPU-only bounded trainer for the owned minimum-phase vocoder candidate.

The trainer is scoped by LYX-POL-001.  It refuses to create an optimizer until a read-only
architecture-coupled V2 authority preflight passes on deterministic owned train/val segments.
It uses the active minimum-phase objective V2, varies deterministic aperiodic-noise realization
per owned utterance/crop, resumes exactly from ``last.pt`` and selects ``best.pt`` only by the
fixed validation objective.  Metrics cannot accept voice quality; complete held-out audio is
required after training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    frame_grid_artifact_excess_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_artifact import (
    CHECKPOINT_SCHEMA_VERSION,
    load_minimum_phase_checkpoint,
    save_minimum_phase_checkpoint,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_noise import (
    NOISE_SEED_VERSION,
    stable_owned_noise_seed,
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
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_train_and_listen_contract import (
    CONTRACT_VERSION as TRAIN_AND_LISTEN_CONTRACT_VERSION,
    MAX_UPDATES_AUTHORIZED,
    require_authorized_run,
)


TRAINER_VERSION = "owned-minimum-phase-resumable-trainer-v2-integrated-authority-preflight"
PREFLIGHT_VERSION = "owned-minimum-phase-v2-authority-preflight-v1"
TRAIN_ORDER_VERSION = "epoch-permutation-v1"
DEFAULT_SEGMENT_MEL_FRAMES = 64
DEFAULT_TRAIN_ITEMS = 48
DEFAULT_VAL_ITEMS = 12
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_UPDATES = MAX_UPDATES_AUTHORIZED
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_EVAL_EVERY = 25
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_DATA_SEED = 20260901
DEFAULT_MODEL_SEED = 20260903
DEFAULT_ORDER_SEED = 20260905
DEFAULT_NOISE_SEED = 97
PREFLIGHT_ITEMS_PER_SPLIT = 2
PREFLIGHT_CONNECTED_HEAD_SCALE = 1.0e-4
PREFLIGHT_MIN_MEAN_WEIGHTED_SHARE = 0.05
PREFLIGHT_MAX_MEAN_WEIGHTED_SHARE = 0.65
PREFLIGHT_MAX_LINEARITY_RELATIVE_ERROR = 1.0e-5
OBJECTIVES = ("reconstruction", "envelope", "presence", "spectral_balance")
STATES = ("neutral", "connected_probe")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_checkpoints(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
    }


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
        "preflight_version": PREFLIGHT_VERSION,
        "train_and_listen_contract_version": TRAIN_AND_LISTEN_CONTRACT_VERSION,
        "train_order_version": TRAIN_ORDER_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "predictor_architecture": PREDICTOR_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "objective_version": ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
        "loss_weight_contract_version": ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION,
        "data_contract_version": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "noise_seed_version": NOISE_SEED_VERSION,
        "active_weights": active_weights(),
        "device": "cpu",
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


def _select(segments: list[OwnedVocoderSegment], indices: list[int]) -> list[OwnedVocoderSegment]:
    selected = [segments[index] for index in indices]
    if not selected:
        raise ValueError("empty minimum-phase batch")
    for segment in selected:
        if segment.conditioning_contract_version != OWNED_VOCODER_SEGMENT_CONTRACT_VERSION:
            raise RuntimeError("minimum-phase trainer received the wrong data contract")
    return selected


def _batch_tensors(selected: list[OwnedVocoderSegment]) -> tuple[torch.Tensor, ...]:
    return (
        torch.stack([item.mel for item in selected], dim=0).cpu(),
        torch.stack([item.f0_hz for item in selected], dim=0).cpu(),
        torch.stack([item.voiced for item in selected], dim=0).cpu(),
        torch.stack([item.periodicity for item in selected], dim=0).cpu(),
        torch.stack([item.waveform for item in selected], dim=0).cpu(),
    )


def _render_selected(
    model: LykenoxFrameRateCepstralPredictorV1,
    selected: list[OwnedVocoderSegment],
    *,
    base_noise_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    mel, f0_hz, voiced, periodicity, target = _batch_tensors(selected)
    cepstrum = model(mel, f0_hz, voiced, periodicity)
    predictions: list[torch.Tensor] = []
    excitations: list[torch.Tensor] = []
    for index, segment in enumerate(selected):
        seed = stable_owned_noise_seed(
            base_noise_seed,
            split=segment.split,
            utterance_id=segment.utterance_id,
            start_frame=segment.start_frame,
        )
        prediction, excitation = render_owned_minimum_phase_vocoder_path(
            cepstrum[index : index + 1],
            f0_hz[index : index + 1],
            voiced[index : index + 1],
            periodicity[index : index + 1],
            noise_seed=seed,
        )
        predictions.append(prediction)
        excitations.append(excitation)
    prediction = torch.cat(predictions, dim=0)
    excitation = torch.cat(excitations, dim=0)
    expected_samples = mel.shape[1] * HOP_LENGTH
    if prediction.shape[-1] != expected_samples or target.shape[-1] != expected_samples:
        raise RuntimeError("minimum-phase trainer violated exact output-length contract")
    return prediction, excitation, (mel, f0_hz, voiced, periodicity, target, cepstrum)


def _forward_selected(
    model: LykenoxFrameRateCepstralPredictorV1,
    objective: OwnedMinimumPhaseObjectiveV2,
    selected: list[OwnedVocoderSegment],
    *,
    base_noise_seed: int,
) -> tuple[Any, torch.Tensor]:
    prediction, _, tensors = _render_selected(model, selected, base_noise_seed=base_noise_seed)
    mel, _, _, _, target, _ = tensors
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


def _evaluate(
    model: LykenoxFrameRateCepstralPredictorV1,
    objective: OwnedMinimumPhaseObjectiveV2,
    segments: list[OwnedVocoderSegment],
    *,
    batch_size: int,
    base_noise_seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {name: 0.0 for name in ("total",) + OBJECTIVES}
    item_count = 0
    with torch.no_grad():
        for offset in range(0, len(segments), batch_size):
            indices = list(range(offset, min(offset + batch_size, len(segments))))
            selected = _select(segments, indices)
            row = _metric_row(
                _forward_selected(
                    model,
                    objective,
                    selected,
                    base_noise_seed=base_noise_seed,
                )[0]
            )
            weight = len(selected)
            item_count += weight
            for name, value in row.items():
                totals[name] += value * weight
    if was_training:
        model.train()
    if item_count < 1:
        raise RuntimeError("minimum-phase validation set is empty")
    return {name: value / item_count for name, value in totals.items()}


def _flatten(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach()))


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = (torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)).clamp_min(1e-12)
    return float(torch.sum(first * second) / denominator)


def _space_metrics(
    gradients: dict[str, torch.Tensor],
    *,
    direct_combined: torch.Tensor,
) -> dict[str, object]:
    weights = active_weights()
    weighted = {name: gradients[name] * float(weights[name]) for name in OBJECTIVES}
    weighted_norms = {name: _norm(weighted[name]) for name in OBJECTIVES}
    norm_sum = max(sum(weighted_norms.values()), 1e-12)
    shares = {name: weighted_norms[name] / norm_sum for name in OBJECTIVES}
    combined = sum(weighted.values())
    denominator = torch.linalg.vector_norm(direct_combined).clamp_min(1e-12)
    linearity_error = float(torch.linalg.vector_norm(direct_combined - combined) / denominator)
    return {
        "raw_gradient_norms": {name: _norm(gradients[name]) for name in OBJECTIVES},
        "weighted_gradient_norm_shares": shares,
        "combined_gradient_alignment_cosines": {
            name: _cosine(gradients[name], combined) for name in OBJECTIVES
        },
        "first_order_descent_dots": {
            name: float(torch.sum(gradients[name] * combined)) for name in OBJECTIVES
        },
        "combined_gradient_norm": _norm(combined),
        "maximum_weighted_gradient_norm_share": max(shares.values()),
        "combined_gradient_linearity_relative_error": linearity_error,
        "all_objective_gradients_finite_nonzero": all(
            bool(torch.isfinite(gradients[name]).all()) and _norm(gradients[name]) > 0.0
            for name in OBJECTIVES
        ),
    }


def _state_model(
    model: LykenoxFrameRateCepstralPredictorV1,
    state: str,
) -> LykenoxFrameRateCepstralPredictorV1:
    if state == "neutral":
        return copy.deepcopy(model).cpu()
    if state != "connected_probe":
        raise ValueError(f"unknown preflight state: {state}")
    probe = copy.deepcopy(model).cpu()
    with torch.no_grad():
        index = torch.arange(probe.cepstral_projection.weight.numel(), dtype=torch.float32)
        pattern = PREFLIGHT_CONNECTED_HEAD_SCALE * torch.sin(index * math.sqrt(2.0))
        probe.cepstral_projection.weight.copy_(pattern.view_as(probe.cepstral_projection.weight))
    return probe


def _preflight_probe(
    model: LykenoxFrameRateCepstralPredictorV1,
    objective: OwnedMinimumPhaseObjectiveV2,
    segment: OwnedVocoderSegment,
    *,
    state: str,
    base_noise_seed: int,
) -> dict[str, object]:
    state_model = _state_model(model, state)
    parameters = tuple(state_model.parameters())
    prediction, excitation, tensors = _render_selected(
        state_model,
        [segment],
        base_noise_seed=base_noise_seed,
    )
    mel, _, _, _, target, cepstrum = tensors
    result = objective(prediction, target, mel)
    terms = {
        "reconstruction": result.reconstruction,
        "envelope": result.envelope,
        "presence": result.presence,
        "spectral_balance": result.spectral_balance,
    }
    cepstrum_gradients: dict[str, torch.Tensor] = {}
    parameter_gradients: dict[str, torch.Tensor] = {}
    for name in OBJECTIVES:
        cepstrum_gradients[name] = torch.autograd.grad(
            terms[name], cepstrum, retain_graph=True, allow_unused=False
        )[0].detach()
        parameter_gradients[name] = _flatten(
            torch.autograd.grad(
                terms[name], parameters, retain_graph=True, allow_unused=False
            )
        ).detach()
    direct_parameter = _flatten(
        torch.autograd.grad(result.total, parameters, retain_graph=True, allow_unused=False)
    ).detach()
    direct_cepstrum = torch.autograd.grad(
        result.total, cepstrum, retain_graph=False, allow_unused=False
    )[0].detach()
    grid = frame_grid_artifact_excess_metrics(
        prediction,
        excitation,
        sample_rate=24000,
        hop_length=HOP_LENGTH,
    )
    return {
        "split": segment.split,
        "utterance_id": segment.utterance_id,
        "start_frame": segment.start_frame,
        "state": state,
        "cepstrum_space": _space_metrics(
            cepstrum_gradients, direct_combined=direct_cepstrum
        ),
        "parameter_space": _space_metrics(
            parameter_gradients, direct_combined=direct_parameter
        ),
        "exact_output_length": prediction.shape[-1] == segment.mel_frames * HOP_LENGTH,
        "severe_grid_excess": bool(grid.severe_grid_excess.any()),
    }


def _summarize_preflight(probes: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for space in ("cepstrum_space", "parameter_space"):
        summary[space] = {}
        for state in STATES:
            state_metrics = [probe[space] for probe in probes if probe["state"] == state]
            shares = {
                name: sum(float(metric["weighted_gradient_norm_shares"][name]) for metric in state_metrics)
                / len(state_metrics)
                for name in OBJECTIVES
            }
            alignments = {
                name: min(float(metric["combined_gradient_alignment_cosines"][name]) for metric in state_metrics)
                for name in OBJECTIVES
            }
            descent = {
                name: min(float(metric["first_order_descent_dots"][name]) for metric in state_metrics)
                for name in OBJECTIVES
            }
            summary[space][state] = {
                "mean_weighted_gradient_norm_shares": shares,
                "minimum_combined_gradient_alignment_cosines": alignments,
                "minimum_first_order_descent_dots": descent,
                "maximum_mean_weighted_gradient_norm_share": max(shares.values()),
                "minimum_mean_weighted_gradient_norm_share": min(shares.values()),
                "maximum_linearity_relative_error": max(
                    float(metric["combined_gradient_linearity_relative_error"])
                    for metric in state_metrics
                ),
                "all_objective_gradients_finite_nonzero": all(
                    bool(metric["all_objective_gradients_finite_nonzero"])
                    for metric in state_metrics
                ),
            }
    return summary


def run_v2_authority_preflight(
    root: Path,
    model: LykenoxFrameRateCepstralPredictorV1,
    *,
    segment_mel_frames: int,
    data_seed: int,
    base_noise_seed: int,
) -> dict[str, object]:
    root = Path(root).resolve()
    objective = OwnedMinimumPhaseObjectiveV2().cpu()
    segments: list[OwnedVocoderSegment] = []
    for split, seed_offset in (("train", 91000000), ("val", 92000000)):
        selected, _ = collect_owned_vocoder_segments(
            root,
            split,
            segment_mel_frames=segment_mel_frames,
            max_items=PREFLIGHT_ITEMS_PER_SPLIT,
            seed=data_seed + seed_offset,
        )
        segments.extend(selected)
    probes = [
        _preflight_probe(
            model,
            objective,
            segment,
            state=state,
            base_noise_seed=base_noise_seed,
        )
        for state in STATES
        for segment in segments
    ]
    summary = _summarize_preflight(probes)
    all_summaries = [summary[space][state] for space in summary for state in STATES]
    gates = {
        "all_objective_gradients_finite_nonzero": all(
            bool(item["all_objective_gradients_finite_nonzero"]) for item in all_summaries
        ),
        "combined_gradient_linearity_exact": all(
            float(item["maximum_linearity_relative_error"])
            <= PREFLIGHT_MAX_LINEARITY_RELATIVE_ERROR
            for item in all_summaries
        ),
        "all_combined_alignments_positive": all(
            min(float(value) for value in item["minimum_combined_gradient_alignment_cosines"].values())
            > 0.0
            for item in all_summaries
        ),
        "all_first_order_descent_dots_positive": all(
            min(float(value) for value in item["minimum_first_order_descent_dots"].values()) > 0.0
            for item in all_summaries
        ),
        "mean_authority_floor_retained": all(
            float(item["minimum_mean_weighted_gradient_norm_share"])
            >= PREFLIGHT_MIN_MEAN_WEIGHTED_SHARE
            for item in all_summaries
        ),
        "mean_dominance_bounded": all(
            float(item["maximum_mean_weighted_gradient_norm_share"])
            <= PREFLIGHT_MAX_MEAN_WEIGHTED_SHARE
            for item in all_summaries
        ),
        "exact_output_length_all_probes": all(bool(probe["exact_output_length"]) for probe in probes),
        "no_severe_grid_excess_all_probes": not any(bool(probe["severe_grid_excess"]) for probe in probes),
    }
    return {
        "status": "pass" if all(gates.values()) else "fail",
        "preflight_version": PREFLIGHT_VERSION,
        "probe_count": len(probes),
        "active_weights": active_weights(),
        "summary": summary,
        "gates": gates,
        "optimizer_created": False,
        "parameter_update_executed": False,
    }


def run_minimum_phase_training_v2(
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
    require_authorized_run(max_updates)
    if segment_mel_frames < 16 or train_items < 2 or val_items < 1 or batch_size < 1:
        raise ValueError("invalid minimum-phase data configuration")
    if max_updates_this_run is not None and max_updates_this_run < 1:
        raise ValueError("max_updates_this_run must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0 or grad_clip <= 0.0:
        raise ValueError("invalid optimizer configuration")
    if eval_every < 1 or checkpoint_every < 1:
        raise ValueError("eval/checkpoint cadence must be positive")

    root = Path(root).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "training" / "vocoder_minimum_phase_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    progress_path = output_dir / "training_progress.json"
    report_path = output_dir / "training_report.json"
    if not last_path.exists() and best_path.exists():
        raise RuntimeError("orphan minimum-phase v2 best.pt exists without last.pt")

    protected = _protected_checkpoints(root)
    protected_before = {name: _sha256(path) for name, path in protected.items()}
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
        model, payload = load_minimum_phase_checkpoint(last_path, expected_run_config=run_config)
        progress = dict(payload["progress"])
        preflight = progress.get("v2_authority_preflight")
        if not isinstance(preflight, dict) or preflight.get("status") != "pass":
            raise RuntimeError("refusing resume without a passed V2 authority preflight")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
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
        preflight = run_v2_authority_preflight(
            root,
            model,
            segment_mel_frames=segment_mel_frames,
            data_seed=data_seed,
            base_noise_seed=noise_seed,
        )
        if preflight["status"] != "pass":
            report = {
                "status": "blocked_by_v2_authority_preflight",
                "trainer_version": TRAINER_VERSION,
                "device": "cpu",
                "v2_authority_preflight": preflight,
                "optimizer_created": False,
                "parameter_update_executed": False,
                "checkpoint_saved": False,
                "next_action": "fix_loss_authority_before_training",
            }
            _atomic_json(report_path, report)
            return report
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        initial_validation = _evaluate(
            model,
            objective,
            val_segments,
            batch_size=batch_size,
            base_noise_seed=noise_seed,
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
            "v2_authority_preflight": preflight,
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
            raise RuntimeError("minimum-phase v2 checkpoint has invalid item offset")
        while next_item_offset < len(order) and global_step < max_updates:
            if max_updates_this_run is not None and updates_this_run >= max_updates_this_run:
                save_last()
                report = {
                    "status": "incomplete",
                    "stop_reason": "max_updates_this_run_reached",
                    "trainer_version": TRAINER_VERSION,
                    "device": "cpu",
                    "global_step": global_step,
                    "epoch": epoch,
                    "next_item_offset": next_item_offset,
                    "updates_this_run": updates_this_run,
                    "initial_validation": initial_validation,
                    "best_val_total": best_val_total,
                    "best_step": best_step,
                    "last_checkpoint": str(last_path),
                    "best_checkpoint": str(best_path) if best_path.exists() else None,
                    "v2_authority_preflight": preflight,
                    "active_weights": active_weights(),
                }
                _atomic_json(progress_path, report)
                return report

            indices = order[next_item_offset : min(next_item_offset + batch_size, len(order))]
            selected = _select(train_segments, indices)
            optimizer.zero_grad(set_to_none=True)
            losses, prediction = _forward_selected(
                model,
                objective,
                selected,
                base_noise_seed=noise_seed,
            )
            if not torch.isfinite(losses.total):
                raise RuntimeError(f"non-finite minimum-phase V2 loss at step {global_step}")
            losses.total.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            if any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise RuntimeError(f"missing/non-finite minimum-phase gradient at step {global_step}")
            raw_grad_norm = float(torch.sqrt(sum(gradient.detach().square().sum() for gradient in gradients)))
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
            next_item_offset += len(indices)
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "train": _metric_row(losses),
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
                    base_noise_seed=noise_seed,
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
        base_noise_seed=noise_seed,
    )
    protected_after = {name: _sha256(path) for name, path in protected.items()}
    protected_unchanged = protected_before == protected_after
    validation_improved = best_step > 0 and best_val_total < float(initial_validation["total"])
    status = "pass" if validation_improved and best_path.exists() and protected_unchanged else "needs_review"
    report = {
        "status": status,
        "stop_reason": "max_updates_reached",
        "trainer_version": TRAINER_VERSION,
        "train_and_listen_contract_version": TRAIN_AND_LISTEN_CONTRACT_VERSION,
        "device": "cpu",
        "objective_version": ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
        "loss_weight_contract_version": ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION,
        "noise_seed_version": NOISE_SEED_VERSION,
        "active_weights": active_weights(),
        "global_step": global_step,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "best_val_total": best_val_total,
        "best_step": best_step,
        "validation_improved": validation_improved,
        "clipped_update_count": clipped_update_count,
        "clipped_update_fraction": clipped_update_count / max(global_step, 1),
        "v2_authority_preflight": preflight,
        "protected_checkpoints_unchanged": protected_unchanged,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "third_party_model_used": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_accept_voice_quality": False,
        "full_held_out_audio_required": True,
        "next_action": "render_complete_val_utterances_from_best_checkpoint" if status == "pass" else "review_training_failure",
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
            run_minimum_phase_training_v2(
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
