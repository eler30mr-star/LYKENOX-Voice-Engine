"""CPU-only bounded minimum-phase trainer with directional fixed-weight calibration.

Fresh runs first derive one positive fixed weight vector from owned gradient directions, then
verify that same vector on disjoint deterministic probes.  Only after verification passes is
a model optimizer created.  The weights are frozen into run config/checkpoints and are never
re-derived during training or exact resume.
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

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
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
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
    valid_context_multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_artifact_v2 import (
    CHECKPOINT_SCHEMA_VERSION,
    load_minimum_phase_checkpoint_v2,
    save_minimum_phase_checkpoint_v2,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_directional_weight_calibration import (
    CALIBRATION_VERSION,
    MIN_MEAN_AUTHORITY_SHARE,
    MAX_MEAN_AUTHORITY_SHARE,
    MIN_WORST_ALIGNMENT,
    OBJECTIVES,
    DirectionalFixedWeights,
    GradientGramProbe,
    calibrate_directional_fixed_weights,
    fixed_weights_from_mapping,
    gradient_gram,
    summarize_weights,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_noise import (
    NOISE_SEED_VERSION,
    stable_owned_noise_seed,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_objective_v3 import (
    ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
    OwnedMinimumPhaseObjectiveV3,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_owned_minimum_phase_vocoder_path,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_train_and_listen_contract_v2 import (
    CONTRACT_VERSION as TRAIN_AND_LISTEN_CONTRACT_VERSION,
    MAX_UPDATES_AUTHORIZED,
    require_authorized_run,
)
from lykenox_voice_engine.training.speech_vocoder_presence_v2 import target_relative_presence_loss_v2
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


TRAINER_VERSION = "owned-minimum-phase-resumable-trainer-v3-directional-fixed"
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
CALIBRATION_MEL_FRAMES = 32
CALIBRATION_ITEMS_PER_SPLIT = 2
CALIBRATION_STATES = ("neutral", "connected_probe")
CALIBRATION_CONNECTED_HEAD_SCALE = 1.0e-4
CALIBRATION_SEED_OFFSETS = {"train": 101000000, "val": 102000000}
VERIFICATION_SEED_OFFSETS = {"train": 103000000, "val": 104000000}


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


def _individual_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mel: torch.Tensor,
    envelope_objective: ConditioningAlignedLogMelEnvelopeLossV2,
) -> dict[str, torch.Tensor]:
    return {
        "reconstruction": valid_context_multi_resolution_reconstruction_loss(prediction, target).total,
        "envelope": envelope_objective(prediction, mel).total,
        "presence": target_relative_presence_loss_v2(
            prediction, target, sample_rate=SAMPLE_RATE
        ).loss,
        "spectral_balance": target_relative_spectral_balance_loss(
            prediction, target, sample_rate=SAMPLE_RATE
        ).loss,
    }


def _state_model(
    base_model: LykenoxFrameRateCepstralPredictorV1,
    state: str,
) -> LykenoxFrameRateCepstralPredictorV1:
    model = copy.deepcopy(base_model).cpu()
    if state == "neutral":
        return model
    if state != "connected_probe":
        raise ValueError(f"unknown calibration state: {state}")
    with torch.no_grad():
        index = torch.arange(model.cepstral_projection.weight.numel(), dtype=torch.float32)
        pattern = CALIBRATION_CONNECTED_HEAD_SCALE * torch.sin(index * math.sqrt(2.0))
        model.cepstral_projection.weight.copy_(pattern.view_as(model.cepstral_projection.weight))
    return model


def _collect_gram_probes(
    root: Path,
    base_model: LykenoxFrameRateCepstralPredictorV1,
    *,
    seed_offsets: dict[str, int],
    data_seed: int,
    base_noise_seed: int,
) -> tuple[list[GradientGramProbe], list[dict[str, object]]]:
    envelope_objective = ConditioningAlignedLogMelEnvelopeLossV2(LykenoxSpeechConfig()).cpu()
    segments: list[OwnedVocoderSegment] = []
    for split in ("train", "val"):
        selected, _ = collect_owned_vocoder_segments(
            root,
            split,
            segment_mel_frames=CALIBRATION_MEL_FRAMES,
            max_items=CALIBRATION_ITEMS_PER_SPLIT,
            seed=data_seed + seed_offsets[split],
        )
        segments.extend(selected)

    gram_probes: list[GradientGramProbe] = []
    public_probes: list[dict[str, object]] = []
    for state in CALIBRATION_STATES:
        for segment in segments:
            model = _state_model(base_model, state)
            parameters = tuple(model.parameters())
            prediction, excitation, tensors = _render_selected(
                model, [segment], base_noise_seed=base_noise_seed
            )
            mel, _, _, _, target, cepstrum = tensors
            terms = _individual_terms(prediction, target, mel, envelope_objective)
            cepstrum_gradients: dict[str, torch.Tensor] = {}
            parameter_gradients: dict[str, torch.Tensor] = {}
            for name in OBJECTIVES:
                cepstrum_gradients[name] = torch.autograd.grad(
                    terms[name], cepstrum, retain_graph=True, allow_unused=False
                )[0].detach()
                parameter_gradients[name] = torch.cat(
                    [
                        value.reshape(-1)
                        for value in torch.autograd.grad(
                            terms[name], parameters, retain_graph=True, allow_unused=False
                        )
                    ]
                ).detach()
            cep_gram = gradient_gram(cepstrum_gradients)
            param_gram = gradient_gram(parameter_gradients)
            prefix = f"{segment.split}:{segment.utterance_id}:{segment.start_frame}:{state}"
            gram_probes.extend(
                [
                    GradientGramProbe(prefix + ":cepstrum", "cepstrum_space", state, cep_gram),
                    GradientGramProbe(prefix + ":parameter", "parameter_space", state, param_gram),
                ]
            )
            grid = frame_grid_artifact_excess_metrics(
                prediction,
                excitation,
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
            )
            public_probes.append(
                {
                    "split": segment.split,
                    "utterance_id": segment.utterance_id,
                    "start_frame": segment.start_frame,
                    "state": state,
                    "exact_output_length": bool(
                        prediction.shape[-1] == segment.mel_frames * HOP_LENGTH
                    ),
                    "severe_grid_excess": bool(grid.severe_grid_excess.any()),
                }
            )
    return gram_probes, public_probes


def run_directional_calibration(
    root: Path,
    base_model: LykenoxFrameRateCepstralPredictorV1,
    *,
    data_seed: int,
    base_noise_seed: int,
) -> tuple[DirectionalFixedWeights | None, dict[str, object]]:
    calibration_grams, calibration_public = _collect_gram_probes(
        root,
        base_model,
        seed_offsets=CALIBRATION_SEED_OFFSETS,
        data_seed=data_seed,
        base_noise_seed=base_noise_seed,
    )
    weights, calibration = calibrate_directional_fixed_weights(calibration_grams)
    if weights is None:
        return None, {
            "status": "fail",
            "calibration_version": CALIBRATION_VERSION,
            "calibration": calibration,
            "verification": None,
            "optimizer_created": False,
            "parameter_update_executed": False,
        }

    verification_grams, verification_public = _collect_gram_probes(
        root,
        base_model,
        seed_offsets=VERIFICATION_SEED_OFFSETS,
        data_seed=data_seed,
        base_noise_seed=base_noise_seed,
    )
    verification_summary = summarize_weights(weights, verification_grams)
    verification_gates = {
        "worst_alignment_positive": float(verification_summary["worst_alignment"])
        >= MIN_WORST_ALIGNMENT,
        "all_first_order_descent_positive": float(verification_summary["minimum_descent_dot"]) > 0.0,
        "authority_floor_retained": float(
            verification_summary["minimum_mean_weighted_gradient_norm_share"]
        )
        >= MIN_MEAN_AUTHORITY_SHARE,
        "dominance_bounded": float(
            verification_summary["maximum_mean_weighted_gradient_norm_share"]
        )
        <= MAX_MEAN_AUTHORITY_SHARE,
        "exact_output_length_calibration": all(
            bool(item["exact_output_length"]) for item in calibration_public
        ),
        "no_severe_grid_calibration": not any(
            bool(item["severe_grid_excess"]) for item in calibration_public
        ),
        "exact_output_length_verification": all(
            bool(item["exact_output_length"]) for item in verification_public
        ),
        "no_severe_grid_verification": not any(
            bool(item["severe_grid_excess"]) for item in verification_public
        ),
    }
    passed = all(verification_gates.values())
    report = {
        "status": "pass" if passed else "fail",
        "calibration_version": CALIBRATION_VERSION,
        "calibrated_weights": weights.as_dict(),
        "calibration": calibration,
        "verification": verification_summary,
        "verification_gates": verification_gates,
        "calibration_model_states": list(CALIBRATION_STATES),
        "calibration_gradient_probe_count": len(calibration_grams),
        "verification_gradient_probe_count": len(verification_grams),
        "optimizer_created": False,
        "parameter_update_executed": False,
        "adaptive_during_training": False,
    }
    return (weights if passed else None), report


def _run_config(
    *,
    weights: DirectionalFixedWeights,
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
        "train_and_listen_contract_version": TRAIN_AND_LISTEN_CONTRACT_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "predictor_architecture": PREDICTOR_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "objective_version": ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "data_contract_version": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "noise_seed_version": NOISE_SEED_VERSION,
        "calibrated_weights": weights.as_dict(),
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


def _forward_selected(
    model: LykenoxFrameRateCepstralPredictorV1,
    objective: OwnedMinimumPhaseObjectiveV3,
    selected: list[OwnedVocoderSegment],
    *,
    base_noise_seed: int,
) -> tuple[Any, torch.Tensor]:
    prediction, _, tensors = _render_selected(model, selected, base_noise_seed=base_noise_seed)
    mel, _, _, _, target, _ = tensors
    return objective(prediction, target, mel), prediction


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
    objective: OwnedMinimumPhaseObjectiveV3,
    segments: list[OwnedVocoderSegment],
    *,
    batch_size: int,
    base_noise_seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {name: 0.0 for name in ("total",) + OBJECTIVES}
    count = 0
    with torch.no_grad():
        for offset in range(0, len(segments), batch_size):
            indices = list(range(offset, min(offset + batch_size, len(segments))))
            selected = _select(segments, indices)
            row = _metric_row(
                _forward_selected(
                    model, objective, selected, base_noise_seed=base_noise_seed
                )[0]
            )
            count += len(selected)
            for name, value in row.items():
                totals[name] += value * len(selected)
    if was_training:
        model.train()
    if count < 1:
        raise RuntimeError("minimum-phase validation set is empty")
    return {name: value / count for name, value in totals.items()}


def run_minimum_phase_training_v3(
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

    root = Path(root).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "training" / "vocoder_minimum_phase_v3"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    report_path = output_dir / "training_report.json"
    progress_path = output_dir / "training_progress.json"
    calibration_path = output_dir / "directional_calibration_report.json"
    protected = _protected_checkpoints(root)
    protected_before = {name: _sha256(path) for name, path in protected.items()}

    if last_path.exists():
        model, payload = load_minimum_phase_checkpoint_v2(last_path)
        run_config = dict(payload["run_config"])
        weights = fixed_weights_from_mapping(run_config["calibrated_weights"])
        expected = _run_config(
            weights=weights,
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
        if run_config != expected:
            raise RuntimeError("Refusing exact resume with changed V3 run configuration")
        progress = dict(payload["progress"])
        calibration_report = progress.get("directional_calibration")
        if not isinstance(calibration_report, dict) or calibration_report.get("status") != "pass":
            raise RuntimeError("Refusing resume without passed directional calibration")
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
        weights, calibration_report = run_directional_calibration(
            root,
            model,
            data_seed=data_seed,
            base_noise_seed=noise_seed,
        )
        _atomic_json(calibration_path, calibration_report)
        if weights is None:
            report = {
                "status": "blocked_by_directional_weight_calibration",
                "trainer_version": TRAINER_VERSION,
                "device": "cpu",
                "directional_calibration": calibration_report,
                "optimizer_created": False,
                "parameter_update_executed": False,
                "checkpoint_saved": False,
                "next_action": "objective_set_has_no_verified_static_common_descent_weights",
            }
            _atomic_json(report_path, report)
            return report
        run_config = _run_config(
            weights=weights,
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        history: list[dict[str, object]] = []
        epoch = 1
        next_item_offset = 0
        global_step = 0
        best_step = 0
        clipped_update_count = 0
        initial_validation = {}
        best_val_total = float("inf")

    objective = OwnedMinimumPhaseObjectiveV3(weights).cpu()
    val_segments, _ = collect_owned_vocoder_segments(
        root,
        "val",
        segment_mel_frames=segment_mel_frames,
        max_items=val_items,
        seed=data_seed + 50000000,
    )
    if not initial_validation:
        initial_validation = _evaluate(
            model,
            objective,
            val_segments,
            batch_size=batch_size,
            base_noise_seed=noise_seed,
        )
        best_val_total = float(initial_validation["total"])

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
            "directional_calibration": calibration_report,
        }

    def save_last() -> None:
        save_minimum_phase_checkpoint_v2(
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
            raise RuntimeError("minimum-phase V3 checkpoint has invalid item offset")
        while next_item_offset < len(order) and global_step < max_updates:
            if max_updates_this_run is not None and updates_this_run >= max_updates_this_run:
                save_last()
                report = {
                    "status": "incomplete",
                    "stop_reason": "max_updates_this_run_reached",
                    "trainer_version": TRAINER_VERSION,
                    "device": "cpu",
                    "global_step": global_step,
                    "calibrated_weights": weights.as_dict(),
                    "last_checkpoint": str(last_path),
                    "best_checkpoint": str(best_path) if best_path.exists() else None,
                }
                _atomic_json(progress_path, report)
                return report

            indices = order[next_item_offset : min(next_item_offset + batch_size, len(order))]
            selected = _select(train_segments, indices)
            optimizer.zero_grad(set_to_none=True)
            losses, prediction = _forward_selected(
                model, objective, selected, base_noise_seed=noise_seed
            )
            if not torch.isfinite(losses.total):
                raise RuntimeError(f"non-finite minimum-phase V3 loss at step {global_step}")
            losses.total.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            if any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise RuntimeError(f"missing/non-finite minimum-phase gradient at step {global_step}")
            raw_grad_norm = float(
                torch.sqrt(sum(gradient.detach().square().sum() for gradient in gradients))
            )
            if not math.isfinite(raw_grad_norm) or raw_grad_norm <= 0.0:
                raise RuntimeError(f"invalid minimum-phase gradient at step {global_step}")
            if raw_grad_norm > grad_clip:
                clipped_update_count += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                    save_minimum_phase_checkpoint_v2(
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
        "calibration_version": CALIBRATION_VERSION,
        "calibrated_weights": weights.as_dict(),
        "global_step": global_step,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "best_val_total": best_val_total,
        "best_step": best_step,
        "validation_improved": validation_improved,
        "clipped_update_count": clipped_update_count,
        "clipped_update_fraction": clipped_update_count / max(global_step, 1),
        "directional_calibration": calibration_report,
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
            run_minimum_phase_training_v3(
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
