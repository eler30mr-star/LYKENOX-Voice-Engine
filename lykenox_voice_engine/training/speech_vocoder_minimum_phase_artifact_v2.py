"""Exact-resume checkpoint for the directional fixed-weight minimum-phase run."""

from __future__ import annotations

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
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_directional_weight_calibration import (
    CALIBRATION_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_objective_v3 import (
    ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import RENDERER_VERSION


CHECKPOINT_SCHEMA_VERSION = "owned-minimum-phase-vocoder-checkpoint-v2-directional-fixed"


def checkpoint_contract() -> dict[str, str]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "predictor_architecture": PREDICTOR_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "objective_version": ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "data_contract_version": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    }


def _validate_contract(payload: dict[str, Any]) -> None:
    if payload.get("contract") != checkpoint_contract():
        raise RuntimeError("Refusing minimum-phase checkpoint with incompatible V3 contract")


def save_minimum_phase_checkpoint_v2(
    path: Path,
    *,
    model: LykenoxFrameRateCepstralPredictorV1,
    optimizer: torch.optim.Optimizer,
    run_config: dict[str, object],
    progress: dict[str, object],
    history: list[dict[str, object]],
) -> None:
    calibrated_weights = run_config.get("calibrated_weights")
    if not isinstance(calibrated_weights, dict) or len(calibrated_weights) != 4:
        raise RuntimeError("directional checkpoint requires frozen calibrated_weights in run_config")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "contract": checkpoint_contract(),
        "run_config": dict(run_config),
        "progress": dict(progress),
        "history": list(history),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_minimum_phase_checkpoint_v2(
    path: Path,
    *,
    expected_run_config: dict[str, object] | None = None,
) -> tuple[LykenoxFrameRateCepstralPredictorV1, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("minimum-phase V3 checkpoint payload must be a dictionary")
    _validate_contract(payload)
    if expected_run_config is not None and payload.get("run_config") != expected_run_config:
        raise RuntimeError("Refusing resume with changed directional weights or run config")
    if not isinstance(payload.get("progress"), dict):
        raise RuntimeError("minimum-phase V3 checkpoint is missing progress state")
    if not isinstance(payload.get("history"), list):
        raise RuntimeError("minimum-phase V3 checkpoint is missing training history")
    if "optimizer_state" not in payload or "torch_rng_state" not in payload:
        raise RuntimeError("minimum-phase V3 checkpoint is missing exact-resume state")
    run_config = payload.get("run_config")
    if not isinstance(run_config, dict) or not isinstance(run_config.get("calibrated_weights"), dict):
        raise RuntimeError("minimum-phase V3 checkpoint is missing frozen calibrated weights")
    model = LykenoxFrameRateCepstralPredictorV1().cpu()
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "checkpoint_contract",
    "save_minimum_phase_checkpoint_v2",
    "load_minimum_phase_checkpoint_v2",
]
