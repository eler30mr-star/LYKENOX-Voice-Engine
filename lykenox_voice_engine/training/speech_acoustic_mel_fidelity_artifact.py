"""Checkpoint contract for isolated acoustic mel-decoder fidelity refinement.

The one-epoch ``mel_decoder``-only candidate is retained for forensic reproducibility but
has been perceptually rejected: it did not produce a useful audible improvement over the
accepted acoustic-v2 baseline. New persistent runs through its default output path are
therefore blocked. Temporary explicit output directories remain available to historical
exact-resume tests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
    save_acoustic_prosody_checkpoint,
)


ISOLATED_MEL_ARTIFACT_VERSION = "acoustic-mel-fidelity-isolated-artifact-v1"
TRAINABLE_PREFIX = "mel_decoder."
ISOLATED_MEL_PERCEPTUALLY_REJECTED = True
ISOLATED_MEL_PERSISTENT_TRAINING_ENABLED = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_checkpoint_path(root: Path) -> Path:
    return (
        Path(root).resolve()
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_frame_context_v2"
        / "best.pt"
    )


def rejected_isolated_output_dir(root: Path) -> Path:
    """Return the historical artifact location without authorizing new training."""
    return (
        Path(root).resolve()
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_mel_fidelity_v1"
    )


def isolated_output_dir(root: Path) -> Path:
    del root
    raise RuntimeError(
        "acoustic_mel_fidelity_v1 was perceptually rejected: persistent mel_decoder-only "
        "training is disabled; keep existing checkpoints for forensic A/B only"
    )


def freeze_except_mel_decoder(model: LykenoxSpeechAcousticModel) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mel_decoder.parameters():
        parameter.requires_grad_(True)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any(not name.startswith(TRAINABLE_PREFIX) for name in names):
        raise RuntimeError("isolated mel refinement may train only mel_decoder parameters")
    return names


def require_frozen_state_exact(
    model: LykenoxSpeechAcousticModel,
    base_model: LykenoxSpeechAcousticModel,
) -> None:
    current = model.state_dict()
    base = base_model.state_dict()
    if current.keys() != base.keys():
        raise RuntimeError("isolated mel checkpoint/base model state keys differ")
    changed = [
        name
        for name in current
        if not name.startswith(TRAINABLE_PREFIX) and not torch.equal(current[name], base[name])
    ]
    if changed:
        preview = ", ".join(changed[:5])
        raise RuntimeError(f"frozen acoustic state changed outside mel_decoder: {preview}")


def _require_run_identity(
    run_config: dict[str, object],
    *,
    base_sha256: str,
) -> None:
    if run_config.get("artifact_version") != ISOLATED_MEL_ARTIFACT_VERSION:
        raise RuntimeError("isolated mel artifact version mismatch")
    if run_config.get("trainable_parameter_prefix") != TRAINABLE_PREFIX:
        raise RuntimeError("isolated mel artifact trainable prefix mismatch")
    if run_config.get("base_checkpoint_sha256") != base_sha256:
        raise RuntimeError("isolated mel artifact base checkpoint identity mismatch")


def save_isolated_mel_checkpoint(
    path: Path,
    model: LykenoxSpeechAcousticModel,
    optimizer: torch.optim.Optimizer,
    *,
    base_model: LykenoxSpeechAcousticModel,
    base_sha256: str,
    frontend: SpanishTextFrontend,
    epoch: int,
    next_item_offset: int,
    global_step: int,
    training_provenance: dict[str, object],
    run_config: dict[str, object],
    training_metadata: dict[str, object],
) -> Path:
    _require_run_identity(run_config, base_sha256=base_sha256)
    require_frozen_state_exact(model, base_model)
    return save_acoustic_prosody_checkpoint(
        path,
        model,
        optimizer,
        frontend=frontend,
        epoch=epoch,
        next_item_offset=next_item_offset,
        global_step=global_step,
        training_provenance=training_provenance,
        run_config=run_config,
        training_metadata=training_metadata,
    )


def load_isolated_mel_checkpoint(
    path: Path,
    *,
    base_checkpoint: Path,
    expected_provenance: dict[str, object],
    expected_run_config: dict[str, object],
) -> tuple[LykenoxSpeechAcousticModel, dict[str, Any]]:
    base_checkpoint = Path(base_checkpoint)
    base_sha256 = file_sha256(base_checkpoint)
    _require_run_identity(expected_run_config, base_sha256=base_sha256)
    model, payload = load_acoustic_prosody_checkpoint(
        path,
        expected_provenance=expected_provenance,
        expected_run_config=expected_run_config,
    )
    base_model, _base_payload = load_acoustic_prosody_checkpoint(base_checkpoint)
    require_frozen_state_exact(model, base_model)
    freeze_except_mel_decoder(model)
    model.cpu().eval()
    return model, payload
