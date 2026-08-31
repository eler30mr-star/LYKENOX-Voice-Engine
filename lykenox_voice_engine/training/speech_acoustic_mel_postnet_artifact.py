"""Checkpoint contract for the rejected acoustic mel postnet experiment.

The one-epoch residual postnet is retained for forensic reproducibility, but its full-
utterance A/B was perceptually rejected because it was effectively tied with, and slightly
below, the accepted v4.2 baseline. New persistent runs through the default output path are
therefore blocked. Historical checkpoints remain loadable for audits and comparisons.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.models.speech.mel_postnet import (
    MEL_POSTNET_ARCHITECTURE_V1,
    LykenoxAcousticMelPostnetCandidate,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)


MEL_POSTNET_CHECKPOINT_VERSION = 1
MEL_POSTNET_CHECKPOINT_KIND = "lykenox_acoustic_mel_postnet_first_epoch_checkpoint"
MEL_POSTNET_HIDDEN_CHANNELS = 128
MEL_POSTNET_PERCEPTUALLY_REJECTED = True
MEL_POSTNET_PERSISTENT_TRAINING_ENABLED = False


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


def rejected_postnet_output_dir(root: Path) -> Path:
    """Return the historical artifact path without authorizing new training."""
    return (
        Path(root).resolve()
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_mel_postnet_v1"
    )


def postnet_output_dir(root: Path) -> Path:
    del root
    raise RuntimeError(
        "acoustic_mel_postnet_v1 was perceptually rejected: persistent postnet training "
        "is disabled; keep existing checkpoints for forensic A/B only"
    )


def build_candidate_from_base(
    base_checkpoint: Path,
    *,
    hidden_channels: int = MEL_POSTNET_HIDDEN_CHANNELS,
) -> LykenoxAcousticMelPostnetCandidate:
    base, _payload = load_acoustic_prosody_checkpoint(Path(base_checkpoint))
    base.cpu().eval()
    candidate = LykenoxAcousticMelPostnetCandidate(
        base,
        hidden_channels=hidden_channels,
    ).cpu()
    candidate.eval()
    names = candidate.trainable_parameter_names()
    if not names or any(not name.startswith("postnet.") for name in names):
        raise RuntimeError("mel postnet artifact permits only postnet trainable parameters")
    if any(parameter.requires_grad for parameter in candidate.base_model.parameters()):
        raise RuntimeError("mel postnet artifact base model must remain frozen")
    return candidate


def _validate_run_config(run_config: dict[str, object], *, base_sha256: str) -> None:
    if run_config.get("architecture") != MEL_POSTNET_ARCHITECTURE_V1:
        raise RuntimeError("mel postnet checkpoint architecture mismatch")
    if run_config.get("base_checkpoint_sha256") != base_sha256:
        raise RuntimeError("mel postnet checkpoint base identity mismatch")
    if int(run_config.get("hidden_channels", -1)) != MEL_POSTNET_HIDDEN_CHANNELS:
        raise RuntimeError("mel postnet checkpoint hidden-channel contract mismatch")
    if run_config.get("trainable_surface") != "postnet_only":
        raise RuntimeError("mel postnet checkpoint trainable surface mismatch")


def save_mel_postnet_checkpoint(
    path: Path,
    candidate: LykenoxAcousticMelPostnetCandidate,
    optimizer: torch.optim.Optimizer,
    *,
    base_sha256: str,
    epoch: int,
    next_item_offset: int,
    global_step: int,
    training_provenance: dict[str, object],
    run_config: dict[str, object],
    training_metadata: dict[str, object],
) -> Path:
    _validate_run_config(run_config, base_sha256=base_sha256)
    names = candidate.trainable_parameter_names()
    if not names or any(not name.startswith("postnet.") for name in names):
        raise RuntimeError("only postnet parameters may be trainable at checkpoint save")
    if any(parameter.requires_grad for parameter in candidate.base_model.parameters()):
        raise RuntimeError("base acoustic model became trainable")
    payload: dict[str, Any] = {
        "checkpoint_version": MEL_POSTNET_CHECKPOINT_VERSION,
        "kind": MEL_POSTNET_CHECKPOINT_KIND,
        "architecture": MEL_POSTNET_ARCHITECTURE_V1,
        "base_checkpoint_sha256": base_sha256,
        "hidden_channels": MEL_POSTNET_HIDDEN_CHANNELS,
        "epoch": int(epoch),
        "next_item_offset": int(next_item_offset),
        "global_step": int(global_step),
        "training_provenance": dict(training_provenance),
        "run_config": dict(run_config),
        "training_metadata": dict(training_metadata),
        "postnet_state": candidate.postnet.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_mel_postnet_checkpoint(
    path: Path,
    *,
    base_checkpoint: Path,
    expected_provenance: dict[str, object] | None = None,
    expected_run_config: dict[str, object] | None = None,
) -> tuple[LykenoxAcousticMelPostnetCandidate, dict[str, Any]]:
    path = Path(path)
    base_checkpoint = Path(base_checkpoint)
    base_sha = file_sha256(base_checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("invalid mel postnet checkpoint payload")
    if payload.get("checkpoint_version") != MEL_POSTNET_CHECKPOINT_VERSION:
        raise RuntimeError("unsupported mel postnet checkpoint version")
    if payload.get("kind") != MEL_POSTNET_CHECKPOINT_KIND:
        raise RuntimeError("unexpected mel postnet checkpoint kind")
    if payload.get("architecture") != MEL_POSTNET_ARCHITECTURE_V1:
        raise RuntimeError("mel postnet checkpoint architecture changed")
    if payload.get("base_checkpoint_sha256") != base_sha:
        raise RuntimeError("mel postnet base checkpoint SHA changed")
    run_config = payload.get("run_config")
    provenance = payload.get("training_provenance")
    if not isinstance(run_config, dict) or not isinstance(provenance, dict):
        raise RuntimeError("mel postnet checkpoint is missing run identity")
    _validate_run_config(run_config, base_sha256=base_sha)
    if expected_run_config is not None and run_config != expected_run_config:
        raise RuntimeError("mel postnet run configuration changed")
    if expected_provenance is not None and provenance != expected_provenance:
        raise RuntimeError("mel postnet training provenance changed")
    if not isinstance(payload.get("postnet_state"), dict):
        raise RuntimeError("mel postnet checkpoint is missing postnet state")
    if not isinstance(payload.get("optimizer_state"), dict):
        raise RuntimeError("mel postnet checkpoint is missing optimizer state")
    if not isinstance(payload.get("torch_rng_state"), torch.Tensor):
        raise RuntimeError("mel postnet checkpoint is missing torch RNG state")

    candidate = build_candidate_from_base(base_checkpoint)
    candidate.postnet.load_state_dict(payload["postnet_state"])
    candidate.cpu().eval()
    return candidate, payload
