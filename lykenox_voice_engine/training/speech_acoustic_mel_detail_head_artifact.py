"""Checkpoint contract for the frame-hidden mel detail-head candidate.

Only ``detail_head.*`` is persisted as trainable state. The accepted acoustic-v2 base is
referenced by SHA-256 and rebuilt immutably on every load. Encoder, frame context,
mel_decoder, duration and prosody tensors therefore cannot drift inside this experiment.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.models.speech.mel_detail_head import (
    MEL_DETAIL_HEAD_ARCHITECTURE_V1,
    LykenoxAcousticFrameHiddenDetailCandidate,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)


MEL_DETAIL_HEAD_CHECKPOINT_VERSION = 1
MEL_DETAIL_HEAD_CHECKPOINT_KIND = "lykenox_acoustic_frame_hidden_mel_detail_first_epoch_checkpoint"


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


def detail_head_output_dir(root: Path) -> Path:
    return (
        Path(root).resolve()
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_mel_detail_head_v1"
    )


def build_candidate_from_base(base_checkpoint: Path) -> LykenoxAcousticFrameHiddenDetailCandidate:
    base, _payload = load_acoustic_prosody_checkpoint(Path(base_checkpoint))
    base.cpu().eval()
    candidate = LykenoxAcousticFrameHiddenDetailCandidate(base).cpu()
    candidate.eval()
    names = candidate.trainable_parameter_names()
    if not names or any(not name.startswith("detail_head.") for name in names):
        raise RuntimeError("detail-head artifact permits only detail_head trainable parameters")
    if any(parameter.requires_grad for parameter in candidate.base_model.parameters()):
        raise RuntimeError("detail-head base acoustic model must remain frozen")
    return candidate


def _validate_run_config(run_config: dict[str, object], *, base_sha256: str) -> None:
    if run_config.get("architecture") != MEL_DETAIL_HEAD_ARCHITECTURE_V1:
        raise RuntimeError("detail-head checkpoint architecture mismatch")
    if run_config.get("base_checkpoint_sha256") != base_sha256:
        raise RuntimeError("detail-head checkpoint base identity mismatch")
    if run_config.get("trainable_surface") != "detail_head_only":
        raise RuntimeError("detail-head trainable surface mismatch")


def save_mel_detail_head_checkpoint(
    path: Path,
    candidate: LykenoxAcousticFrameHiddenDetailCandidate,
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
    if not names or any(not name.startswith("detail_head.") for name in names):
        raise RuntimeError("only detail_head parameters may be trainable at checkpoint save")
    if any(parameter.requires_grad for parameter in candidate.base_model.parameters()):
        raise RuntimeError("base acoustic model became trainable")
    payload: dict[str, Any] = {
        "checkpoint_version": MEL_DETAIL_HEAD_CHECKPOINT_VERSION,
        "kind": MEL_DETAIL_HEAD_CHECKPOINT_KIND,
        "architecture": MEL_DETAIL_HEAD_ARCHITECTURE_V1,
        "base_checkpoint_sha256": base_sha256,
        "epoch": int(epoch),
        "next_item_offset": int(next_item_offset),
        "global_step": int(global_step),
        "training_provenance": dict(training_provenance),
        "run_config": dict(run_config),
        "training_metadata": dict(training_metadata),
        "detail_head_state": candidate.detail_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_mel_detail_head_checkpoint(
    path: Path,
    *,
    base_checkpoint: Path,
    expected_provenance: dict[str, object] | None = None,
    expected_run_config: dict[str, object] | None = None,
) -> tuple[LykenoxAcousticFrameHiddenDetailCandidate, dict[str, Any]]:
    path = Path(path)
    base_checkpoint = Path(base_checkpoint)
    base_sha = file_sha256(base_checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("invalid detail-head checkpoint payload")
    if payload.get("checkpoint_version") != MEL_DETAIL_HEAD_CHECKPOINT_VERSION:
        raise RuntimeError("unsupported detail-head checkpoint version")
    if payload.get("kind") != MEL_DETAIL_HEAD_CHECKPOINT_KIND:
        raise RuntimeError("unexpected detail-head checkpoint kind")
    if payload.get("architecture") != MEL_DETAIL_HEAD_ARCHITECTURE_V1:
        raise RuntimeError("detail-head checkpoint architecture changed")
    if payload.get("base_checkpoint_sha256") != base_sha:
        raise RuntimeError("detail-head base checkpoint SHA changed")
    run_config = payload.get("run_config")
    provenance = payload.get("training_provenance")
    if not isinstance(run_config, dict) or not isinstance(provenance, dict):
        raise RuntimeError("detail-head checkpoint is missing run identity")
    _validate_run_config(run_config, base_sha256=base_sha)
    if expected_run_config is not None and run_config != expected_run_config:
        raise RuntimeError("detail-head run configuration changed")
    if expected_provenance is not None and provenance != expected_provenance:
        raise RuntimeError("detail-head training provenance changed")
    if not isinstance(payload.get("detail_head_state"), dict):
        raise RuntimeError("detail-head checkpoint is missing detail-head state")
    if not isinstance(payload.get("optimizer_state"), dict):
        raise RuntimeError("detail-head checkpoint is missing optimizer state")
    if not isinstance(payload.get("torch_rng_state"), torch.Tensor):
        raise RuntimeError("detail-head checkpoint is missing torch RNG state")

    candidate = build_candidate_from_base(base_checkpoint)
    candidate.detail_head.load_state_dict(payload["detail_head_state"])
    candidate.cpu().eval()
    return candidate, payload
