"""Versioned persistent artifact contract for the LYKENOX speech aligner."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxCTCAligner, LykenoxCTCAlignerConfig

ALIGNER_ARTIFACT_VERSION = 1
ALIGNER_KIND = "lykenox_ctc_aligner"


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_aligner_checkpoint(
    path: Path,
    model: LykenoxCTCAligner,
    *,
    frontend: SpanishTextFrontend,
    speech_config: dict[str, object],
    epoch: int,
    validation_ctc_loss: float,
    training_metadata: dict[str, object],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": ALIGNER_ARTIFACT_VERSION,
        "kind": ALIGNER_KIND,
        "frontend_version": frontend.version,
        "vocabulary": frontend.vocabulary(),
        "aligner_config": asdict(model.config),
        "speech_config": dict(speech_config),
        "epoch": int(epoch),
        "validation_ctc_loss": float(validation_ctc_loss),
        "training_metadata": dict(training_metadata),
        "model_state": model.state_dict(),
    }
    torch.save(payload, path)
    return path


def load_aligner_checkpoint(
    path: Path,
) -> tuple[LykenoxCTCAligner, dict[str, object]]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid LYKENOX aligner checkpoint payload")
    if payload.get("artifact_version") != ALIGNER_ARTIFACT_VERSION:
        raise RuntimeError(
            f"Unsupported aligner artifact version: {payload.get('artifact_version')}"
        )
    if payload.get("kind") != ALIGNER_KIND:
        raise RuntimeError(f"Unexpected aligner artifact kind: {payload.get('kind')}")

    frontend = SpanishTextFrontend()
    if payload.get("frontend_version") != frontend.version:
        raise RuntimeError(
            "Aligner frontend version does not match the current LYKENOX frontend"
        )
    if payload.get("vocabulary") != frontend.vocabulary():
        raise RuntimeError("Aligner vocabulary does not match the current LYKENOX vocabulary")

    config_payload = payload.get("aligner_config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("Aligner checkpoint is missing aligner_config")
    model = LykenoxCTCAligner(LykenoxCTCAlignerConfig(**config_payload))
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("Aligner checkpoint is missing model_state")
    model.load_state_dict(state)
    model.cpu().eval()
    return model, payload
