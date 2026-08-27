"""Versioned persistent training checkpoint contract for LYKENOX Speech."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_aligner_train import _manifest_path
from lykenox_voice_engine.training.speech_aligned_data import EXPECTED_DURATION_CACHE_VERSION
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset


SPEECH_ACOUSTIC_CHECKPOINT_VERSION = 1
SPEECH_ACOUSTIC_CHECKPOINT_KIND = "lykenox_speech_acoustic_checkpoint"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vocabulary_sha256(vocabulary: dict[str, int]) -> str:
    payload = json.dumps(
        vocabulary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_training_provenance(root: Path, duration_root: Path) -> dict[str, object]:
    """Hash the exact text/timing inputs that define an acoustic training run."""

    root = Path(root).resolve()
    duration_root = Path(duration_root).resolve()
    train_manifest = _manifest_path(root, "train")
    val_manifest = _manifest_path(root, "val")
    duration_audit = duration_root / "duration_audit.json"
    if not duration_audit.exists():
        raise FileNotFoundError(f"Duration audit not found: {duration_audit}")

    audit = json.loads(duration_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Acoustic training provenance requires a passing duration audit")
    if audit.get("duration_cache_version") != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError("Acoustic training provenance requires alignment-v3")
    if int(audit.get("suspicious_utterance_count", 0)) != 0:
        raise RuntimeError("Acoustic training provenance requires zero duration outliers")

    return {
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": file_sha256(train_manifest),
        "val_manifest": str(val_manifest),
        "val_manifest_sha256": file_sha256(val_manifest),
        "duration_root": str(duration_root),
        "duration_audit": str(duration_audit),
        "duration_audit_sha256": file_sha256(duration_audit),
        "duration_cache_version": EXPECTED_DURATION_CACHE_VERSION,
        "mel_cache_version": LykenoxSpeechDataset.CACHE_VERSION,
    }


def save_speech_acoustic_checkpoint(
    path: Path,
    model: LykenoxSpeechAcousticModel,
    *,
    frontend: SpanishTextFrontend,
    epoch: int,
    global_step: int,
    validation_loss: float | None,
    training_provenance: dict[str, object],
    optimizer: torch.optim.Optimizer | None = None,
    training_metadata: dict[str, object] | None = None,
) -> Path:
    """Persist an exact resumable training checkpoint, not a runtime artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary = frontend.vocabulary()
    if model.config.vocab_size != frontend.vocab_size:
        raise RuntimeError(
            "Acoustic model vocab_size does not match the active LYKENOX frontend"
        )
    if training_provenance.get("duration_cache_version") != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError("Speech checkpoint requires alignment-v3 provenance")

    payload: dict[str, Any] = {
        "artifact_version": SPEECH_ACOUSTIC_CHECKPOINT_VERSION,
        "kind": SPEECH_ACOUSTIC_CHECKPOINT_KIND,
        "frontend_version": frontend.version,
        "vocabulary": vocabulary,
        "vocabulary_sha256": vocabulary_sha256(vocabulary),
        "model_config": model.config.to_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation_loss": (
            float(validation_loss) if validation_loss is not None else None
        ),
        "training_provenance": dict(training_provenance),
        "training_metadata": dict(training_metadata or {}),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(payload, path)
    return path


def load_speech_acoustic_checkpoint(
    path: Path,
) -> tuple[LykenoxSpeechAcousticModel, dict[str, object]]:
    """Load a checkpoint only when frontend/vocabulary/config contracts still match."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid LYKENOX speech acoustic checkpoint payload")
    if payload.get("artifact_version") != SPEECH_ACOUSTIC_CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported speech checkpoint version: {payload.get('artifact_version')}"
        )
    if payload.get("kind") != SPEECH_ACOUSTIC_CHECKPOINT_KIND:
        raise RuntimeError(f"Unexpected speech checkpoint kind: {payload.get('kind')}")

    frontend = SpanishTextFrontend()
    vocabulary = frontend.vocabulary()
    if payload.get("frontend_version") != frontend.version:
        raise RuntimeError("Speech checkpoint frontend version mismatch")
    if payload.get("vocabulary") != vocabulary:
        raise RuntimeError("Speech checkpoint vocabulary does not match current frontend")
    if payload.get("vocabulary_sha256") != vocabulary_sha256(vocabulary):
        raise RuntimeError("Speech checkpoint vocabulary checksum mismatch")

    config_payload = payload.get("model_config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("Speech checkpoint is missing model_config")
    config = LykenoxSpeechConfig(**config_payload)
    if config.vocab_size != frontend.vocab_size:
        raise RuntimeError("Speech checkpoint model vocab_size is not exact")

    provenance = payload.get("training_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Speech checkpoint is missing training provenance")
    if provenance.get("duration_cache_version") != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError("Speech checkpoint was not trained from alignment-v3")

    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("Speech checkpoint is missing model_state")
    model = LykenoxSpeechAcousticModel(config)
    model.load_state_dict(state)
    model.cpu().eval()
    return model, payload
