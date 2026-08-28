"""Versioned exactly-resumable checkpoint contract for LYKENOX acoustic prosody training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_aligned_data import EXPECTED_DURATION_CACHE_VERSION
from lykenox_voice_engine.training.speech_aligner_train import _manifest_path
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_pitch_cache import (
    PITCH_CACHE_VERSION,
    load_pitch_cache_index,
)


ACOUSTIC_PROSODY_CHECKPOINT_VERSION = 1
ACOUSTIC_PROSODY_CHECKPOINT_KIND = "lykenox_acoustic_prosody_training_checkpoint"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vocabulary_sha256(vocabulary: dict[str, int]) -> str:
    return json_sha256(vocabulary)


def build_acoustic_prosody_provenance(
    root: Path,
    *,
    duration_root: Path,
    config: LykenoxSpeechConfig,
) -> dict[str, object]:
    """Hash every persistent supervision contract used by acoustic prosody training."""

    root = Path(root).resolve()
    duration_root = Path(duration_root).resolve()
    train_manifest = _manifest_path(root, "train")
    val_manifest = _manifest_path(root, "val")
    duration_audit = duration_root / "duration_audit.json"
    if not duration_audit.exists():
        raise FileNotFoundError(f"Duration audit not found: {duration_audit}")
    audit = json.loads(duration_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Acoustic prosody provenance requires a passing duration audit")
    if audit.get("duration_cache_version") != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError("Acoustic prosody provenance requires alignment-v3")
    if int(audit.get("suspicious_utterance_count", 0)) != 0:
        raise RuntimeError("Acoustic prosody provenance requires zero duration outliers")

    pitch_index = load_pitch_cache_index(root)
    if pitch_index.get("cache_version") != PITCH_CACHE_VERSION:
        raise RuntimeError("Acoustic prosody provenance requires speech-pitch-cache-v1")
    if pitch_index.get("pitch_target_version") != PITCH_TARGET_VERSION:
        raise RuntimeError("Acoustic prosody provenance pitch target version mismatch")
    pitch_index_path = (
        root
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "features"
        / "speech"
        / "pitch-v1"
        / "cache_index.json"
    )
    if int(pitch_index.get("total_count", -1)) != 132:
        raise RuntimeError("Acoustic prosody provenance requires the complete 132-item pitch cache")

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
        "pitch_cache_version": PITCH_CACHE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "pitch_index": str(pitch_index_path),
        "pitch_index_sha256": file_sha256(pitch_index_path),
        "speech_config": config.to_dict(),
        "speech_config_sha256": json_sha256(config.to_dict()),
    }


def save_acoustic_prosody_checkpoint(
    path: Path,
    model: LykenoxSpeechAcousticModel,
    optimizer: torch.optim.Optimizer,
    *,
    frontend: SpanishTextFrontend,
    epoch: int,
    next_item_offset: int,
    global_step: int,
    training_provenance: dict[str, object],
    run_config: dict[str, object],
    training_metadata: dict[str, object],
) -> Path:
    """Atomically save an exact continuation point for persistent acoustic training."""

    if model.config.vocab_size != frontend.vocab_size:
        raise RuntimeError("Acoustic model vocabulary does not match active frontend")
    if training_provenance.get("duration_cache_version") != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError("Acoustic prosody checkpoint requires alignment-v3 provenance")
    if training_provenance.get("pitch_cache_version") != PITCH_CACHE_VERSION:
        raise RuntimeError("Acoustic prosody checkpoint requires pitch-cache provenance")

    vocabulary = frontend.vocabulary()
    payload: dict[str, Any] = {
        "artifact_version": ACOUSTIC_PROSODY_CHECKPOINT_VERSION,
        "kind": ACOUSTIC_PROSODY_CHECKPOINT_KIND,
        "frontend_version": frontend.version,
        "vocabulary": vocabulary,
        "vocabulary_sha256": vocabulary_sha256(vocabulary),
        "model_config": model.config.to_dict(),
        "epoch": int(epoch),
        "next_item_offset": int(next_item_offset),
        "global_step": int(global_step),
        "training_provenance": dict(training_provenance),
        "run_config": dict(run_config),
        "training_metadata": dict(training_metadata),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_acoustic_prosody_checkpoint(
    path: Path,
    *,
    expected_provenance: dict[str, object] | None = None,
    expected_run_config: dict[str, object] | None = None,
) -> tuple[LykenoxSpeechAcousticModel, dict[str, Any]]:
    """Load only a checkpoint whose frontend, data provenance and run identity still match."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid LYKENOX acoustic prosody checkpoint payload")
    if payload.get("artifact_version") != ACOUSTIC_PROSODY_CHECKPOINT_VERSION:
        raise RuntimeError("Unsupported acoustic prosody checkpoint version")
    if payload.get("kind") != ACOUSTIC_PROSODY_CHECKPOINT_KIND:
        raise RuntimeError("Unexpected acoustic prosody checkpoint kind")

    frontend = SpanishTextFrontend()
    vocabulary = frontend.vocabulary()
    if payload.get("frontend_version") != frontend.version:
        raise RuntimeError("Acoustic prosody checkpoint frontend version mismatch")
    if payload.get("vocabulary") != vocabulary:
        raise RuntimeError("Acoustic prosody checkpoint vocabulary mismatch")
    if payload.get("vocabulary_sha256") != vocabulary_sha256(vocabulary):
        raise RuntimeError("Acoustic prosody checkpoint vocabulary checksum mismatch")

    config_payload = payload.get("model_config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("Acoustic prosody checkpoint is missing model_config")
    config = LykenoxSpeechConfig(**config_payload)
    if config.vocab_size != frontend.vocab_size:
        raise RuntimeError("Acoustic prosody checkpoint model vocab_size is not exact")

    provenance = payload.get("training_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Acoustic prosody checkpoint is missing provenance")
    if provenance.get("duration_cache_version") != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError("Acoustic prosody checkpoint was not trained from alignment-v3")
    if provenance.get("pitch_cache_version") != PITCH_CACHE_VERSION:
        raise RuntimeError("Acoustic prosody checkpoint was not trained from pitch-cache-v1")
    if expected_provenance is not None and provenance != expected_provenance:
        raise RuntimeError("Acoustic prosody checkpoint data provenance changed")

    run_config = payload.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("Acoustic prosody checkpoint is missing run_config")
    if expected_run_config is not None and run_config != expected_run_config:
        raise RuntimeError("Acoustic prosody checkpoint run configuration changed")

    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("Acoustic prosody checkpoint is missing model_state")
    if not isinstance(payload.get("optimizer_state"), dict):
        raise RuntimeError("Acoustic prosody checkpoint is missing optimizer_state")
    if not isinstance(payload.get("torch_rng_state"), torch.Tensor):
        raise RuntimeError("Acoustic prosody checkpoint is missing torch RNG state")

    model = LykenoxSpeechAcousticModel(config)
    model.load_state_dict(state)
    model.cpu().eval()
    return model, payload
