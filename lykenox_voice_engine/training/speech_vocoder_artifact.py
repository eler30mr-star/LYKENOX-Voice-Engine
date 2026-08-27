"""Persistent checkpoint/provenance contract for the LYKENOX neural vocoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import (
    DISCRIMINATOR_ARCHITECTURE,
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderConfig,
    LykenoxVocoderGenerator,
)
from lykenox_voice_engine.training.speech_aligner_train import _manifest_path
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset
from lykenox_voice_engine.training.speech_vocoder_data import VOCODER_SEGMENT_CONTRACT_VERSION
from lykenox_voice_engine.training.speech_vocoder_losses import VOCODER_LOSS_RECIPE_VERSION


VOCODER_CHECKPOINT_VERSION = 1
VOCODER_CHECKPOINT_KIND = "lykenox_vocoder_training_checkpoint"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_vocoder_training_provenance(
    root: Path,
    *,
    segment_mel_frames: int,
    seed: int,
) -> dict[str, object]:
    root = Path(root).resolve()
    train_manifest = _manifest_path(root, "train")
    val_manifest = _manifest_path(root, "val")
    speech_config = LykenoxSpeechConfig()
    return {
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": _file_sha256(train_manifest),
        "val_manifest": str(val_manifest),
        "val_manifest_sha256": _file_sha256(val_manifest),
        "mel_cache_version": LykenoxSpeechDataset.CACHE_VERSION,
        "speech_mel_config": speech_config.to_dict(),
        "speech_mel_config_sha256": _json_sha256(speech_config.to_dict()),
        "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
        "segment_mel_frames": int(segment_mel_frames),
        "segment_seed": int(seed),
        "loss_recipe_version": VOCODER_LOSS_RECIPE_VERSION,
    }


def save_vocoder_checkpoint(
    path: Path,
    generator: LykenoxVocoderGenerator,
    discriminator: LykenoxMultiScaleWaveformDiscriminator,
    *,
    epoch: int,
    global_step: int,
    validation_reconstruction_loss: float | None,
    training_provenance: dict[str, object],
    generator_optimizer: torch.optim.Optimizer | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    training_metadata: dict[str, object] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if training_provenance.get("segment_contract_version") != VOCODER_SEGMENT_CONTRACT_VERSION:
        raise RuntimeError("Vocoder checkpoint segment contract mismatch")
    if training_provenance.get("loss_recipe_version") != VOCODER_LOSS_RECIPE_VERSION:
        raise RuntimeError("Vocoder checkpoint loss recipe mismatch")

    payload: dict[str, Any] = {
        "artifact_version": VOCODER_CHECKPOINT_VERSION,
        "kind": VOCODER_CHECKPOINT_KIND,
        "generator_architecture": "lykenox_compact_transposed_conv_v0",
        "discriminator_architecture": DISCRIMINATOR_ARCHITECTURE,
        "generator_config": generator.config.to_dict(),
        "discriminator_scales": discriminator.scales,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation_reconstruction_loss": (
            float(validation_reconstruction_loss)
            if validation_reconstruction_loss is not None
            else None
        ),
        "training_provenance": dict(training_provenance),
        "training_metadata": dict(training_metadata or {}),
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "generator_optimizer_state": (
            generator_optimizer.state_dict() if generator_optimizer is not None else None
        ),
        "discriminator_optimizer_state": (
            discriminator_optimizer.state_dict()
            if discriminator_optimizer is not None
            else None
        ),
    }
    torch.save(payload, path)
    return path


def load_vocoder_checkpoint(
    path: Path,
) -> tuple[
    LykenoxVocoderGenerator,
    LykenoxMultiScaleWaveformDiscriminator,
    dict[str, object],
]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid LYKENOX vocoder checkpoint payload")
    if payload.get("artifact_version") != VOCODER_CHECKPOINT_VERSION:
        raise RuntimeError(f"Unsupported vocoder checkpoint version: {payload.get('artifact_version')}")
    if payload.get("kind") != VOCODER_CHECKPOINT_KIND:
        raise RuntimeError(f"Unexpected vocoder checkpoint kind: {payload.get('kind')}")
    if payload.get("generator_architecture") != "lykenox_compact_transposed_conv_v0":
        raise RuntimeError("Vocoder generator architecture mismatch")
    if payload.get("discriminator_architecture") != DISCRIMINATOR_ARCHITECTURE:
        raise RuntimeError("Vocoder discriminator architecture mismatch")

    config_payload = payload.get("generator_config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("Vocoder checkpoint is missing generator_config")
    config = LykenoxVocoderConfig(**config_payload)
    speech_config = LykenoxSpeechConfig()
    if (
        config.mel_bins != speech_config.mel_bins
        or config.sample_rate != speech_config.sample_rate
        or config.hop_length != speech_config.hop_length
    ):
        raise RuntimeError("Vocoder checkpoint is incompatible with active speech mel contract")

    provenance = payload.get("training_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Vocoder checkpoint is missing training provenance")
    if provenance.get("segment_contract_version") != VOCODER_SEGMENT_CONTRACT_VERSION:
        raise RuntimeError("Vocoder checkpoint segment contract is stale")
    if provenance.get("loss_recipe_version") != VOCODER_LOSS_RECIPE_VERSION:
        raise RuntimeError("Vocoder checkpoint loss recipe is stale")

    generator = LykenoxVocoderGenerator(config)
    discriminator = LykenoxMultiScaleWaveformDiscriminator(
        scales=int(payload.get("discriminator_scales", 2))
    )
    generator_state = payload.get("generator_state")
    discriminator_state = payload.get("discriminator_state")
    if not isinstance(generator_state, dict) or not isinstance(discriminator_state, dict):
        raise RuntimeError("Vocoder checkpoint is missing model states")
    generator.load_state_dict(generator_state)
    discriminator.load_state_dict(discriminator_state)
    generator.cpu().eval()
    discriminator.cpu().eval()
    return generator, discriminator, payload
