"""Persistent checkpoint contract for the accepted LYKENOX v4.1 vocoder candidate."""

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
    LykenoxVocoderGeneratorV41,
    VOCODER_GENERATOR_V4_1_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_aligner_train import _manifest_path
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_vocoder_data import VOCODER_SEGMENT_CONTRACT_VERSION
from lykenox_voice_engine.training.speech_vocoder_losses import VOCODER_LOSS_RECIPE_VERSION
from lykenox_voice_engine.training.speech_vocoder_source_balance import VOCODER_SOURCE_BALANCE_VERSION


SOURCE_FILTER_CHECKPOINT_VERSION = 1
SOURCE_FILTER_CHECKPOINT_KIND = "lykenox_source_filter_vocoder_training_checkpoint"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_filter_training_provenance(
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
        "pitch_target_version": PITCH_TARGET_VERSION,
        "source_balance_version": VOCODER_SOURCE_BALANCE_VERSION,
    }


def _validate_provenance(provenance: dict[str, object]) -> None:
    expected = {
        "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
        "loss_recipe_version": VOCODER_LOSS_RECIPE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "source_balance_version": VOCODER_SOURCE_BALANCE_VERSION,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"Source-filter checkpoint provenance mismatch: {key}")


def save_source_filter_checkpoint(
    path: Path,
    generator: LykenoxVocoderGeneratorV41,
    discriminator: LykenoxMultiScaleWaveformDiscriminator,
    *,
    epoch: int,
    global_step: int,
    next_item_offset: int,
    validation_reconstruction_loss: float | None,
    validation_spectral_balance_loss: float | None,
    validation_selection_score: float | None,
    training_provenance: dict[str, object],
    generator_optimizer: torch.optim.Optimizer | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    training_metadata: dict[str, object] | None = None,
) -> Path:
    _validate_provenance(training_provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact_version": SOURCE_FILTER_CHECKPOINT_VERSION,
        "kind": SOURCE_FILTER_CHECKPOINT_KIND,
        "generator_architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
        "discriminator_architecture": DISCRIMINATOR_ARCHITECTURE,
        "generator_config": generator.config.to_dict(),
        "generator_hyperparameters": {
            "hidden_channels": generator.hidden_channels,
            "harmonics": generator.harmonics,
            "harmonic_log_range": generator.harmonic_log_range,
            "highpass_cutoff_hz": generator.highpass_cutoff_hz,
            "highpass_kernel_size": generator.highpass_kernel_size,
        },
        "discriminator_scales": discriminator.scales,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "next_item_offset": int(next_item_offset),
        "validation_reconstruction_loss": None if validation_reconstruction_loss is None else float(validation_reconstruction_loss),
        "validation_spectral_balance_loss": None if validation_spectral_balance_loss is None else float(validation_spectral_balance_loss),
        "validation_selection_score": None if validation_selection_score is None else float(validation_selection_score),
        "training_provenance": dict(training_provenance),
        "training_metadata": dict(training_metadata or {}),
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "generator_optimizer_state": None if generator_optimizer is None else generator_optimizer.state_dict(),
        "discriminator_optimizer_state": None if discriminator_optimizer is None else discriminator_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, path)
    return path


def load_source_filter_checkpoint(
    path: Path,
) -> tuple[LykenoxVocoderGeneratorV41, LykenoxMultiScaleWaveformDiscriminator, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid LYKENOX source-filter checkpoint payload")
    if payload.get("artifact_version") != SOURCE_FILTER_CHECKPOINT_VERSION:
        raise RuntimeError(f"Unsupported source-filter checkpoint version: {payload.get('artifact_version')}")
    if payload.get("kind") != SOURCE_FILTER_CHECKPOINT_KIND:
        raise RuntimeError(f"Unexpected source-filter checkpoint kind: {payload.get('kind')}")
    if payload.get("generator_architecture") != VOCODER_GENERATOR_V4_1_ARCHITECTURE:
        raise RuntimeError("Source-filter generator architecture mismatch")
    if payload.get("discriminator_architecture") != DISCRIMINATOR_ARCHITECTURE:
        raise RuntimeError("Source-filter discriminator architecture mismatch")

    config_payload = payload.get("generator_config")
    hyper = payload.get("generator_hyperparameters")
    provenance = payload.get("training_provenance")
    if not isinstance(config_payload, dict) or not isinstance(hyper, dict) or not isinstance(provenance, dict):
        raise RuntimeError("Source-filter checkpoint metadata is incomplete")
    _validate_provenance(provenance)

    config = LykenoxVocoderConfig(**config_payload)
    speech = LykenoxSpeechConfig()
    if (config.mel_bins, config.sample_rate, config.hop_length) != (speech.mel_bins, speech.sample_rate, speech.hop_length):
        raise RuntimeError("Source-filter checkpoint is incompatible with active speech mel contract")

    generator = LykenoxVocoderGeneratorV41(config, **hyper)
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=int(payload.get("discriminator_scales", 2)))
    generator_state = payload.get("generator_state")
    discriminator_state = payload.get("discriminator_state")
    if not isinstance(generator_state, dict) or not isinstance(discriminator_state, dict):
        raise RuntimeError("Source-filter checkpoint is missing model states")
    generator.load_state_dict(generator_state)
    discriminator.load_state_dict(discriminator_state)
    generator.cpu().eval()
    discriminator.cpu().eval()
    return generator, discriminator, payload
