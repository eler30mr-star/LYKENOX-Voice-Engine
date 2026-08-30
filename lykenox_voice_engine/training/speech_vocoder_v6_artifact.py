"""Persistent checkpoint contract for LYKENOX vocoder v6 training."""

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
    LykenoxVocoderGeneratorV6,
    VOCODER_GENERATOR_V6_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_aligner_train import _manifest_path
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_vocoder_data import VOCODER_SEGMENT_CONTRACT_VERSION
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import VOCODER_ENVELOPE_LOSS_VERSION
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
    VOCODER_LOCAL_SPECTRAL_CONTRAST_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_losses import VOCODER_LOSS_RECIPE_VERSION
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    VOCODER_SOURCE_BALANCE_VERSION,
)


V6_CHECKPOINT_VERSION = 1
V6_CHECKPOINT_KIND = "lykenox_v6_direct_waveform_training_checkpoint"


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


def build_v6_training_provenance(
    root: Path,
    *,
    segment_mel_frames: int,
    seed: int,
) -> dict[str, object]:
    root = Path(root).resolve()
    train_manifest = _manifest_path(root, "train")
    val_manifest = _manifest_path(root, "val")
    speech_config = LykenoxSpeechConfig()
    feature_config = speech_config.feature_cache_dict()
    return {
        "generator_architecture": VOCODER_GENERATOR_V6_ARCHITECTURE,
        "source_family": LykenoxVocoderGeneratorV6.source_family,
        "explicit_source": False,
        "explicit_sinusoidal_carrier": False,
        "deterministic_harmonics": 0,
        "voiced_noise_source": False,
        "raw_source_bypass": False,
        "conditioning_only_waveform": True,
        "waveform_shape_level_decoupled": True,
        "level_control_family": LykenoxVocoderGeneratorV6.level_control_family,
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": _file_sha256(train_manifest),
        "val_manifest": str(val_manifest),
        "val_manifest_sha256": _file_sha256(val_manifest),
        "mel_cache_version": LykenoxSpeechDataset.CACHE_VERSION,
        "speech_feature_config": feature_config,
        "speech_feature_config_sha256": _json_sha256(feature_config),
        "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
        "segment_mel_frames": int(segment_mel_frames),
        "segment_seed": int(seed),
        "loss_recipe_version": VOCODER_LOSS_RECIPE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "source_balance_version": VOCODER_SOURCE_BALANCE_VERSION,
        "envelope_loss_version": VOCODER_ENVELOPE_LOSS_VERSION,
        "local_spectral_contrast_version": VOCODER_LOCAL_SPECTRAL_CONTRAST_VERSION,
        "level_presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
    }


def _validate_provenance(provenance: dict[str, object]) -> None:
    expected = {
        "generator_architecture": VOCODER_GENERATOR_V6_ARCHITECTURE,
        "source_family": LykenoxVocoderGeneratorV6.source_family,
        "explicit_source": False,
        "explicit_sinusoidal_carrier": False,
        "deterministic_harmonics": 0,
        "voiced_noise_source": False,
        "raw_source_bypass": False,
        "conditioning_only_waveform": True,
        "waveform_shape_level_decoupled": True,
        "level_control_family": LykenoxVocoderGeneratorV6.level_control_family,
        "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
        "loss_recipe_version": VOCODER_LOSS_RECIPE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "source_balance_version": VOCODER_SOURCE_BALANCE_VERSION,
        "envelope_loss_version": VOCODER_ENVELOPE_LOSS_VERSION,
        "local_spectral_contrast_version": VOCODER_LOCAL_SPECTRAL_CONTRAST_VERSION,
        "level_presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"v6 checkpoint provenance mismatch: {key}")


def save_v6_checkpoint(
    path: Path,
    generator: LykenoxVocoderGeneratorV6,
    discriminator: LykenoxMultiScaleWaveformDiscriminator,
    *,
    epoch: int,
    global_step: int,
    next_item_offset: int,
    validation_metrics: dict[str, float] | None,
    training_provenance: dict[str, object],
    generator_optimizer: torch.optim.Optimizer | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    training_metadata: dict[str, object] | None = None,
) -> Path:
    _validate_provenance(training_provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact_version": V6_CHECKPOINT_VERSION,
        "kind": V6_CHECKPOINT_KIND,
        "generator_architecture": VOCODER_GENERATOR_V6_ARCHITECTURE,
        "source_family": generator.source_family,
        "explicit_source": generator.explicit_source,
        "explicit_sinusoidal_carrier": generator.explicit_sinusoidal_carrier,
        "deterministic_harmonics": generator.deterministic_harmonics,
        "voiced_noise_source": generator.voiced_noise_source,
        "raw_source_bypass": generator.raw_source_bypass,
        "conditioning_only_waveform": generator.conditioning_only_waveform,
        "waveform_shape_level_decoupled": generator.waveform_shape_level_decoupled,
        "level_control_family": generator.level_control_family,
        "discriminator_architecture": DISCRIMINATOR_ARCHITECTURE,
        "generator_config": generator.config.to_dict(),
        "generator_hyperparameters": {
            "frame_channels": generator.frame_channels,
            "upsample_channels": generator.upsample_channels,
            "sample_channels": generator.sample_channels,
            "upsample_factors": generator.upsample_factors,
            "sample_dilations": generator.sample_dilations,
            "initial_frame_rms": 0.012,
            "min_frame_rms": generator.min_frame_rms,
            "max_frame_rms": generator.max_frame_rms,
        },
        "discriminator_scales": discriminator.scales,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "next_item_offset": int(next_item_offset),
        "validation_metrics": None if validation_metrics is None else dict(validation_metrics),
        "training_provenance": dict(training_provenance),
        "training_metadata": dict(training_metadata or {}),
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "generator_optimizer_state": (
            None if generator_optimizer is None else generator_optimizer.state_dict()
        ),
        "discriminator_optimizer_state": (
            None if discriminator_optimizer is None else discriminator_optimizer.state_dict()
        ),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, path)
    return path


def load_v6_checkpoint(
    path: Path,
) -> tuple[
    LykenoxVocoderGeneratorV6,
    LykenoxMultiScaleWaveformDiscriminator,
    dict[str, object],
]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid LYKENOX v6 checkpoint payload")
    if payload.get("artifact_version") != V6_CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported v6 checkpoint version: {payload.get('artifact_version')}"
        )
    if payload.get("kind") != V6_CHECKPOINT_KIND:
        raise RuntimeError(f"Unexpected v6 checkpoint kind: {payload.get('kind')}")
    expected_flags = {
        "generator_architecture": VOCODER_GENERATOR_V6_ARCHITECTURE,
        "source_family": LykenoxVocoderGeneratorV6.source_family,
        "explicit_source": False,
        "explicit_sinusoidal_carrier": False,
        "deterministic_harmonics": 0,
        "voiced_noise_source": False,
        "raw_source_bypass": False,
        "conditioning_only_waveform": True,
        "waveform_shape_level_decoupled": True,
        "level_control_family": LykenoxVocoderGeneratorV6.level_control_family,
        "discriminator_architecture": DISCRIMINATOR_ARCHITECTURE,
    }
    for key, value in expected_flags.items():
        if payload.get(key) != value:
            raise RuntimeError(f"v6 checkpoint contract mismatch: {key}")

    config_payload = payload.get("generator_config")
    hyper = payload.get("generator_hyperparameters")
    provenance = payload.get("training_provenance")
    if (
        not isinstance(config_payload, dict)
        or not isinstance(hyper, dict)
        or not isinstance(provenance, dict)
    ):
        raise RuntimeError("v6 checkpoint metadata is incomplete")
    _validate_provenance(provenance)

    config = LykenoxVocoderConfig(**config_payload)
    speech = LykenoxSpeechConfig()
    if (config.mel_bins, config.sample_rate, config.hop_length) != (
        speech.mel_bins,
        speech.sample_rate,
        speech.hop_length,
    ):
        raise RuntimeError("v6 checkpoint is incompatible with active speech mel contract")

    generator = LykenoxVocoderGeneratorV6(config, **hyper)
    discriminator = LykenoxMultiScaleWaveformDiscriminator(
        scales=int(payload.get("discriminator_scales", 2))
    )
    generator_state = payload.get("generator_state")
    discriminator_state = payload.get("discriminator_state")
    if not isinstance(generator_state, dict) or not isinstance(discriminator_state, dict):
        raise RuntimeError("v6 checkpoint is missing model states")
    generator.load_state_dict(generator_state)
    discriminator.load_state_dict(discriminator_state)
    generator.cpu().eval()
    discriminator.cpu().eval()
    return generator, discriminator, payload
