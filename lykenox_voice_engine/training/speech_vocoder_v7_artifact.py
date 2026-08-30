"""Versioned checkpoint contract for gated V7 first-epoch training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import LykenoxVocoderConfig, LykenoxVocoderGeneratorV7, VOCODER_GENERATOR_V7_ARCHITECTURE
from lykenox_voice_engine.training.speech_aligner_train import _manifest_path
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_vocoder_data import VOCODER_SEGMENT_CONTRACT_VERSION
from lykenox_voice_engine.training.speech_vocoder_v7_content_loss import VOCODER_V7_CONTENT_LOSS_VERSION

V7_CHECKPOINT_VERSION = 1
V7_CHECKPOINT_KIND = "lykenox_v7_source_free_first_epoch_checkpoint"
V7_TRAINING_PHASE = "first_epoch_pre_oracle"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_v7_training_provenance(root: Path, *, segment_mel_frames: int, seed: int) -> dict[str, object]:
    root = Path(root).resolve()
    train = _manifest_path(root, "train")
    val = _manifest_path(root, "val")
    speech = LykenoxSpeechConfig()
    features = speech.feature_cache_dict()
    return {
        "generator_architecture": VOCODER_GENERATOR_V7_ARCHITECTURE,
        "source_family": LykenoxVocoderGeneratorV7.source_family,
        "source_free": True,
        "sample_phase_conditioning": False,
        "sample_rate_pitch_features": False,
        "pitch_conditioning_scope": "frame_latent_only",
        "deterministic_noise_conditioning": False,
        "local_unit_rms_shape_normalization": False,
        "global_unit_rms_shape_normalization": False,
        "level_rescue_branch": False,
        "training_phase": V7_TRAINING_PHASE,
        "train_manifest": str(train), "train_manifest_sha256": _sha(train),
        "val_manifest": str(val), "val_manifest_sha256": _sha(val),
        "mel_cache_version": LykenoxSpeechDataset.CACHE_VERSION,
        "speech_feature_config": features, "speech_feature_config_sha256": _json_sha(features),
        "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
        "segment_mel_frames": int(segment_mel_frames), "segment_seed": int(seed),
        "pitch_target_version": PITCH_TARGET_VERSION,
        "v7_content_loss_version": VOCODER_V7_CONTENT_LOSS_VERSION,
    }


def _validate_provenance(p: dict[str, object]) -> None:
    expected = {
        "generator_architecture": VOCODER_GENERATOR_V7_ARCHITECTURE,
        "source_family": LykenoxVocoderGeneratorV7.source_family,
        "source_free": True,
        "sample_phase_conditioning": False,
        "sample_rate_pitch_features": False,
        "pitch_conditioning_scope": "frame_latent_only",
        "deterministic_noise_conditioning": False,
        "local_unit_rms_shape_normalization": False,
        "global_unit_rms_shape_normalization": False,
        "level_rescue_branch": False,
        "training_phase": V7_TRAINING_PHASE,
        "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "v7_content_loss_version": VOCODER_V7_CONTENT_LOSS_VERSION,
    }
    for key, value in expected.items():
        if p.get(key) != value:
            raise RuntimeError(f"v7 checkpoint provenance mismatch: {key}")


def save_v7_checkpoint(path: Path, generator: LykenoxVocoderGeneratorV7, *, epoch: int, global_step: int, next_item_offset: int, validation_metrics: dict[str, float] | None, training_provenance: dict[str, object], generator_optimizer: torch.optim.Optimizer | None = None, training_metadata: dict[str, object] | None = None) -> Path:
    _validate_provenance(training_provenance)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": V7_CHECKPOINT_VERSION, "kind": V7_CHECKPOINT_KIND,
        "training_phase": V7_TRAINING_PHASE, "generator_architecture": VOCODER_GENERATOR_V7_ARCHITECTURE,
        "source_family": generator.source_family, "source_free": generator.source_free,
        "sample_phase_conditioning": generator.sample_phase_conditioning,
        "sample_rate_pitch_features": generator.sample_rate_pitch_features,
        "pitch_conditioning_scope": generator.pitch_conditioning_scope,
        "deterministic_noise_conditioning": generator.deterministic_noise_conditioning,
        "local_unit_rms_shape_normalization": generator.local_unit_rms_shape_normalization,
        "global_unit_rms_shape_normalization": generator.global_unit_rms_shape_normalization,
        "level_rescue_branch": generator.level_rescue_branch,
        "generator_config": generator.config.to_dict(),
        "generator_hyperparameters": {"frame_channels": generator.frame_channels, "upsample_channels": generator.upsample_channels, "upsample_factors": generator.upsample_factors, "residual_kernels": generator.residual_kernels, "residual_dilations": generator.residual_dilations},
        "epoch": int(epoch), "global_step": int(global_step), "next_item_offset": int(next_item_offset),
        "validation_metrics": None if validation_metrics is None else dict(validation_metrics),
        "training_provenance": dict(training_provenance), "training_metadata": dict(training_metadata or {}),
        "generator_state": generator.state_dict(),
        "generator_optimizer_state": None if generator_optimizer is None else generator_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, path)
    return path


def load_v7_checkpoint(path: Path) -> tuple[LykenoxVocoderGeneratorV7, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_version") != V7_CHECKPOINT_VERSION or payload.get("kind") != V7_CHECKPOINT_KIND:
        raise RuntimeError("Invalid LYKENOX v7 checkpoint payload")
    checks = {"training_phase": V7_TRAINING_PHASE, "generator_architecture": VOCODER_GENERATOR_V7_ARCHITECTURE, "source_family": LykenoxVocoderGeneratorV7.source_family, "source_free": True, "sample_phase_conditioning": False, "sample_rate_pitch_features": False, "pitch_conditioning_scope": "frame_latent_only", "deterministic_noise_conditioning": False, "local_unit_rms_shape_normalization": False, "global_unit_rms_shape_normalization": False, "level_rescue_branch": False}
    for key, value in checks.items():
        if payload.get(key) != value:
            raise RuntimeError(f"v7 checkpoint contract mismatch: {key}")
    config_payload, hyper, provenance = payload.get("generator_config"), payload.get("generator_hyperparameters"), payload.get("training_provenance")
    if not isinstance(config_payload, dict) or not isinstance(hyper, dict) or not isinstance(provenance, dict):
        raise RuntimeError("v7 checkpoint metadata is incomplete")
    _validate_provenance(provenance)
    config = LykenoxVocoderConfig(**config_payload); speech = LykenoxSpeechConfig()
    if (config.mel_bins, config.sample_rate, config.hop_length) != (speech.mel_bins, speech.sample_rate, speech.hop_length):
        raise RuntimeError("v7 checkpoint is incompatible with active speech mel contract")
    generator = LykenoxVocoderGeneratorV7(config, **hyper)
    state = payload.get("generator_state")
    if not isinstance(state, dict):
        raise RuntimeError("v7 checkpoint is missing generator state")
    generator.load_state_dict(state); generator.cpu().eval()
    return generator, payload
