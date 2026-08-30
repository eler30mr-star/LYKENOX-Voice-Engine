"""Mandatory full-utterance perceptual gate for V7 after exactly one epoch.

For three fixed held-out validation utterances this probe writes, in matching order:
reference audio, the accepted v4.2 oracle baseline, and the V7 epoch-1 oracle output.
All systems receive the same target mel/F0/voicing grid. No predicted-duration changes,
post-hoc gain/normalization, EQ, denoising, or checkpoint mutation are permitted.

Objective metrics may reject V7, but cannot grant perceptual acceptance. Listening against
v4.2 is mandatory before any V7 epoch 2 can ever be considered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from lykenox_voice_engine.audio.io import load_audio
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import (
    VOCODER_GENERATOR_V4_2_ARCHITECTURE,
    VOCODER_GENERATOR_V7_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import _wave_metrics
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import LogMelEnvelopeLoss
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    target_relative_level_loss,
    target_relative_presence_loss,
)
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
    target_relative_local_spectral_contrast_loss,
)
from lykenox_voice_engine.training.speech_vocoder_losses import multi_resolution_reconstruction_loss
from lykenox_voice_engine.training.speech_vocoder_source_balance import target_relative_spectral_balance_loss
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v7_artifact import load_v7_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v7_content_loss import (
    VOCODER_V7_CONTENT_LOSS_VERSION,
    V7MelContentConsistencyLoss,
)

AUDIT_VERSION = "vocoder-v7-epoch1-full-utterance-oracle-probe-v1"
VALIDATION_INDICES = (0, 1, 2)
V7_ARTIFACT_DIR_NAME = "vocoder_source_free_v7_first_epoch"
OUTPUT_DIR_NAME = "vocoder_v7_epoch1_full_utterance_oracle_probe_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _load_reference_waveform(path: Path, *, sample_rate: int, samples: int) -> torch.Tensor:
    waveform, source_rate = load_audio(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(source_rate) != int(sample_rate):
        waveform = torchaudio.functional.resample(waveform, int(source_rate), int(sample_rate))
    wave = waveform[0].to(torch.float32)
    if int(wave.numel()) < samples:
        wave = F.pad(wave, (0, samples - int(wave.numel())))
    else:
        wave = wave[:samples]
    return wave.contiguous()


def _db_ratio(value: float, reference: float) -> float:
    if value <= 0.0 or reference <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(value / reference)


def _quality_metrics(
    generator,
    envelope_loss: LogMelEnvelopeLoss,
    content_loss: V7MelContentConsistencyLoss,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    conditioning_mel: torch.Tensor,
) -> dict[str, object]:
    prediction_batch = prediction.unsqueeze(0)
    reference_batch = reference.unsqueeze(0)
    with torch.no_grad():
        reconstruction = multi_resolution_reconstruction_loss(prediction_batch, reference_batch).total
        envelope = envelope_loss(prediction_batch, reference_batch)
        content = content_loss(prediction_batch, conditioning_mel)
        balance = target_relative_spectral_balance_loss(
            prediction_batch, reference_batch, sample_rate=generator.config.sample_rate
        )
        contrast = target_relative_local_spectral_contrast_loss(
            prediction_batch, reference_batch, hop_length=generator.config.hop_length
        )
        level = target_relative_level_loss(prediction_batch, reference_batch)
        presence = target_relative_presence_loss(
            prediction_batch,
            reference_batch,
            sample_rate=generator.config.sample_rate,
            hop_length=generator.config.hop_length,
        )
    wave = _wave_metrics(prediction, generator.config.sample_rate)
    reference_wave = _wave_metrics(reference, generator.config.sample_rate)
    return {
        **wave,
        "reconstruction_loss": round(float(reconstruction.detach()), 6),
        "envelope_loss": round(float(envelope.total.detach()), 6),
        "content_loss": round(float(content.total.detach()), 6),
        "content_log_mel": round(float(content.log_mel_l1.detach()), 6),
        "content_centered_shape": round(float(content.centered_shape_l1.detach()), 6),
        "content_spectral_delta": round(float(content.spectral_delta_l1.detach()), 6),
        "content_temporal_delta": round(float(content.temporal_delta_l1.detach()), 6),
        "content_temporal_acceleration": round(float(content.temporal_acceleration_l1.detach()), 6),
        "spectral_balance_loss": round(float(balance.loss.detach()), 6),
        "local_spectral_contrast_loss": round(float(contrast.loss.detach()), 6),
        "level_loss": round(float(level.loss.detach()), 6),
        "rms_error_db": round(float(level.rms_error_db.detach()), 6),
        "rms_relative_to_reference_db": round(
            _db_ratio(float(wave["rms"]), float(reference_wave["rms"])), 3
        ),
        "presence_loss": round(float(presence.loss.detach()), 6),
        "presence_1k_8k_error_db": round(float(presence.presence_1k_8k_error_db.detach()), 6),
        "band_80_300": round(float(presence.prediction_band_fractions[0].detach()), 6),
        "band_300_1000": round(float(presence.prediction_band_fractions[1].detach()), 6),
        "band_1k_3k": round(float(presence.prediction_band_fractions[2].detach()), 6),
        "band_3k_8k": round(float(presence.prediction_band_fractions[3].detach()), 6),
        "target_band_80_300": round(float(presence.target_band_fractions[0].detach()), 6),
        "target_band_300_1000": round(float(presence.target_band_fractions[1].detach()), 6),
        "target_band_1k_3k": round(float(presence.target_band_fractions[2].detach()), 6),
        "target_band_3k_8k": round(float(presence.target_band_fractions[3].detach()), 6),
    }


def run_v7_full_utterance_oracle_probe(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    training = root / "models" / "lykenox_identity" / "training"
    v7_dir = training / V7_ARTIFACT_DIR_NAME
    v7_best = v7_dir / "best.pt"
    v7_last = v7_dir / "last.pt"
    v4_2_best = training / "vocoder_source_filter_v4_2" / "best.pt"
    v6_prior = training / "vocoder_direct_waveform_v6"
    v6_clarity = training / "vocoder_direct_waveform_v6_clarity_guard_v1"
    protected_paths = {
        "v4_2_best": v4_2_best,
        "v6_prior_best": v6_prior / "best.pt",
        "v6_prior_last": v6_prior / "last.pt",
        "v6_clarity_best": v6_clarity / "best.pt",
        "v6_clarity_last": v6_clarity / "last.pt",
        "v7_best": v7_best,
        "v7_last": v7_last,
    }
    missing = [name for name, path in protected_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing V7 oracle checkpoints: {missing}")
    before = {name: _sha256(path) for name, path in protected_paths.items()}

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    v7, v7_payload = load_v7_checkpoint(v7_best)
    v4_2, _v4_2_discriminator, v4_2_payload = load_v4_2_checkpoint(v4_2_best)
    v7.cpu().eval(); v4_2.cpu().eval()

    v7_meta = v7_payload.get("training_metadata")
    v7_provenance = v7_payload.get("training_provenance")
    if not isinstance(v7_meta, dict) or not isinstance(v7_provenance, dict):
        raise RuntimeError("V7 checkpoint lacks training metadata/provenance")
    history = v7_meta.get("history")
    best_epoch = int(v7_meta.get("best_epoch", 0))
    v7_identity_exact = (
        v7.architecture == VOCODER_GENERATOR_V7_ARCHITECTURE
        and v7_payload.get("generator_architecture") == VOCODER_GENERATOR_V7_ARCHITECTURE
        and v7_payload.get("source_free") is True
        and v7_payload.get("sample_phase_conditioning") is False
        and v7_payload.get("sample_rate_pitch_features") is False
        and v7_payload.get("deterministic_noise_conditioning") is False
        and v7_payload.get("level_rescue_branch") is False
        and v7_payload.get("epoch") == 2
        and v7_payload.get("next_item_offset") == 0
        and isinstance(history, list)
        and len(history) == 1
        and best_epoch == 1
        and v7_provenance.get("v7_content_loss_version") == VOCODER_V7_CONTENT_LOSS_VERSION
    )
    v4_2_identity_exact = (
        v4_2.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and v4_2_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not v7_identity_exact or not v4_2_identity_exact:
        raise RuntimeError("V7 oracle checkpoint architecture/training-gate identity mismatch")

    speech_config = LykenoxSpeechConfig()
    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        speech_config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("Not enough held-out validation items for V7 oracle probe")

    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_utterance_oracle_probe_report.json"
    envelope_loss = LogMelEnvelopeLoss(speech_config).cpu()
    content_loss = V7MelContentConsistencyLoss(speech_config).cpu()
    rows: list[dict[str, object]] = []
    structural_checks: list[bool] = []

    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            item = dataset[dataset_index]
            base_item = dataset.base[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("V7 oracle requires cached target F0/voicing")
            frames = int(batch.mel_lengths[0])
            expected_samples = frames * speech_config.hop_length
            teacher_duration_sum = int(batch.durations[0].sum())
            if teacher_duration_sum != frames:
                raise RuntimeError("Teacher duration grid does not match full mel grid")
            mel = batch.mel[:, :frames]
            f0_hz = batch.f0_hz[:, :frames]
            voiced = batch.voiced[:, :frames]
            v4_2_batch = v4_2(mel, f0_hz, voiced)
            v7_batch = v7(mel, f0_hz, voiced)
            shapes_exact = (
                tuple(v4_2_batch.shape) == (1, expected_samples)
                and tuple(v7_batch.shape) == (1, expected_samples)
            )
            finite = bool(torch.isfinite(v4_2_batch).all()) and bool(torch.isfinite(v7_batch).all())
            structural_checks.extend((shapes_exact, finite))
            if not shapes_exact or not finite:
                raise RuntimeError("V7 full-utterance oracle structural contract failed")

            reference = _load_reference_waveform(
                Path(str(base_item["wav_path"])),
                sample_rate=speech_config.sample_rate,
                samples=expected_samples,
            )
            v4_2_wave = v4_2_batch[0].detach().cpu().to(torch.float32).contiguous()
            v7_wave = v7_batch[0].detach().cpu().to(torch.float32).contiguous()
            prefix = f"{audit_index:02d}"
            reference_path = output_dir / f"{prefix}_reference.wav"
            v4_2_path = output_dir / f"{prefix}_v4_2_oracle.wav"
            v7_path = output_dir / f"{prefix}_v7_epoch1_oracle.wav"
            sf.write(str(reference_path), reference.numpy(), speech_config.sample_rate, subtype="PCM_16")
            sf.write(str(v4_2_path), v4_2_wave.numpy(), speech_config.sample_rate, subtype="PCM_16")
            sf.write(str(v7_path), v7_wave.numpy(), speech_config.sample_rate, subtype="PCM_16")

            rows.append({
                "audit_index": audit_index,
                "dataset_index": dataset_index,
                "utterance_id": str(item["utterance_id"]),
                "text": str(item["text"]),
                "teacher_mel_frames": frames,
                "teacher_duration_sum_frames": teacher_duration_sum,
                "duration_seconds": round(expected_samples / speech_config.sample_rate, 4),
                "target_voiced_fraction": round(float(voiced.mean()), 6),
                "reference": {"wav_path": str(reference_path), **_wave_metrics(reference, speech_config.sample_rate)},
                "v4_2_oracle": {
                    "wav_path": str(v4_2_path),
                    **_quality_metrics(v4_2, envelope_loss, content_loss, v4_2_wave, reference, mel),
                },
                "v7_epoch1_oracle": {
                    "wav_path": str(v7_path),
                    **_quality_metrics(v7, envelope_loss, content_loss, v7_wave, reference, mel),
                },
            })

    after = {name: _sha256(path) for name, path in protected_paths.items()}
    checkpoints_unchanged = before == after
    structural_gate_pass = all(structural_checks) and checkpoints_unchanged
    report: dict[str, object] = {
        "status": "needs_listening" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "v7_checkpoint": str(v7_best),
        "v7_best_epoch": best_epoch,
        "v7_global_step": int(v7_payload.get("global_step", -1)),
        "v7_identity_exact": v7_identity_exact,
        "v7_content_loss_version": str(v7_provenance.get("v7_content_loss_version", "")),
        "v4_2_checkpoint": str(v4_2_best),
        "v4_2_identity_exact": v4_2_identity_exact,
        "structural_gate_pass": structural_gate_pass,
        "checkpoints_unchanged": checkpoints_unchanged,
        "full_utterance_count": len(rows),
        "oracle_conditioning_used_for_audit_only": True,
        "reference_audio_required_for_product_inference": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "persistent_training_complete": False,
        "full_utterance_perceptual_acceptance": False,
        "epoch2_training_authorized": False,
        "listening_order": "reference -> v4.2 -> v7 epoch1",
        "listening_gate": {
            "intelligibility": "v7 must be at least as intelligible as v4.2",
            "gangoso_nasal_muffled": "v7 must not be worse than v4.2",
            "periodic_whine_metallic": "v7 must not be worse than v4.2",
            "consonants_formants": "v7 must preserve recognisable phonetic detail",
            "volume": "judge raw model level; no post-hoc normalization is allowed",
        },
        "items": rows,
        "next_gate": "listen_v7_epoch1_vs_v4_2_then_accept_or_reject_before_any_epoch2",
    }
    _atomic_json(report_path, report)
    return {**report, "report_path": str(report_path), "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_v7_full_utterance_oracle_probe(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
