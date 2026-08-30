"""Early full-utterance listening probe for the V6 clarity-guard vocoder.

This diagnostic intentionally runs before persistent V6 training is complete.  Its purpose
is to answer the perceptual question that crop metrics cannot: does the current best V6
clarity checkpoint actually sound cleaner, less nasal/muffled, and usefully loud on complete
held-out utterances?

For three fixed validation items it writes the real reference, the accepted v4.2 baseline,
and the current V6 clarity-guard best checkpoint.  It uses target mel + target F0 + target
voicing only for this oracle audit.  It never applies gain normalization, EQ, denoising, or
predicted-duration changes, and it never mutates training checkpoints.
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
    VOCODER_GENERATOR_V6_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import (
    _wave_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import LogMelEnvelopeLoss
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
    target_relative_level_loss,
    target_relative_presence_loss,
)
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
    target_relative_local_spectral_contrast_loss,
)
from lykenox_voice_engine.training.speech_vocoder_losses import (
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v6_artifact import load_v6_checkpoint


AUDIT_VERSION = "vocoder-v6-clarity-full-utterance-oracle-probe-v1"
EXPECTED_LEVEL_PRESENCE_VERSION = "vocoder-level-presence-v3"
VALIDATION_INDICES = (0, 1, 2)


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


def _load_reference_waveform(
    wav_path: Path,
    *,
    sample_rate: int,
    samples: int,
) -> torch.Tensor:
    waveform, source_rate = load_audio(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(source_rate) != int(sample_rate):
        waveform = torchaudio.functional.resample(
            waveform,
            int(source_rate),
            int(sample_rate),
        )
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
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    prediction_batch = prediction.unsqueeze(0)
    reference_batch = reference.unsqueeze(0)
    with torch.no_grad():
        reconstruction = multi_resolution_reconstruction_loss(
            prediction_batch,
            reference_batch,
        ).total
        envelope = envelope_loss(prediction_batch, reference_batch)
        balance = target_relative_spectral_balance_loss(
            prediction_batch,
            reference_batch,
            sample_rate=generator.config.sample_rate,
        )
        contrast = target_relative_local_spectral_contrast_loss(
            prediction_batch,
            reference_batch,
            hop_length=generator.config.hop_length,
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
        "spectral_balance_loss": round(float(balance.loss.detach()), 6),
        "local_spectral_contrast_loss": round(float(contrast.detach()), 6),
        "level_loss": round(float(level.loss.detach()), 6),
        "rms_error_db": round(float(level.rms_error_db.detach()), 6),
        "rms_relative_to_reference_db": round(
            _db_ratio(float(wave["rms"]), float(reference_wave["rms"])), 3
        ),
        "presence_loss": round(float(presence.loss.detach()), 6),
        "presence_1k_8k_error_db": round(
            float(presence.presence_1k_8k_error_db.detach()), 6
        ),
        "band_80_300": round(float(presence.prediction_band_fractions[0].detach()), 6),
        "band_300_1000": round(float(presence.prediction_band_fractions[1].detach()), 6),
        "band_1k_3k": round(float(presence.prediction_band_fractions[2].detach()), 6),
        "band_3k_8k": round(float(presence.prediction_band_fractions[3].detach()), 6),
        "target_band_80_300": round(float(presence.target_band_fractions[0].detach()), 6),
        "target_band_300_1000": round(float(presence.target_band_fractions[1].detach()), 6),
        "target_band_1k_3k": round(float(presence.target_band_fractions[2].detach()), 6),
        "target_band_3k_8k": round(float(presence.target_band_fractions[3].detach()), 6),
    }


def run_v6_clarity_full_utterance_oracle_probe(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    clarity_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_direct_waveform_v6_clarity_guard_v1"
    )
    v6_path = clarity_dir / "best.pt"
    v6_last_path = clarity_dir / "last.pt"
    v4_2_path = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_source_filter_v4_2"
        / "best.pt"
    )
    prior_v6_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_direct_waveform_v6"
    )
    protected_paths = {
        "v4_2_best": v4_2_path,
        "v6_prior_best": prior_v6_dir / "best.pt",
        "v6_prior_last": prior_v6_dir / "last.pt",
        "v6_clarity_best": v6_path,
        "v6_clarity_last": v6_last_path,
    }
    missing = [name for name, path in protected_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing oracle-probe checkpoints: {missing}")
    before = {name: _sha256(path) for name, path in protected_paths.items()}

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_v6_clarity_full_utterance_oracle_probe_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_utterance_oracle_probe_report.json"

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    v6, _v6_discriminator, v6_payload = load_v6_checkpoint(v6_path)
    v4_2, _v4_2_discriminator, v4_2_payload = load_v4_2_checkpoint(v4_2_path)
    v6.cpu().eval()
    v4_2.cpu().eval()

    v6_meta = v6_payload.get("training_metadata")
    v6_provenance = v6_payload.get("training_provenance")
    if not isinstance(v6_meta, dict) or not isinstance(v6_provenance, dict):
        raise RuntimeError("V6 clarity checkpoint lacks training metadata/provenance")
    best_epoch = int(v6_meta.get("best_epoch", 0))
    loss_version = str(v6_provenance.get("level_presence_loss_version", ""))
    v6_identity_exact = (
        v6.architecture == VOCODER_GENERATOR_V6_ARCHITECTURE
        and v6_payload.get("generator_architecture") == VOCODER_GENERATOR_V6_ARCHITECTURE
        and v6_payload.get("waveform_shape_level_decoupled") is True
        and v6_payload.get("explicit_source") is False
        and v6_payload.get("voiced_noise_source") is False
        and v6_payload.get("raw_source_bypass") is False
    )
    v4_2_identity_exact = (
        v4_2.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and v4_2_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not v6_identity_exact or not v4_2_identity_exact:
        raise RuntimeError("Oracle probe checkpoint architecture identity mismatch")
    if best_epoch < 1:
        raise RuntimeError("Oracle probe requires a V6 clarity best checkpoint from a completed epoch")
    if loss_version != EXPECTED_LEVEL_PRESENCE_VERSION:
        raise RuntimeError(
            f"Oracle probe requires {EXPECTED_LEVEL_PRESENCE_VERSION}, got {loss_version}"
        )
    if VOCODER_LEVEL_PRESENCE_VERSION != EXPECTED_LEVEL_PRESENCE_VERSION:
        raise RuntimeError("Active level/presence implementation version mismatch")

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
        raise RuntimeError("Not enough held-out validation items for V6 oracle probe")

    envelope_loss = LogMelEnvelopeLoss(speech_config).cpu()
    items: list[dict[str, object]] = []
    structural_checks: list[bool] = []

    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            item = dataset[dataset_index]
            base_item = dataset.base[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("V6 oracle probe requires cached target F0/voicing")

            frames = int(batch.mel_lengths[0])
            expected_samples = frames * speech_config.hop_length
            teacher_duration_sum = int(batch.durations[0].sum())
            if teacher_duration_sum != frames:
                raise RuntimeError("Teacher duration grid does not match full mel grid")

            mel = batch.mel[:, :frames]
            f0_hz = batch.f0_hz[:, :frames]
            voiced = batch.voiced[:, :frames]
            v4_2_batch = v4_2(mel, f0_hz, voiced)
            v6_batch = v6(mel, f0_hz, voiced)
            shapes_exact = (
                tuple(v4_2_batch.shape) == (1, expected_samples)
                and tuple(v6_batch.shape) == (1, expected_samples)
            )
            finite = bool(torch.isfinite(v4_2_batch).all()) and bool(
                torch.isfinite(v6_batch).all()
            )
            structural_checks.extend((shapes_exact, finite))
            if not shapes_exact or not finite:
                raise RuntimeError("V6 full-utterance oracle structural contract failed")

            reference = _load_reference_waveform(
                Path(str(base_item["wav_path"])),
                sample_rate=speech_config.sample_rate,
                samples=expected_samples,
            )
            v4_2_wave = v4_2_batch[0].detach().cpu().to(torch.float32).contiguous()
            v6_wave = v6_batch[0].detach().cpu().to(torch.float32).contiguous()

            prefix = f"{audit_index:02d}"
            reference_path = output_dir / f"{prefix}_reference.wav"
            v4_2_wav_path = output_dir / f"{prefix}_v4_2_oracle.wav"
            v6_wav_path = output_dir / f"{prefix}_v6_clarity_oracle.wav"
            sf.write(str(reference_path), reference.numpy(), speech_config.sample_rate, subtype="PCM_16")
            sf.write(str(v4_2_wav_path), v4_2_wave.numpy(), speech_config.sample_rate, subtype="PCM_16")
            sf.write(str(v6_wav_path), v6_wave.numpy(), speech_config.sample_rate, subtype="PCM_16")

            reference_metrics = _wave_metrics(reference, speech_config.sample_rate)
            v4_2_metrics = _quality_metrics(v4_2, envelope_loss, v4_2_wave, reference)
            v6_metrics = _quality_metrics(v6, envelope_loss, v6_wave, reference)
            items.append(
                {
                    "audit_index": audit_index,
                    "dataset_index": dataset_index,
                    "utterance_id": str(item["utterance_id"]),
                    "text": str(item["text"]),
                    "teacher_mel_frames": frames,
                    "teacher_duration_sum_frames": teacher_duration_sum,
                    "duration_seconds": round(expected_samples / speech_config.sample_rate, 4),
                    "target_voiced_fraction": round(float(voiced.mean()), 6),
                    "reference": {"wav_path": str(reference_path), **reference_metrics},
                    "v4_2_oracle": {"wav_path": str(v4_2_wav_path), **v4_2_metrics},
                    "v6_clarity_oracle": {"wav_path": str(v6_wav_path), **v6_metrics},
                }
            )

    after = {name: _sha256(path) for name, path in protected_paths.items()}
    checkpoints_unchanged = before == after
    structural_gate_pass = all(structural_checks) and checkpoints_unchanged
    report: dict[str, object] = {
        "status": "needs_listening" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "v6_checkpoint": str(v6_path),
        "v6_best_epoch": best_epoch,
        "v6_level_presence_loss_version": loss_version,
        "v6_identity_exact": v6_identity_exact,
        "v4_2_checkpoint": str(v4_2_path),
        "v4_2_identity_exact": v4_2_identity_exact,
        "structural_gate_pass": structural_gate_pass,
        "checkpoints_unchanged": checkpoints_unchanged,
        "full_utterance_count": len(items),
        "oracle_conditioning_used_for_audit_only": True,
        "reference_audio_required_for_product_inference": False,
        "waveform_pitch_target_required_for_product_inference": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "persistent_training_complete": False,
        "full_utterance_perceptual_acceptance": False,
        "items": items,
        "listening_protocol": (
            "For each numbered item listen in this order: reference, v4.2 oracle, V6 clarity oracle. "
            "Judge useful level first, then nasal/gangoso or muffled coloration, consonant/formant "
            "definition, high-band air/fricatives, periodic or radio-like artifacts, and overall "
            "naturalness. Do not accept V6 from metrics alone."
        ),
        "acceptance_rule": (
            "This is an early diagnostic, not product approval. V6 may continue training only if "
            "the complete-utterance outputs are not materially more nasal/muffled/noisy than v4.2 "
            "and show a credible improvement in useful level without sacrificing intelligibility. "
            "If the old audible failure remains, stop and revise before additional epochs."
        ),
        "next_gate": (
            "listen_v6_clarity_epoch2_vs_v4_2_before_more_training"
            if structural_gate_pass
            else "fix_v6_clarity_full_utterance_probe_before_more_training"
        ),
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_v6_clarity_full_utterance_oracle_probe(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
