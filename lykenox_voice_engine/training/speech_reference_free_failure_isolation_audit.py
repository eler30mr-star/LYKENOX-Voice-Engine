"""Held-out crossover audit for the first reference-free intelligibility failure.

The first text-only acoustic-v2 -> v4.1 waveform smoke is structurally correct but the
result is not sufficiently intelligible by listening.  This gate does not train anything.
It holds alignment-v3 teacher durations fixed on real held-out utterances and swaps acoustic
conditioning factors before the accepted v4.1 vocoder so the failure can be localized:

1. target mel + target F0/voicing          (vocoder/oracle baseline)
2. predicted mel + target F0/voicing       (mel-only substitution)
3. target mel + predicted F0/voicing       (prosody-only substitution)
4. predicted mel + predicted F0/voicing    (full acoustic substitution)

It also audits the predicted token durations against alignment-v3 separately.  The purpose
is to distinguish an acoustic-mel problem, an F0/voicing problem, a duration problem, or an
interaction before any further training is authorized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.models.speech.duration_policy import regulate_predicted_durations
from lykenox_voice_engine.models.vocoder.network_v4_1 import VOCODER_GENERATOR_V4_1_ARCHITECTURE
from lykenox_voice_engine.runtime.speech_conditioning import prepare_speech_vocoder_conditioning
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import load_acoustic_prosody_checkpoint
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    FRAME_CONTEXT_VERSION,
    TRAINER_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_end_to_end_smoke import _band_fractions
from lykenox_voice_engine.training.speech_vocoder_source_filter_artifact import load_source_filter_checkpoint


AUDIT_VERSION = "reference-free-failure-isolation-audit-v1"
VARIANTS = (
    "target_mel_target_prosody",
    "predicted_mel_target_prosody",
    "target_mel_predicted_prosody",
    "predicted_mel_predicted_prosody",
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _spectral_centroid_hz(waveform: torch.Tensor, sample_rate: int) -> float:
    wave = waveform.to(torch.float64)
    if wave.numel() < 2:
        return 0.0
    window = torch.hann_window(wave.numel(), periodic=False, dtype=wave.dtype)
    power = torch.fft.rfft(wave * window).abs().square()
    frequencies = torch.fft.rfftfreq(wave.numel(), d=1.0 / float(sample_rate))
    denominator = float(power.sum())
    if not math.isfinite(denominator) or denominator <= 1e-20:
        return 0.0
    return float((power * frequencies).sum()) / denominator


def _wave_metrics(wave: torch.Tensor, sample_rate: int) -> dict[str, object]:
    rms = float(torch.sqrt(torch.mean(wave.square())))
    peak = float(wave.abs().max())
    return {
        "rms": round(rms, 7),
        "peak": round(peak, 7),
        "spectral_centroid_hz": round(_spectral_centroid_hz(wave, sample_rate), 3),
        "band_power_fraction": {
            key: round(value, 6)
            for key, value in _band_fractions(wave, sample_rate).items()
        },
    }


def run_failure_isolation_audit(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    acoustic_path = root / "models" / "lykenox_identity" / "training" / "acoustic_frame_context_v2" / "best.pt"
    vocoder_path = root / "models" / "lykenox_identity" / "training" / "vocoder_source_filter_v4_1" / "best.pt"
    if not acoustic_path.exists():
        raise FileNotFoundError(f"Accepted acoustic v2 checkpoint not found: {acoustic_path}")
    if not vocoder_path.exists():
        raise FileNotFoundError(f"Accepted v4.1 vocoder checkpoint not found: {vocoder_path}")

    output_dir = root / "models" / "lykenox_identity" / "evaluation" / "reference_free_failure_isolation_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "failure_isolation_report.json"

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    acoustic, acoustic_payload = load_acoustic_prosody_checkpoint(acoustic_path)
    run_config = acoustic_payload.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("Accepted acoustic v2 checkpoint is missing run_config")
    acoustic_identity_exact = (
        acoustic.config.frame_context_version == FRAME_CONTEXT_VERSION
        and run_config.get("trainer_contract_version") == TRAINER_CONTRACT_VERSION
        and run_config.get("frame_context_version") == FRAME_CONTEXT_VERSION
    )
    if not acoustic_identity_exact:
        raise RuntimeError("Failure isolation audit requires the accepted acoustic v2 identity")

    vocoder, _discriminator, vocoder_payload = load_source_filter_checkpoint(vocoder_path)
    vocoder_identity_exact = (
        vocoder.architecture == VOCODER_GENERATOR_V4_1_ARCHITECTURE
        and vocoder_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_1_ARCHITECTURE
    )
    if not vocoder_identity_exact:
        raise RuntimeError("Failure isolation audit requires the accepted v4.1 vocoder identity")

    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        acoustic.config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(dataset) < 3:
        raise RuntimeError("Failure isolation audit requires at least three validation items")

    acoustic.cpu().eval()
    vocoder.cpu().eval()
    items: list[dict[str, object]] = []
    duration_abs_errors: list[float] = []
    duration_total_ratios: list[float] = []
    duration_exact_tokens = 0
    duration_valid_tokens = 0

    with torch.inference_mode():
        for audit_index, dataset_index in enumerate((0, 1, 2), start=1):
            item = dataset[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("Failure isolation audit requires cached pitch targets")

            output = acoustic(batch.token_ids, batch.token_mask, batch.durations)
            frames = int(batch.mel_lengths[0])
            predicted = prepare_speech_vocoder_conditioning(output)

            target_mel = batch.mel[:, :frames]
            target_f0 = batch.f0_hz[:, :frames]
            target_voiced = batch.voiced[:, :frames]
            predicted_mel = predicted.mel[:, :frames]
            predicted_f0 = predicted.f0_hz[:, :frames]
            predicted_voiced = predicted.voiced[:, :frames]

            predicted_durations = regulate_predicted_durations(
                batch.token_ids,
                batch.token_mask,
                output["duration_prediction"],
            )
            valid = batch.token_mask[0]
            teacher_duration = batch.durations[0, valid]
            inferred_duration = predicted_durations[0, valid]
            abs_error = torch.abs(inferred_duration - teacher_duration).to(torch.float32)
            duration_abs_errors.extend(float(value) for value in abs_error.tolist())
            duration_exact_tokens += int((inferred_duration == teacher_duration).sum())
            duration_valid_tokens += int(valid.sum())
            teacher_total = int(teacher_duration.sum())
            inferred_total = int(inferred_duration.sum())
            duration_total_ratios.append(inferred_total / max(1, teacher_total))

            variant_inputs = {
                "target_mel_target_prosody": (target_mel, target_f0, target_voiced),
                "predicted_mel_target_prosody": (predicted_mel, target_f0, target_voiced),
                "target_mel_predicted_prosody": (target_mel, predicted_f0, predicted_voiced),
                "predicted_mel_predicted_prosody": (predicted_mel, predicted_f0, predicted_voiced),
            }
            variant_reports: dict[str, object] = {}
            for variant_name in VARIANTS:
                mel, f0_hz, voiced = variant_inputs[variant_name]
                waveform = vocoder(mel, f0_hz, voiced)
                expected_samples = frames * acoustic.config.hop_length
                if tuple(waveform.shape) != (1, expected_samples):
                    raise RuntimeError(f"Vocoder length mismatch in {variant_name}")
                wave = waveform[0].detach().cpu().to(torch.float32).contiguous()
                if not bool(torch.isfinite(wave).all()):
                    raise RuntimeError(f"Non-finite waveform in {variant_name}")
                wav_path = output_dir / f"{audit_index:02d}_{variant_name}.wav"
                sf.write(str(wav_path), wave.numpy(), acoustic.config.sample_rate, subtype="PCM_16")
                variant_reports[variant_name] = {
                    **_wave_metrics(wave, acoustic.config.sample_rate),
                    "wav_path": str(wav_path),
                }

            items.append(
                {
                    "audit_index": audit_index,
                    "dataset_index": dataset_index,
                    "utterance_id": str(item["utterance_id"]),
                    "text": str(item["text"]),
                    "teacher_mel_frames": frames,
                    "predicted_duration_sum_frames": inferred_total,
                    "teacher_duration_sum_frames": teacher_total,
                    "predicted_to_teacher_total_duration_ratio": round(inferred_total / max(1, teacher_total), 6),
                    "duration_token_mae_frames": round(float(abs_error.mean()), 6),
                    "duration_token_exact_fraction": round(float((inferred_duration == teacher_duration).float().mean()), 6),
                    "target_voiced_fraction": round(float(target_voiced.mean()), 6),
                    "predicted_voiced_fraction_teacher_grid": round(float(predicted_voiced.mean()), 6),
                    "variants": variant_reports,
                }
            )

    mean_duration_mae = sum(duration_abs_errors) / max(1, len(duration_abs_errors))
    mean_total_ratio = sum(duration_total_ratios) / max(1, len(duration_total_ratios))
    duration_exact_fraction = duration_exact_tokens / max(1, duration_valid_tokens)

    report: dict[str, object] = {
        "status": "needs_listening",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "acoustic_checkpoint": str(acoustic_path),
        "vocoder_checkpoint": str(vocoder_path),
        "acoustic_identity_exact": acoustic_identity_exact,
        "vocoder_identity_exact": vocoder_identity_exact,
        "teacher_duration_isolation": True,
        "reference_audio_required_for_product_inference": False,
        "validation_item_count": len(items),
        "duration_summary": {
            "mean_token_mae_frames": round(mean_duration_mae, 6),
            "token_exact_fraction": round(duration_exact_fraction, 6),
            "mean_predicted_to_teacher_total_ratio": round(mean_total_ratio, 6),
        },
        "variants": list(VARIANTS),
        "items": items,
        "interpretation": (
            "Listen across each row. If predicted_mel_target_prosody loses intelligibility versus "
            "target_mel_target_prosody, predicted mel is a primary failure source. If "
            "target_mel_predicted_prosody degrades strongly, F0/voicing is a primary source. "
            "If both single substitutions remain acceptable but predicted_mel_predicted_prosody "
            "fails, the main problem is acoustic/vocoder interaction. Duration metrics are reported "
            "separately because this crossover holds teacher durations fixed."
        ),
        "next_gate": "listen_crossover_wavs_and_assign_intelligibility_failure_source",
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_failure_isolation_audit(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
