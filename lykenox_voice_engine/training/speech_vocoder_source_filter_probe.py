"""Bounded architecture gate for LYKENOX pitch-conditioned source-filter vocoder v4.

After three mel-only generators learned a 93.75 Hz hop carrier or collapsed spectrally,
this probe changes the conditioning contract instead of adding another upsampling trick.
Pitch and voicing are extracted from the owned target waveform during training and are
fed explicitly to a sample-rate source-filter generator. No old vocoder checkpoint is
resumed.

A pass means only that the new interface improves held-out reconstruction and clears the
known frame-lock/sub-bass failure modes. Listening remains mandatory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import time

import numpy as np
import soundfile as sf
import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderGeneratorV4,
    VOCODER_GENERATOR_V4_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import (
    PITCH_TARGET_VERSION,
    PitchFrames,
    extract_pitch_frames,
)
from lykenox_voice_engine.training.speech_vocoder_data import VocoderSegment, collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_losses import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_periodicity_probe import (
    _generated_specific_frame_lock,
)
from lykenox_voice_engine.training.speech_vocoder_polyphase_forensic import (
    _pitch_metrics,
    _spectral_metrics,
)


PROBE_VERSION = "pitch-source-filter-probe-v4"
DEFAULT_TIME_BUDGET_SECONDS = 85.0


@dataclass(frozen=True)
class ConditionedSegment:
    segment: VocoderSegment
    pitch: PitchFrames


def _condition_segments(segments: list[VocoderSegment]) -> list[ConditionedSegment]:
    conditioned: list[ConditionedSegment] = []
    for segment in segments:
        pitch = extract_pitch_frames(
            segment.waveform,
            frame_count=segment.mel_frames,
        )
        conditioned.append(ConditionedSegment(segment=segment, pitch=pitch))
    return conditioned


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _generate(
    generator: LykenoxVocoderGeneratorV4,
    item: ConditionedSegment,
) -> torch.Tensor:
    return generator(
        item.segment.mel.unsqueeze(0),
        item.pitch.f0_hz.unsqueeze(0),
        item.pitch.voiced.unsqueeze(0),
    )


def _validation_reconstruction(
    generator: LykenoxVocoderGeneratorV4,
    items: list[ConditionedSegment],
) -> float:
    generator.eval()
    values: list[float] = []
    with torch.no_grad():
        for item in items:
            prediction = _generate(generator, item)
            target = item.segment.waveform.unsqueeze(0)
            value = float(
                multi_resolution_reconstruction_loss(prediction, target).total.detach().cpu()
            )
            if not math.isfinite(value):
                raise RuntimeError("Non-finite v4 held-out reconstruction loss")
            values.append(value)
    generator.train()
    return statistics.fmean(values)


def _write_listening_pairs(
    generator: LykenoxVocoderGeneratorV4,
    items: list[ConditionedSegment],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    rows: list[dict[str, object]] = []
    confirmed_locks = 0
    collapse_count = 0
    with torch.no_grad():
        for index, item in enumerate(items[:3], start=1):
            generated_tensor = _generate(generator, item).squeeze(0).detach().cpu()
            reference_tensor = item.segment.waveform.detach().cpu()
            generated = generated_tensor.numpy().astype(np.float64, copy=False)
            reference = reference_tensor.numpy().astype(np.float64, copy=False)

            confirmed, forensic = _generated_specific_frame_lock(
                generated,
                reference,
                generator.config.sample_rate,
            )
            generated_spectral = _spectral_metrics(generated, generator.config.sample_rate)
            generated_pitch = _pitch_metrics(generated, generator.config.sample_rate)
            reference_pitch = _pitch_metrics(reference, generator.config.sample_rate)
            confirmed_locks += int(confirmed)
            collapse_count += int(bool(generated_spectral["subbass_or_silence_collapsed"]))

            generated_path = output_dir / f"val_{index:02d}_{item.segment.utterance_id}_generated.wav"
            reference_path = output_dir / f"val_{index:02d}_{item.segment.utterance_id}_reference.wav"
            sf.write(str(generated_path), generated_tensor.numpy(), generator.config.sample_rate, subtype="PCM_16")
            sf.write(str(reference_path), reference_tensor.numpy(), generator.config.sample_rate, subtype="PCM_16")

            generated_median = generated_pitch.get("median_pitch_hz")
            reference_median = reference_pitch.get("median_pitch_hz")
            pitch_error = None
            if generated_median is not None and reference_median is not None:
                pitch_error = abs(float(generated_median) - float(reference_median))

            rows.append(
                {
                    "utterance_id": item.segment.utterance_id,
                    "generated": str(generated_path),
                    "reference": str(reference_path),
                    "voiced_frame_fraction": round(float(item.pitch.voiced.mean()), 6),
                    "median_conditioning_f0_hz": (
                        round(float(item.pitch.f0_hz[item.pitch.voiced > 0.5].median()), 4)
                        if bool((item.pitch.voiced > 0.5).any())
                        else None
                    ),
                    "generated_pitch": generated_pitch,
                    "reference_pitch": reference_pitch,
                    "generated_reference_median_pitch_error_hz": (
                        round(pitch_error, 4) if pitch_error is not None else None
                    ),
                    "generated_spectral_diagnostic": generated_spectral,
                    **forensic,
                }
            )
    generator.train()
    return rows, confirmed_locks, collapse_count


def run_source_filter_probe(
    root: Path,
    *,
    segment_mel_frames: int = 64,
    train_items: int = 12,
    val_items: int = 4,
    epochs: int = 6,
    warmup_epochs: int = 4,
    seed: int = 1337,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    if epochs < 1 or warmup_epochs < 0 or warmup_epochs > epochs:
        raise ValueError("Invalid v4 probe epoch configuration")

    root = Path(root).resolve()
    started = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_segments, _ = collect_vocoder_segments(
        root,
        "train",
        segment_mel_frames=segment_mel_frames,
        max_items=train_items,
        seed=seed,
    )
    val_segments, _ = collect_vocoder_segments(
        root,
        "val",
        segment_mel_frames=segment_mel_frames,
        max_items=val_items,
        seed=seed,
    )
    train_items_conditioned = _condition_segments(train_segments)
    val_items_conditioned = _condition_segments(val_segments)

    generator = LykenoxVocoderGeneratorV4().cpu().train()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=2e-4, weight_decay=1e-5)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, weight_decay=1e-5)

    initial_validation = _validation_reconstruction(generator, val_items_conditioned)
    best_validation = initial_validation
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    update_times: list[float] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        order = list(range(len(train_items_conditioned)))
        random.Random(seed + epoch).shuffle(order)
        reconstruction_values: list[float] = []
        discriminator_values: list[float] = []
        adversarial_values: list[float] = []
        feature_values: list[float] = []

        for item_index in order:
            if time.perf_counter() - started >= time_budget_seconds:
                artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_source_filter_probe"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "status": "incomplete",
                    "probe_version": PROBE_VERSION,
                    "architecture": VOCODER_GENERATOR_V4_ARCHITECTURE,
                    "epochs_completed": epoch - 1,
                    "global_step": global_step,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "next_gate": "rerun_same_v4_probe",
                }
                report_path = artifact_dir / "probe_report.json"
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                report["report_path"] = str(report_path)
                return report

            item = train_items_conditioned[item_index]
            target = item.segment.waveform.unsqueeze(0)
            update_started = time.perf_counter()

            if epoch <= warmup_epochs:
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = _generate(generator, item)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                if not torch.isfinite(reconstruction.total):
                    raise RuntimeError("Non-finite v4 reconstruction loss")
                reconstruction.total.backward()
                grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(grad)):
                    raise RuntimeError("Non-finite v4 generator gradient")
                generator_optimizer.step()
            else:
                _set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    detached = _generate(generator, item)
                real_output = discriminator(target)
                fake_output = discriminator(detached)
                discriminator_loss = discriminator_hinge_loss(real_output, fake_output)
                discriminator_loss.backward()
                d_grad = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
                if not math.isfinite(float(d_grad)):
                    raise RuntimeError("Non-finite v4 discriminator gradient")
                discriminator_optimizer.step()

                _set_requires_grad(discriminator, False)
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = _generate(generator, item)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                with torch.no_grad():
                    real_features = discriminator(target)
                fake_features = discriminator(prediction)
                adversarial = generator_adversarial_loss(fake_features)
                feature_match = feature_matching_loss(real_features, fake_features)
                generator_loss = reconstruction.total + 0.10 * adversarial + 2.0 * feature_match
                if not torch.isfinite(generator_loss):
                    raise RuntimeError("Non-finite v4 adversarial generator loss")
                generator_loss.backward()
                g_grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(g_grad)):
                    raise RuntimeError("Non-finite v4 adversarial generator gradient")
                generator_optimizer.step()
                _set_requires_grad(discriminator, True)

                discriminator_values.append(float(discriminator_loss.detach().cpu()))
                adversarial_values.append(float(adversarial.detach().cpu()))
                feature_values.append(float(feature_match.detach().cpu()))

            reconstruction_values.append(float(reconstruction.total.detach().cpu()))
            update_times.append(time.perf_counter() - update_started)
            global_step += 1

        validation = _validation_reconstruction(generator, val_items_conditioned)
        history.append(
            {
                "epoch": epoch,
                "phase": "source_filter_reconstruction_warmup" if epoch <= warmup_epochs else "source_filter_adversarial",
                "train_reconstruction": statistics.fmean(reconstruction_values),
                "validation_reconstruction": validation,
                "discriminator_loss": statistics.fmean(discriminator_values) if discriminator_values else None,
                "generator_adversarial_loss": statistics.fmean(adversarial_values) if adversarial_values else None,
                "feature_matching_loss": statistics.fmean(feature_values) if feature_values else None,
            }
        )
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(generator.state_dict())

    artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_source_filter_probe"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "probe_report.json"
    if best_state is None:
        report = {
            "status": "needs_review",
            "device": "cpu",
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V4_ARCHITECTURE,
            "parameters": generator.parameter_count(),
            "epochs_completed": epochs,
            "initial_validation_reconstruction": round(initial_validation, 6),
            "best_validation_reconstruction": round(best_validation, 6),
            "next_gate": "review_v4_optimization",
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    generator.load_state_dict(best_state)
    best_generator_path = artifact_dir / "best_probe_generator.pt"
    torch.save(
        {
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V4_ARCHITECTURE,
            "pitch_target_version": PITCH_TARGET_VERSION,
            "config": generator.config.to_dict(),
            "hidden_channels": generator.hidden_channels,
            "harmonics": generator.harmonics,
            "best_epoch": best_epoch,
            "best_validation_reconstruction": best_validation,
            "generator_state": best_state,
        },
        best_generator_path,
    )

    listening_pairs, confirmed_locks, collapse_count = _write_listening_pairs(
        generator,
        val_items_conditioned,
        artifact_dir / "listening",
    )
    validation_improved = best_validation < initial_validation
    artifact_gate = confirmed_locks == 0 and collapse_count == 0
    status = "pass" if validation_improved and artifact_gate else "needs_review"
    report = {
        "status": status,
        "device": "cpu",
        "probe_version": PROBE_VERSION,
        "architecture": VOCODER_GENERATOR_V4_ARCHITECTURE,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "parameters": generator.parameter_count(),
        "epochs_completed": epochs,
        "warmup_epochs": warmup_epochs,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "segment_mel_frames": segment_mel_frames,
        "segment_audio_seconds": round(segment_mel_frames * generator.config.hop_length / generator.config.sample_rate, 4),
        "initial_validation_reconstruction": round(initial_validation, 6),
        "best_validation_reconstruction": round(best_validation, 6),
        "validation_improved_over_initial": validation_improved,
        "confirmed_generated_specific_frame_locks": confirmed_locks,
        "subbass_or_silence_collapse_count": collapse_count,
        "automatic_artifact_gate_pass": artifact_gate,
        "mean_seconds_per_update": round(statistics.fmean(update_times), 4),
        "min_seconds_per_update": round(min(update_times), 4),
        "max_seconds_per_update": round(max(update_times), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "best_probe_generator": str(best_generator_path),
        "listening_pairs": listening_pairs,
        "next_gate": (
            "listen_v4_source_filter_validation_wavs"
            if status == "pass"
            else "review_v4_source_filter_failure_before_more_training"
        ),
        "architectural_note": (
            "A successful v4 means the speech acoustic contract must gain predicted F0/voicing; "
            "the final runtime will not extract pitch from reference audio."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=64)
    parser.add_argument("--train-items", type=int, default=12)
    parser.add_argument("--val-items", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_source_filter_probe(
                args.root,
                segment_mel_frames=args.segment_mel_frames,
                train_items=args.train_items,
                val_items=args.val_items,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                seed=args.seed,
                time_budget_seconds=args.time_budget_seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
