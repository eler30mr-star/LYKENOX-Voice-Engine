"""Bounded architecture-selection probe for LYKENOX polyphase vocoder v2.

Held-out listening rejected both earlier compute prototypes for different structural
reasons: v0 produced a mel-frame-rate buzz and resize-conv v1 removed that buzz but
collapsed almost all generated energy below 80 Hz. This command tests a learned polyphase
upsampler that has explicit sample-phase capacity without overlapping transposed
convolutions.

The probe is intentionally short. It selects the lowest held-out reconstruction epoch,
writes three generated/reference WAV pairs, and reports two automatic failure diagnostics:
frame-rate pitch lock and sub-bass/silence collapse. Human listening remains the final
architecture-selection gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import soundfile as sf
import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderGeneratorV2,
    VOCODER_GENERATOR_V2_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_vocoder_data import VocoderSegment, collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_losses import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)


PROBE_VERSION = "polyphase-perceptual-probe-v2"
DEFAULT_TIME_BUDGET_SECONDS = 85.0


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _validation_reconstruction(
    generator: LykenoxVocoderGeneratorV2,
    segments: list[VocoderSegment],
) -> float:
    generator.eval()
    values: list[float] = []
    with torch.no_grad():
        for segment in segments:
            prediction = generator(segment.mel.unsqueeze(0))
            target = segment.waveform.unsqueeze(0)
            value = float(
                multi_resolution_reconstruction_loss(prediction, target).total.detach().cpu()
            )
            if not math.isfinite(value):
                raise RuntimeError("Non-finite polyphase validation loss")
            values.append(value)
    generator.train()
    return statistics.fmean(values)


def _frame_rate_lock_metrics(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_length: int,
) -> dict[str, float | bool | None]:
    """Detect a stable pitch locked to sample_rate / hop_length."""

    y = waveform.detach().cpu().numpy().astype(np.float64, copy=False)
    frame_length = 1024
    analysis_hop = 256
    min_lag = max(1, int(sample_rate / 300.0))
    max_lag = min(int(sample_rate / 60.0), frame_length - 2)
    frames: list[np.ndarray] = []
    rms_values: list[float] = []
    for start in range(0, max(0, len(y) - frame_length + 1), analysis_hop):
        frame = y[start : start + frame_length]
        frames.append(frame)
        rms_values.append(float(np.sqrt(np.mean(frame * frame))))
    frame_rate = sample_rate / hop_length
    if not frames:
        return {
            "median_pitch_hz": None,
            "pitch_std_hz": None,
            "frame_rate_hz": round(frame_rate, 4),
            "frame_rate_locked": False,
        }

    threshold = max(rms_values) * 0.20
    estimates: list[float] = []
    for frame, rms in zip(frames, rms_values, strict=True):
        if rms < threshold:
            continue
        centered = frame - np.mean(frame)
        energy = float(np.dot(centered, centered)) + 1e-12
        best_lag = None
        best_score = -1.0
        for lag in range(min_lag, max_lag + 1):
            score = float(np.dot(centered[:-lag], centered[lag:]) / energy)
            if score > best_score:
                best_score = score
                best_lag = lag
        if best_lag is not None and best_score > 0.15:
            estimates.append(sample_rate / best_lag)

    if not estimates:
        return {
            "median_pitch_hz": None,
            "pitch_std_hz": None,
            "frame_rate_hz": round(frame_rate, 4),
            "frame_rate_locked": False,
        }
    median = float(np.median(estimates))
    std = float(np.std(estimates))
    return {
        "median_pitch_hz": round(median, 4),
        "pitch_std_hz": round(std, 4),
        "frame_rate_hz": round(frame_rate, 4),
        "frame_rate_locked": bool(abs(median - frame_rate) <= 0.75 and std <= 1.0),
    }


def _spectral_collapse_metrics(
    waveform: torch.Tensor,
    sample_rate: int,
) -> dict[str, float | bool]:
    """Detect the near-DC/sub-bass collapse observed in resize-conv v1 outputs."""

    y = waveform.detach().cpu().numpy().astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(y * y)))
    if len(y) < 2:
        return {
            "rms": rms,
            "sub_80hz_energy_fraction": 1.0,
            "above_300hz_energy_fraction": 0.0,
            "spectral_centroid_hz": 0.0,
            "subbass_or_silence_collapsed": True,
        }
    window = np.hanning(len(y))
    power = np.abs(np.fft.rfft(y * window)) ** 2
    frequencies = np.fft.rfftfreq(len(y), d=1.0 / sample_rate)
    total = float(power.sum()) + 1e-20
    sub_80 = float(power[frequencies < 80.0].sum() / total)
    above_300 = float(power[frequencies >= 300.0].sum() / total)
    centroid = float((frequencies * power).sum() / total)
    collapsed = rms < 1e-4 or (sub_80 >= 0.97 and above_300 <= 0.01)
    return {
        "rms": round(rms, 8),
        "sub_80hz_energy_fraction": round(sub_80, 6),
        "above_300hz_energy_fraction": round(above_300, 6),
        "spectral_centroid_hz": round(centroid, 3),
        "subbass_or_silence_collapsed": bool(collapsed),
    }


def _write_listening_pairs(
    generator: LykenoxVocoderGeneratorV2,
    segments: list[VocoderSegment],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    rows: list[dict[str, object]] = []
    frame_lock_count = 0
    collapse_count = 0
    with torch.no_grad():
        for index, segment in enumerate(segments[:3], start=1):
            generated = generator(segment.mel.unsqueeze(0)).squeeze(0).detach().cpu()
            reference = segment.waveform.detach().cpu()
            pitch = _frame_rate_lock_metrics(
                generated,
                generator.config.sample_rate,
                generator.config.hop_length,
            )
            spectral = _spectral_collapse_metrics(generated, generator.config.sample_rate)
            frame_lock_count += int(bool(pitch["frame_rate_locked"]))
            collapse_count += int(bool(spectral["subbass_or_silence_collapsed"]))

            generated_path = output_dir / f"val_{index:02d}_{segment.utterance_id}_generated.wav"
            reference_path = output_dir / f"val_{index:02d}_{segment.utterance_id}_reference.wav"
            sf.write(
                str(generated_path),
                generated.numpy(),
                generator.config.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                str(reference_path),
                reference.numpy(),
                generator.config.sample_rate,
                subtype="PCM_16",
            )
            rows.append(
                {
                    "utterance_id": segment.utterance_id,
                    "generated": str(generated_path),
                    "reference": str(reference_path),
                    "generated_pitch_diagnostic": pitch,
                    "generated_spectral_diagnostic": spectral,
                }
            )
    generator.train()
    return rows, frame_lock_count, collapse_count


def run_polyphase_probe(
    root: Path,
    *,
    segment_mel_frames: int = 96,
    train_items: int = 16,
    val_items: int = 6,
    epochs: int = 6,
    warmup_epochs: int = 4,
    seed: int = 1337,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    if epochs < 1 or warmup_epochs < 0 or warmup_epochs > epochs:
        raise ValueError("Invalid polyphase probe epoch configuration")

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

    generator = LykenoxVocoderGeneratorV2().cpu().train()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=2e-4, weight_decay=1e-5)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, weight_decay=1e-5)

    initial_validation = _validation_reconstruction(generator, val_segments)
    best_validation = initial_validation
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    update_times: list[float] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        order = list(range(len(train_segments)))
        random.Random(seed + epoch).shuffle(order)
        reconstruction_values: list[float] = []
        discriminator_values: list[float] = []
        adversarial_values: list[float] = []
        feature_values: list[float] = []

        for segment_index in order:
            if time.perf_counter() - started >= time_budget_seconds:
                artifact_dir = (
                    root
                    / "models"
                    / "lykenox_identity"
                    / "training"
                    / "vocoder_polyphase_probe"
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "status": "incomplete",
                    "probe_version": PROBE_VERSION,
                    "architecture": VOCODER_GENERATOR_V2_ARCHITECTURE,
                    "epochs_completed": epoch - 1,
                    "global_step": global_step,
                    "initial_validation_reconstruction": initial_validation,
                    "best_validation_reconstruction": best_validation,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "next_gate": "rerun_polyphase_probe",
                }
                report_path = artifact_dir / "probe_report.json"
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                report["report_path"] = str(report_path)
                return report

            segment = train_segments[segment_index]
            mel = segment.mel.unsqueeze(0)
            target = segment.waveform.unsqueeze(0)
            update_started = time.perf_counter()

            if epoch <= warmup_epochs:
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = generator(mel)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                if not torch.isfinite(reconstruction.total):
                    raise RuntimeError("Non-finite polyphase reconstruction loss")
                reconstruction.total.backward()
                grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(grad)):
                    raise RuntimeError("Non-finite polyphase generator gradient")
                generator_optimizer.step()
            else:
                _set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    detached = generator(mel)
                real_output = discriminator(target)
                fake_output = discriminator(detached)
                discriminator_loss = discriminator_hinge_loss(real_output, fake_output)
                discriminator_loss.backward()
                d_grad = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
                if not math.isfinite(float(d_grad)):
                    raise RuntimeError("Non-finite polyphase discriminator gradient")
                discriminator_optimizer.step()

                _set_requires_grad(discriminator, False)
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = generator(mel)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                with torch.no_grad():
                    real_features = discriminator(target)
                fake_features = discriminator(prediction)
                adversarial = generator_adversarial_loss(fake_features)
                feature_match = feature_matching_loss(real_features, fake_features)
                generator_loss = reconstruction.total + 0.10 * adversarial + 2.0 * feature_match
                if not torch.isfinite(generator_loss):
                    raise RuntimeError("Non-finite polyphase adversarial generator loss")
                generator_loss.backward()
                g_grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(g_grad)):
                    raise RuntimeError("Non-finite polyphase adversarial generator gradient")
                generator_optimizer.step()
                _set_requires_grad(discriminator, True)

                discriminator_values.append(float(discriminator_loss.detach().cpu()))
                adversarial_values.append(float(adversarial.detach().cpu()))
                feature_values.append(float(feature_match.detach().cpu()))

            reconstruction_values.append(float(reconstruction.total.detach().cpu()))
            update_times.append(time.perf_counter() - update_started)
            global_step += 1

        validation = _validation_reconstruction(generator, val_segments)
        history.append(
            {
                "epoch": epoch,
                "phase": "reconstruction_warmup" if epoch <= warmup_epochs else "adversarial",
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

    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_polyphase_probe"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "probe_report.json"
    if best_state is None:
        report = {
            "status": "needs_review",
            "device": "cpu",
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V2_ARCHITECTURE,
            "parameters": generator.parameter_count(),
            "epochs_completed": epochs,
            "best_epoch": 0,
            "global_step": global_step,
            "initial_validation_reconstruction": round(initial_validation, 6),
            "best_validation_reconstruction": round(best_validation, 6),
            "validation_improved_over_initial": False,
            "history": history,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "next_gate": "review_polyphase_optimization",
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    generator.load_state_dict(best_state)
    best_generator_path = artifact_dir / "best_probe_generator.pt"
    torch.save(
        {
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V2_ARCHITECTURE,
            "config": generator.config.to_dict(),
            "best_epoch": best_epoch,
            "best_validation_reconstruction": best_validation,
            "generator_state": best_state,
        },
        best_generator_path,
    )
    listening_pairs, frame_lock_count, collapse_count = _write_listening_pairs(
        generator,
        val_segments,
        artifact_dir / "listening",
    )

    validation_improved = best_validation < initial_validation
    automatic_artifact_gate = frame_lock_count == 0 and collapse_count == 0
    status = "pass" if validation_improved and automatic_artifact_gate else "needs_review"
    report = {
        "status": status,
        "device": "cpu",
        "probe_version": PROBE_VERSION,
        "architecture": VOCODER_GENERATOR_V2_ARCHITECTURE,
        "parameters": generator.parameter_count(),
        "epochs_completed": epochs,
        "warmup_epochs": warmup_epochs,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "initial_validation_reconstruction": round(initial_validation, 6),
        "best_validation_reconstruction": round(best_validation, 6),
        "validation_improved_over_initial": validation_improved,
        "frame_rate_lock_count_across_3_generated": frame_lock_count,
        "subbass_or_silence_collapse_count_across_3_generated": collapse_count,
        "automatic_artifact_gate_pass": automatic_artifact_gate,
        "mean_seconds_per_update": round(statistics.fmean(update_times), 4),
        "min_seconds_per_update": round(min(update_times), 4),
        "max_seconds_per_update": round(max(update_times), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "best_probe_generator": str(best_generator_path),
        "listening_pairs": listening_pairs,
        "next_gate": (
            "listen_polyphase_validation_wavs"
            if status == "pass"
            else "review_polyphase_artifacts_before_more_training"
        ),
        "warning": (
            "Automatic diagnostics only reject known v0/v1 failure modes. Human listening "
            "is still required before adopting the v2 architecture."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=96)
    parser.add_argument("--train-items", type=int, default=16)
    parser.add_argument("--val-items", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_polyphase_probe(
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
