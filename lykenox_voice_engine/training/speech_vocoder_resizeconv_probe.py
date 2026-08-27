"""Bounded perceptual probe for LYKENOX resize-convolution vocoder v1.

The transposed-convolution v0 passed numeric training gates but failed the first human
listening gate with a frame-rate-locked buzz.  This command tests the architectural fix
without touching or resuming v0 checkpoints.  It trains v1 briefly on the same owned
mel/wave contract, selects the lowest held-out reconstruction epoch, and writes three
held-out generated/reference WAV pairs for listening.

This is an architecture-selection probe, not final vocoder training.
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
    LykenoxVocoderGeneratorV1,
    VOCODER_GENERATOR_V1_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_vocoder_data import VocoderSegment, collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_losses import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)


DEFAULT_TIME_BUDGET_SECONDS = 85.0
PROBE_VERSION = "resizeconv-perceptual-probe-v1"


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _validation_reconstruction(
    generator: LykenoxVocoderGeneratorV1,
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
                raise RuntimeError("Non-finite resize-conv validation loss")
            values.append(value)
    generator.train()
    return statistics.fmean(values)


def _frame_rate_lock_metrics(waveform: torch.Tensor, sample_rate: int, hop_length: int) -> dict[str, float | bool | None]:
    """Detect the exact mel-frame-rate pitch lock seen in the failed v0 listening WAVs."""

    y = waveform.detach().cpu().numpy().astype(np.float64, copy=False)
    frame_length = 1024
    analysis_hop = 256
    min_lag = max(1, int(sample_rate / 300.0))
    max_lag = int(sample_rate / 60.0)
    estimates: list[float] = []
    rms_values: list[float] = []
    frames: list[np.ndarray] = []
    for start in range(0, max(0, len(y) - frame_length + 1), analysis_hop):
        frame = y[start : start + frame_length]
        rms = float(np.sqrt(np.mean(frame * frame)))
        rms_values.append(rms)
        frames.append(frame)
    if not frames:
        return {"median_pitch_hz": None, "pitch_std_hz": None, "frame_rate_hz": sample_rate / hop_length, "frame_rate_locked": False}
    threshold = max(rms_values) * 0.20
    for frame, rms in zip(frames, rms_values, strict=True):
        if rms < threshold:
            continue
        centered = frame - np.mean(frame)
        best_lag = None
        best_score = -1.0
        energy = float(np.dot(centered, centered)) + 1e-12
        for lag in range(min_lag, min(max_lag, frame_length - 2) + 1):
            score = float(np.dot(centered[:-lag], centered[lag:]) / energy)
            if score > best_score:
                best_score = score
                best_lag = lag
        if best_lag is not None and best_score > 0.15:
            estimates.append(sample_rate / best_lag)
    frame_rate = sample_rate / hop_length
    if not estimates:
        return {"median_pitch_hz": None, "pitch_std_hz": None, "frame_rate_hz": frame_rate, "frame_rate_locked": False}
    median = float(np.median(estimates))
    std = float(np.std(estimates))
    locked = abs(median - frame_rate) <= 0.75 and std <= 1.0
    return {
        "median_pitch_hz": round(median, 4),
        "pitch_std_hz": round(std, 4),
        "frame_rate_hz": round(frame_rate, 4),
        "frame_rate_locked": bool(locked),
    }


def _write_listening_pairs(
    generator: LykenoxVocoderGeneratorV1,
    segments: list[VocoderSegment],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    rows: list[dict[str, object]] = []
    lock_count = 0
    with torch.no_grad():
        for index, segment in enumerate(segments[:3], start=1):
            generated = generator(segment.mel.unsqueeze(0)).squeeze(0).detach().cpu()
            reference = segment.waveform.detach().cpu()
            metrics = _frame_rate_lock_metrics(
                generated,
                generator.config.sample_rate,
                generator.config.hop_length,
            )
            lock_count += int(bool(metrics["frame_rate_locked"]))
            generated_path = output_dir / f"val_{index:02d}_{segment.utterance_id}_generated.wav"
            reference_path = output_dir / f"val_{index:02d}_{segment.utterance_id}_reference.wav"
            sf.write(str(generated_path), generated.numpy(), generator.config.sample_rate, subtype="PCM_16")
            sf.write(str(reference_path), reference.numpy(), generator.config.sample_rate, subtype="PCM_16")
            rows.append(
                {
                    "utterance_id": segment.utterance_id,
                    "generated": str(generated_path),
                    "reference": str(reference_path),
                    "generated_pitch_diagnostic": metrics,
                }
            )
    generator.train()
    return rows, lock_count


def run_resizeconv_probe(
    root: Path,
    *,
    segment_mel_frames: int = 96,
    train_items: int = 16,
    val_items: int = 6,
    epochs: int = 7,
    warmup_epochs: int = 2,
    seed: int = 1337,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    root = Path(root).resolve()
    started = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_segments, _ = collect_vocoder_segments(
        root, "train", segment_mel_frames=segment_mel_frames, max_items=train_items, seed=seed
    )
    val_segments, _ = collect_vocoder_segments(
        root, "val", segment_mel_frames=segment_mel_frames, max_items=val_items, seed=seed
    )

    generator = LykenoxVocoderGeneratorV1().cpu().train()
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
                report = {
                    "status": "incomplete",
                    "architecture": VOCODER_GENERATOR_V1_ARCHITECTURE,
                    "epochs_completed": epoch - 1,
                    "global_step": global_step,
                    "initial_validation_reconstruction": initial_validation,
                    "best_validation_reconstruction": best_validation,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "next_gate": "rerun_resizeconv_probe",
                }
                artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_resizeconv_probe"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                (artifact_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                return report

            segment = train_segments[segment_index]
            mel = segment.mel.unsqueeze(0)
            target = segment.waveform.unsqueeze(0)
            update_started = time.perf_counter()

            if epoch <= warmup_epochs:
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = generator(mel)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                reconstruction.total.backward()
                grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(grad)):
                    raise RuntimeError("Non-finite resize-conv generator gradient")
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
                    raise RuntimeError("Non-finite resize-conv discriminator gradient")
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
                generator_loss.backward()
                g_grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(g_grad)):
                    raise RuntimeError("Non-finite resize-conv adversarial generator gradient")
                generator_optimizer.step()
                _set_requires_grad(discriminator, True)

                discriminator_values.append(float(discriminator_loss.detach().cpu()))
                adversarial_values.append(float(adversarial.detach().cpu()))
                feature_values.append(float(feature_match.detach().cpu()))

            reconstruction_values.append(float(reconstruction.total.detach().cpu()))
            update_times.append(time.perf_counter() - update_started)
            global_step += 1

        validation = _validation_reconstruction(generator, val_segments)
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(generator.state_dict())
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

    if best_state is None:
        raise RuntimeError("Resize-conv probe did not improve over random initialization")
    generator.load_state_dict(best_state)

    artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_resizeconv_probe"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifact_dir / "best_probe_generator.pt"
    torch.save(
        {
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V1_ARCHITECTURE,
            "config": generator.config.to_dict(),
            "best_epoch": best_epoch,
            "validation_reconstruction": best_validation,
            "generator_state": generator.state_dict(),
        },
        state_path,
    )
    listening_pairs, frame_rate_lock_count = _write_listening_pairs(
        generator,
        val_segments,
        artifact_dir / "listening",
    )

    report = {
        "status": "pass",
        "device": "cpu",
        "probe_version": PROBE_VERSION,
        "architecture": VOCODER_GENERATOR_V1_ARCHITECTURE,
        "parameters": generator.parameter_count(),
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "initial_validation_reconstruction": round(initial_validation, 6),
        "best_validation_reconstruction": round(best_validation, 6),
        "validation_improved_over_initial": best_validation < initial_validation,
        "frame_rate_lock_count_across_3_generated": frame_rate_lock_count,
        "mean_seconds_per_update": round(statistics.fmean(update_times), 4),
        "min_seconds_per_update": round(min(update_times), 4),
        "max_seconds_per_update": round(max(update_times), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "best_probe_generator": str(state_path),
        "listening_pairs": listening_pairs,
        "next_gate": "listen_resizeconv_validation_wavs",
        "warning": "Numeric pass does not accept v1. The listening pairs decide whether the frame-rate buzz was removed.",
    }
    report_path = artifact_dir / "probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    print(json.dumps(run_resizeconv_probe(args.root, time_budget_seconds=args.time_budget_seconds), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
