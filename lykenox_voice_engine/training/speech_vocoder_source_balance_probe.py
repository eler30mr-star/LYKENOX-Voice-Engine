"""Bounded perceptual gate for LYKENOX source-filter vocoder v4.1.

V4 is the first architecture that cleared the confirmed frame-rate and sub-bass collapse
failures.  Its first listening gate showed a much narrower problem: real harmonic
structure is present, but the fixed source envelope leaves too much energy in the
fundamental/lower bands and too little in the formant/upper-speech bands.

This probe keeps the v4 source-filter contract and tests only the v4.1 balance changes:
mel-conditioned bounded harmonic weights, a 45 Hz DC/subgrave blocker, and a paired
spectral-band balance objective.  It does not resume v4 or any earlier checkpoint.
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
    LykenoxVocoderGeneratorV41,
    VOCODER_GENERATOR_V4_1_ARCHITECTURE,
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
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    SPECTRAL_BANDS_HZ,
    VOCODER_SOURCE_BALANCE_VERSION,
    spectral_band_fractions,
    target_relative_spectral_balance_loss,
)


PROBE_VERSION = "source-balance-perceptual-probe-v4_1"
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
    generator: LykenoxVocoderGeneratorV41,
    item: ConditionedSegment,
) -> torch.Tensor:
    return generator(
        item.segment.mel.unsqueeze(0),
        item.pitch.f0_hz.unsqueeze(0),
        item.pitch.voiced.unsqueeze(0),
    )


def _validation_metrics(
    generator: LykenoxVocoderGeneratorV41,
    items: list[ConditionedSegment],
    *,
    balance_weight: float,
) -> tuple[float, float, float]:
    generator.eval()
    reconstruction_values: list[float] = []
    balance_values: list[float] = []
    with torch.no_grad():
        for item in items:
            prediction = _generate(generator, item)
            target = item.segment.waveform.unsqueeze(0)
            reconstruction = multi_resolution_reconstruction_loss(prediction, target).total
            balance = target_relative_spectral_balance_loss(
                prediction,
                target,
                sample_rate=generator.config.sample_rate,
            ).loss
            reconstruction_value = float(reconstruction.detach().cpu())
            balance_value = float(balance.detach().cpu())
            if not math.isfinite(reconstruction_value) or not math.isfinite(balance_value):
                raise RuntimeError("Non-finite v4.1 held-out validation metric")
            reconstruction_values.append(reconstruction_value)
            balance_values.append(balance_value)
    generator.train()
    reconstruction_mean = statistics.fmean(reconstruction_values)
    balance_mean = statistics.fmean(balance_values)
    selection_score = reconstruction_mean + balance_weight * balance_mean
    return reconstruction_mean, balance_mean, selection_score


def _conditioning_f0_support(
    waveform: np.ndarray,
    item: ConditionedSegment,
    *,
    sample_rate: int,
    hop_length: int,
    frame_length: int = 1024,
) -> float | None:
    """Measure periodic support at the supplied F0 rather than choosing a harmonic peak.

    A generated signal can perceptually support the intended fundamental even when a
    generic pitch detector selects its second or third harmonic.  This diagnostic asks a
    narrower question: does the waveform repeat at the period implied by the known F0?
    """

    voiced_indices = torch.nonzero(item.pitch.voiced > 0.5, as_tuple=False).flatten().tolist()
    if not voiced_indices:
        return None
    half = frame_length // 2
    values: list[float] = []
    for frame_index in voiced_indices:
        f0 = float(item.pitch.f0_hz[frame_index])
        if f0 <= 0.0:
            continue
        lag = int(round(sample_rate / f0))
        center = frame_index * hop_length
        start = max(0, center - half)
        end = min(len(waveform), start + frame_length)
        frame = waveform[start:end]
        if len(frame) <= lag + 8:
            continue
        centered = frame - np.mean(frame)
        left = centered[:-lag]
        right = centered[lag:]
        denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right))) + 1e-12
        values.append(float(np.dot(left, right) / denominator))
    if not values:
        return None
    return float(statistics.fmean(values))


def _band_fraction_dict(fractions: torch.Tensor) -> dict[str, float]:
    values = fractions.detach().cpu().flatten().tolist()
    result: dict[str, float] = {}
    for (low, high), value in zip(SPECTRAL_BANDS_HZ, values, strict=True):
        result[f"{int(low)}_{int(high)}_hz"] = round(float(value), 6)
    return result


def _write_listening_pairs(
    generator: LykenoxVocoderGeneratorV41,
    items: list[ConditionedSegment],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    rows: list[dict[str, object]] = []
    confirmed_locks = 0
    collapse_count = 0
    upper_voice_missing_count = 0

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
            reference_spectral = _spectral_metrics(reference, generator.config.sample_rate)
            generated_pitch = _pitch_metrics(generated, generator.config.sample_rate)
            reference_pitch = _pitch_metrics(reference, generator.config.sample_rate)
            generated_bands = spectral_band_fractions(
                generated_tensor.unsqueeze(0),
                sample_rate=generator.config.sample_rate,
            ).squeeze(0)
            reference_bands = spectral_band_fractions(
                reference_tensor.unsqueeze(0),
                sample_rate=generator.config.sample_rate,
            ).squeeze(0)
            generated_above_300 = float(generated_bands[2:].sum())
            reference_above_300 = float(reference_bands[2:].sum())
            # A deliberately loose target-relative floor.  It rejects the known
            # near-empty upper-band failures without pretending this is a naturalness
            # metric. Human listening remains the real perceptual gate.
            upper_voice_missing = generated_above_300 < max(
                0.03,
                0.15 * reference_above_300,
            )

            confirmed_locks += int(confirmed)
            collapse_count += int(bool(generated_spectral["subbass_or_silence_collapsed"]))
            upper_voice_missing_count += int(upper_voice_missing)

            generated_path = output_dir / f"val_{index:02d}_{item.segment.utterance_id}_generated.wav"
            reference_path = output_dir / f"val_{index:02d}_{item.segment.utterance_id}_reference.wav"
            sf.write(
                str(generated_path),
                generated_tensor.numpy(),
                generator.config.sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                str(reference_path),
                reference_tensor.numpy(),
                generator.config.sample_rate,
                subtype="PCM_16",
            )

            support_generated = _conditioning_f0_support(
                generated,
                item,
                sample_rate=generator.config.sample_rate,
                hop_length=generator.config.hop_length,
            )
            support_reference = _conditioning_f0_support(
                reference,
                item,
                sample_rate=generator.config.sample_rate,
                hop_length=generator.config.hop_length,
            )
            voiced_mask = item.pitch.voiced > 0.5
            median_conditioning_f0 = (
                float(item.pitch.f0_hz[voiced_mask].median())
                if bool(voiced_mask.any())
                else None
            )
            harmonic_weights = generator.harmonic_weight_snapshot(item.segment.mel.unsqueeze(0))

            rows.append(
                {
                    "utterance_id": item.segment.utterance_id,
                    "generated": str(generated_path),
                    "reference": str(reference_path),
                    "voiced_frame_fraction": round(float(item.pitch.voiced.mean()), 6),
                    "median_conditioning_f0_hz": (
                        round(median_conditioning_f0, 4)
                        if median_conditioning_f0 is not None
                        else None
                    ),
                    "generated_conditioning_f0_support": (
                        round(support_generated, 6) if support_generated is not None else None
                    ),
                    "reference_conditioning_f0_support": (
                        round(support_reference, 6) if support_reference is not None else None
                    ),
                    "generated_pitch_detector": generated_pitch,
                    "reference_pitch_detector": reference_pitch,
                    "generated_spectral_diagnostic": generated_spectral,
                    "reference_spectral_diagnostic": reference_spectral,
                    "generated_band_fractions": _band_fraction_dict(generated_bands),
                    "reference_band_fractions": _band_fraction_dict(reference_bands),
                    "generated_above_300hz_fraction": round(generated_above_300, 6),
                    "reference_above_300hz_fraction": round(reference_above_300, 6),
                    "upper_voice_band_missing": bool(upper_voice_missing),
                    "median_harmonic_weights": [round(value, 6) for value in harmonic_weights],
                    **forensic,
                }
            )

    generator.train()
    return rows, confirmed_locks, collapse_count, upper_voice_missing_count


def run_source_balance_probe(
    root: Path,
    *,
    segment_mel_frames: int = 64,
    train_items: int = 12,
    val_items: int = 4,
    epochs: int = 8,
    warmup_epochs: int = 6,
    seed: int = 1337,
    balance_weight: float = 0.50,
    adversarial_weight: float = 0.05,
    feature_matching_weight: float = 1.0,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    if epochs < 1 or warmup_epochs < 0 or warmup_epochs > epochs:
        raise ValueError("Invalid v4.1 probe epoch configuration")
    if balance_weight <= 0.0:
        raise ValueError("balance_weight must be positive")

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
    train_conditioned = _condition_segments(train_segments)
    val_conditioned = _condition_segments(val_segments)

    generator = LykenoxVocoderGeneratorV41().cpu().train()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=2e-4, weight_decay=1e-5)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, weight_decay=1e-5)

    initial_reconstruction, initial_balance, initial_score = _validation_metrics(
        generator,
        val_conditioned,
        balance_weight=balance_weight,
    )
    best_reconstruction = initial_reconstruction
    best_balance = initial_balance
    best_score = initial_score
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    update_times: list[float] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        order = list(range(len(train_conditioned)))
        random.Random(seed + epoch).shuffle(order)
        reconstruction_values: list[float] = []
        balance_values: list[float] = []
        discriminator_values: list[float] = []
        adversarial_values: list[float] = []
        feature_values: list[float] = []

        for item_index in order:
            if time.perf_counter() - started >= time_budget_seconds:
                artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_source_balance_probe"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "status": "incomplete",
                    "device": "cpu",
                    "probe_version": PROBE_VERSION,
                    "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
                    "epochs_completed": epoch - 1,
                    "global_step": global_step,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "next_gate": "rerun_same_v4_1_probe",
                }
                report_path = artifact_dir / "probe_report.json"
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                report["report_path"] = str(report_path)
                return report

            item = train_conditioned[item_index]
            target = item.segment.waveform.unsqueeze(0)
            update_started = time.perf_counter()

            if epoch <= warmup_epochs:
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = _generate(generator, item)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                balance = target_relative_spectral_balance_loss(
                    prediction,
                    target,
                    sample_rate=generator.config.sample_rate,
                )
                generator_loss = reconstruction.total + balance_weight * balance.loss
                if not torch.isfinite(generator_loss):
                    raise RuntimeError("Non-finite v4.1 warm-up generator loss")
                generator_loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(grad)):
                    raise RuntimeError("Non-finite v4.1 generator gradient")
                generator_optimizer.step()
            else:
                _set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    detached = _generate(generator, item)
                real_output = discriminator(target)
                fake_output = discriminator(detached)
                discriminator_loss = discriminator_hinge_loss(real_output, fake_output)
                if not torch.isfinite(discriminator_loss):
                    raise RuntimeError("Non-finite v4.1 discriminator loss")
                discriminator_loss.backward()
                d_grad = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
                if not math.isfinite(float(d_grad)):
                    raise RuntimeError("Non-finite v4.1 discriminator gradient")
                discriminator_optimizer.step()

                _set_requires_grad(discriminator, False)
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = _generate(generator, item)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                balance = target_relative_spectral_balance_loss(
                    prediction,
                    target,
                    sample_rate=generator.config.sample_rate,
                )
                with torch.no_grad():
                    real_features = discriminator(target)
                fake_features = discriminator(prediction)
                adversarial = generator_adversarial_loss(fake_features)
                feature_match = feature_matching_loss(real_features, fake_features)
                generator_loss = (
                    reconstruction.total
                    + balance_weight * balance.loss
                    + adversarial_weight * adversarial
                    + feature_matching_weight * feature_match
                )
                if not torch.isfinite(generator_loss):
                    raise RuntimeError("Non-finite v4.1 adversarial generator loss")
                generator_loss.backward()
                g_grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(g_grad)):
                    raise RuntimeError("Non-finite v4.1 adversarial generator gradient")
                generator_optimizer.step()
                _set_requires_grad(discriminator, True)

                discriminator_values.append(float(discriminator_loss.detach().cpu()))
                adversarial_values.append(float(adversarial.detach().cpu()))
                feature_values.append(float(feature_match.detach().cpu()))

            reconstruction_values.append(float(reconstruction.total.detach().cpu()))
            balance_values.append(float(balance.loss.detach().cpu()))
            update_times.append(time.perf_counter() - update_started)
            global_step += 1

        validation_reconstruction, validation_balance, validation_score = _validation_metrics(
            generator,
            val_conditioned,
            balance_weight=balance_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "phase": "source_balance_warmup" if epoch <= warmup_epochs else "source_balance_adversarial",
                "train_reconstruction": statistics.fmean(reconstruction_values),
                "train_spectral_balance": statistics.fmean(balance_values),
                "validation_reconstruction": validation_reconstruction,
                "validation_spectral_balance": validation_balance,
                "validation_selection_score": validation_score,
                "discriminator_loss": statistics.fmean(discriminator_values) if discriminator_values else None,
                "generator_adversarial_loss": statistics.fmean(adversarial_values) if adversarial_values else None,
                "feature_matching_loss": statistics.fmean(feature_values) if feature_values else None,
                "harmonic_weight_snapshot": generator.harmonic_weight_snapshot(
                    val_conditioned[0].segment.mel.unsqueeze(0)
                ),
            }
        )
        if validation_score < best_score:
            best_reconstruction = validation_reconstruction
            best_balance = validation_balance
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(generator.state_dict())

    artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_source_balance_probe"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "probe_report.json"
    if best_state is None:
        report = {
            "status": "needs_review",
            "device": "cpu",
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
            "parameters": generator.parameter_count(),
            "epochs_completed": epochs,
            "initial_validation_selection_score": round(initial_score, 6),
            "best_validation_selection_score": round(best_score, 6),
            "next_gate": "review_v4_1_optimization",
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    generator.load_state_dict(best_state)
    best_generator_path = artifact_dir / "best_probe_generator.pt"
    torch.save(
        {
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
            "pitch_target_version": PITCH_TARGET_VERSION,
            "source_balance_version": VOCODER_SOURCE_BALANCE_VERSION,
            "config": generator.config.to_dict(),
            "hidden_channels": generator.hidden_channels,
            "harmonics": generator.harmonics,
            "harmonic_log_range": generator.harmonic_log_range,
            "highpass_cutoff_hz": generator.highpass_cutoff_hz,
            "highpass_kernel_size": generator.highpass_kernel_size,
            "best_epoch": best_epoch,
            "best_validation_reconstruction": best_reconstruction,
            "best_validation_spectral_balance": best_balance,
            "best_validation_selection_score": best_score,
            "generator_state": best_state,
        },
        best_generator_path,
    )

    listening_pairs, confirmed_locks, collapse_count, upper_missing_count = _write_listening_pairs(
        generator,
        val_conditioned,
        artifact_dir / "listening",
    )
    selection_improved = best_score < initial_score
    balance_improved = best_balance < initial_balance
    automatic_artifact_gate = (
        confirmed_locks == 0
        and collapse_count == 0
        and upper_missing_count == 0
    )
    status = "pass" if selection_improved and balance_improved and automatic_artifact_gate else "needs_review"

    report = {
        "status": status,
        "device": "cpu",
        "probe_version": PROBE_VERSION,
        "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "source_balance_version": VOCODER_SOURCE_BALANCE_VERSION,
        "parameters": generator.parameter_count(),
        "epochs_completed": epochs,
        "warmup_epochs": warmup_epochs,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "segment_mel_frames": segment_mel_frames,
        "segment_audio_seconds": round(
            segment_mel_frames * generator.config.hop_length / generator.config.sample_rate,
            4,
        ),
        "highpass_cutoff_hz": generator.highpass_cutoff_hz,
        "initial_validation_reconstruction": round(initial_reconstruction, 6),
        "best_validation_reconstruction": round(best_reconstruction, 6),
        "initial_validation_spectral_balance": round(initial_balance, 6),
        "best_validation_spectral_balance": round(best_balance, 6),
        "initial_validation_selection_score": round(initial_score, 6),
        "best_validation_selection_score": round(best_score, 6),
        "validation_selection_improved": selection_improved,
        "validation_spectral_balance_improved": balance_improved,
        "confirmed_generated_specific_frame_locks": confirmed_locks,
        "subbass_or_silence_collapse_count": collapse_count,
        "upper_voice_band_missing_count": upper_missing_count,
        "automatic_artifact_gate_pass": automatic_artifact_gate,
        "mean_seconds_per_update": round(statistics.fmean(update_times), 4),
        "min_seconds_per_update": round(min(update_times), 4),
        "max_seconds_per_update": round(max(update_times), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "best_probe_generator": str(best_generator_path),
        "listening_pairs": listening_pairs,
        "next_gate": (
            "listen_v4_1_source_balance_validation_wavs"
            if status == "pass"
            else "review_v4_1_source_balance_failure_before_more_training"
        ),
        "warning": (
            "A pass only clears bounded numeric/known-artifact gates. The v4.1 architecture "
            "is not accepted until generated/reference WAVs contain recognizable speech "
            "with materially improved timbre and upper-band structure."
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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--balance-weight", type=float, default=0.50)
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_source_balance_probe(
                args.root,
                segment_mel_frames=args.segment_mel_frames,
                train_items=args.train_items,
                val_items=args.val_items,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                seed=args.seed,
                balance_weight=args.balance_weight,
                time_budget_seconds=args.time_budget_seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
