"""Bounded architecture-selection probe for LYKENOX vocoder v3.

V0 and free-polyphase v2 both learned a generated-specific 93.75 Hz frame-grid carrier;
resize-conv v1 avoided that carrier but collapsed spectrally. V3 combines a smooth base
branch with zero-mean gated phase residuals and adds a target-referenced periodicity loss.

This probe does not resume any older vocoder checkpoint. It trains from scratch on the
same deterministic LYKENOX mel/wave segments, selects a held-out composite validation
score, writes listening pairs, and applies the differential forensic rule that previously
confirmed the v2 artifact.
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
    LykenoxVocoderGeneratorV3,
    VOCODER_GENERATOR_V3_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_vocoder_data import VocoderSegment, collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_losses import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_periodicity import (
    VOCODER_PERIODICITY_CONTROL_VERSION,
    target_referenced_periodicity_loss,
)
from lykenox_voice_engine.training.speech_vocoder_polyphase_forensic import (
    _hop_periodicity,
    _pitch_metrics,
    _spectral_metrics,
)


PROBE_VERSION = "periodicity-controlled-perceptual-probe-v3"
DEFAULT_TIME_BUDGET_SECONDS = 85.0


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _validation_metrics(
    generator: LykenoxVocoderGeneratorV3,
    segments: list[VocoderSegment],
    *,
    periodicity_weight: float,
) -> tuple[float, float, float]:
    generator.eval()
    reconstruction_values: list[float] = []
    periodicity_values: list[float] = []
    with torch.no_grad():
        for segment in segments:
            prediction = generator(segment.mel.unsqueeze(0))
            target = segment.waveform.unsqueeze(0)
            reconstruction = multi_resolution_reconstruction_loss(prediction, target).total
            periodicity = target_referenced_periodicity_loss(
                prediction,
                target,
                hop_length=generator.config.hop_length,
            ).loss
            reconstruction_value = float(reconstruction.detach().cpu())
            periodicity_value = float(periodicity.detach().cpu())
            if not math.isfinite(reconstruction_value) or not math.isfinite(periodicity_value):
                raise RuntimeError("Non-finite held-out v3 validation metric")
            reconstruction_values.append(reconstruction_value)
            periodicity_values.append(periodicity_value)
    generator.train()
    reconstruction_mean = statistics.fmean(reconstruction_values)
    periodicity_mean = statistics.fmean(periodicity_values)
    score = reconstruction_mean + periodicity_weight * periodicity_mean
    return reconstruction_mean, periodicity_mean, score


def _generated_specific_frame_lock(
    generated: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
) -> tuple[bool, dict[str, object]]:
    generated_pitch = _pitch_metrics(generated, sample_rate)
    reference_pitch = _pitch_metrics(reference, sample_rate)
    generated_hop = _hop_periodicity(generated)
    reference_hop = _hop_periodicity(reference)

    generated_raw = bool(generated_pitch["frame_rate_locked_raw"])
    reference_raw = bool(reference_pitch["frame_rate_locked_raw"])
    hop_excess_delta = (
        float(generated_hop["lag_256_excess"])
        - float(reference_hop["lag_256_excess"])
    )
    confirmed = generated_raw and not reference_raw and hop_excess_delta >= -0.02
    return confirmed, {
        "generated_pitch": generated_pitch,
        "reference_pitch": reference_pitch,
        "generated_hop_periodicity": generated_hop,
        "reference_hop_periodicity": reference_hop,
        "hop_excess_delta_generated_minus_reference": round(hop_excess_delta, 6),
        "generated_specific_frame_lock_confirmed": confirmed,
    }


def _write_listening_pairs(
    generator: LykenoxVocoderGeneratorV3,
    segments: list[VocoderSegment],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    rows: list[dict[str, object]] = []
    confirmed_locks = 0
    collapse_count = 0
    with torch.no_grad():
        for index, segment in enumerate(segments[:3], start=1):
            generated_tensor = generator(segment.mel.unsqueeze(0)).squeeze(0).detach().cpu()
            reference_tensor = segment.waveform.detach().cpu()
            generated = generated_tensor.numpy().astype(np.float64, copy=False)
            reference = reference_tensor.numpy().astype(np.float64, copy=False)

            confirmed, forensic = _generated_specific_frame_lock(
                generated,
                reference,
                generator.config.sample_rate,
            )
            spectral = _spectral_metrics(generated, generator.config.sample_rate)
            confirmed_locks += int(confirmed)
            collapse_count += int(bool(spectral["subbass_or_silence_collapsed"]))

            generated_path = output_dir / f"val_{index:02d}_{segment.utterance_id}_generated.wav"
            reference_path = output_dir / f"val_{index:02d}_{segment.utterance_id}_reference.wav"
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
            rows.append(
                {
                    "utterance_id": segment.utterance_id,
                    "generated": str(generated_path),
                    "reference": str(reference_path),
                    "generated_spectral_diagnostic": spectral,
                    **forensic,
                }
            )
    generator.train()
    return rows, confirmed_locks, collapse_count


def run_periodicity_controlled_probe(
    root: Path,
    *,
    segment_mel_frames: int = 96,
    train_items: int = 16,
    val_items: int = 6,
    epochs: int = 8,
    warmup_epochs: int = 6,
    seed: int = 1337,
    periodicity_weight: float = 0.50,
    adversarial_weight: float = 0.10,
    feature_matching_weight: float = 2.0,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    if epochs < 1 or warmup_epochs < 0 or warmup_epochs > epochs:
        raise ValueError("Invalid v3 probe epoch configuration")
    if periodicity_weight <= 0.0:
        raise ValueError("periodicity_weight must be positive")

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

    generator = LykenoxVocoderGeneratorV3().cpu().train()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=2e-4, weight_decay=1e-5)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, weight_decay=1e-5)

    initial_reconstruction, initial_periodicity, initial_score = _validation_metrics(
        generator,
        val_segments,
        periodicity_weight=periodicity_weight,
    )
    best_reconstruction = initial_reconstruction
    best_periodicity = initial_periodicity
    best_score = initial_score
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    update_times: list[float] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        order = list(range(len(train_segments)))
        random.Random(seed + epoch).shuffle(order)
        reconstruction_values: list[float] = []
        periodicity_values: list[float] = []
        discriminator_values: list[float] = []
        adversarial_values: list[float] = []
        feature_values: list[float] = []

        for segment_index in order:
            if time.perf_counter() - started >= time_budget_seconds:
                artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_periodicity_probe"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "status": "incomplete",
                    "device": "cpu",
                    "probe_version": PROBE_VERSION,
                    "architecture": VOCODER_GENERATOR_V3_ARCHITECTURE,
                    "epochs_completed": epoch - 1,
                    "global_step": global_step,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "next_gate": "rerun_same_v3_probe",
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
                periodicity = target_referenced_periodicity_loss(
                    prediction,
                    target,
                    hop_length=generator.config.hop_length,
                )
                generator_loss = reconstruction.total + periodicity_weight * periodicity.loss
                if not torch.isfinite(generator_loss):
                    raise RuntimeError("Non-finite v3 warm-up generator loss")
                generator_loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(grad)):
                    raise RuntimeError("Non-finite v3 generator gradient")
                generator_optimizer.step()
            else:
                _set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    detached = generator(mel)
                real_output = discriminator(target)
                fake_output = discriminator(detached)
                discriminator_loss = discriminator_hinge_loss(real_output, fake_output)
                if not torch.isfinite(discriminator_loss):
                    raise RuntimeError("Non-finite v3 discriminator loss")
                discriminator_loss.backward()
                d_grad = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
                if not math.isfinite(float(d_grad)):
                    raise RuntimeError("Non-finite v3 discriminator gradient")
                discriminator_optimizer.step()

                _set_requires_grad(discriminator, False)
                generator_optimizer.zero_grad(set_to_none=True)
                prediction = generator(mel)
                reconstruction = multi_resolution_reconstruction_loss(prediction, target)
                periodicity = target_referenced_periodicity_loss(
                    prediction,
                    target,
                    hop_length=generator.config.hop_length,
                )
                with torch.no_grad():
                    real_features = discriminator(target)
                fake_features = discriminator(prediction)
                adversarial = generator_adversarial_loss(fake_features)
                feature_match = feature_matching_loss(real_features, fake_features)
                generator_loss = (
                    reconstruction.total
                    + periodicity_weight * periodicity.loss
                    + adversarial_weight * adversarial
                    + feature_matching_weight * feature_match
                )
                if not torch.isfinite(generator_loss):
                    raise RuntimeError("Non-finite v3 adversarial generator loss")
                generator_loss.backward()
                g_grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                if not math.isfinite(float(g_grad)):
                    raise RuntimeError("Non-finite v3 adversarial generator gradient")
                generator_optimizer.step()
                _set_requires_grad(discriminator, True)

                discriminator_values.append(float(discriminator_loss.detach().cpu()))
                adversarial_values.append(float(adversarial.detach().cpu()))
                feature_values.append(float(feature_match.detach().cpu()))

            reconstruction_values.append(float(reconstruction.total.detach().cpu()))
            periodicity_values.append(float(periodicity.loss.detach().cpu()))
            update_times.append(time.perf_counter() - update_started)
            global_step += 1

        validation_reconstruction, validation_periodicity, validation_score = _validation_metrics(
            generator,
            val_segments,
            periodicity_weight=periodicity_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "phase": "reconstruction_periodicity_warmup" if epoch <= warmup_epochs else "adversarial",
                "train_reconstruction": statistics.fmean(reconstruction_values),
                "train_periodicity_loss": statistics.fmean(periodicity_values),
                "validation_reconstruction": validation_reconstruction,
                "validation_periodicity_loss": validation_periodicity,
                "validation_selection_score": validation_score,
                "discriminator_loss": statistics.fmean(discriminator_values) if discriminator_values else None,
                "generator_adversarial_loss": statistics.fmean(adversarial_values) if adversarial_values else None,
                "feature_matching_loss": statistics.fmean(feature_values) if feature_values else None,
                "phase_scales": generator.phase_scales(),
            }
        )
        if validation_score < best_score:
            best_score = validation_score
            best_reconstruction = validation_reconstruction
            best_periodicity = validation_periodicity
            best_epoch = epoch
            best_state = copy.deepcopy(generator.state_dict())

    artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_periodicity_probe"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "probe_report.json"
    if best_state is None:
        report = {
            "status": "needs_review",
            "device": "cpu",
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V3_ARCHITECTURE,
            "parameters": generator.parameter_count(),
            "epochs_completed": epochs,
            "best_epoch": 0,
            "initial_validation_selection_score": round(initial_score, 6),
            "best_validation_selection_score": round(best_score, 6),
            "next_gate": "review_v3_optimization",
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    generator.load_state_dict(best_state)
    best_generator_path = artifact_dir / "best_probe_generator.pt"
    torch.save(
        {
            "probe_version": PROBE_VERSION,
            "architecture": VOCODER_GENERATOR_V3_ARCHITECTURE,
            "periodicity_control_version": VOCODER_PERIODICITY_CONTROL_VERSION,
            "config": generator.config.to_dict(),
            "best_epoch": best_epoch,
            "best_validation_reconstruction": best_reconstruction,
            "best_validation_periodicity_loss": best_periodicity,
            "best_validation_selection_score": best_score,
            "phase_scales": generator.phase_scales(),
            "generator_state": best_state,
        },
        best_generator_path,
    )

    listening_pairs, confirmed_lock_count, collapse_count = _write_listening_pairs(
        generator,
        val_segments,
        artifact_dir / "listening",
    )
    validation_improved = best_score < initial_score
    automatic_artifact_gate = confirmed_lock_count == 0 and collapse_count == 0
    status = "pass" if validation_improved and automatic_artifact_gate else "needs_review"

    report = {
        "status": status,
        "device": "cpu",
        "probe_version": PROBE_VERSION,
        "architecture": VOCODER_GENERATOR_V3_ARCHITECTURE,
        "periodicity_control_version": VOCODER_PERIODICITY_CONTROL_VERSION,
        "parameters": generator.parameter_count(),
        "epochs_completed": epochs,
        "warmup_epochs": warmup_epochs,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "initial_validation_reconstruction": round(initial_reconstruction, 6),
        "best_validation_reconstruction": round(best_reconstruction, 6),
        "initial_validation_periodicity_loss": round(initial_periodicity, 6),
        "best_validation_periodicity_loss": round(best_periodicity, 6),
        "initial_validation_selection_score": round(initial_score, 6),
        "best_validation_selection_score": round(best_score, 6),
        "validation_selection_improved": validation_improved,
        "confirmed_generated_specific_frame_locks": confirmed_lock_count,
        "subbass_or_silence_collapse_count": collapse_count,
        "automatic_artifact_gate_pass": automatic_artifact_gate,
        "phase_scales": generator.phase_scales(),
        "mean_seconds_per_update": round(statistics.fmean(update_times), 4),
        "min_seconds_per_update": round(min(update_times), 4),
        "max_seconds_per_update": round(max(update_times), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "best_probe_generator": str(best_generator_path),
        "listening_pairs": listening_pairs,
        "next_gate": (
            "listen_v3_validation_wavs"
            if status == "pass"
            else "review_v3_periodicity_or_spectral_failure"
        ),
        "warning": (
            "A pass only clears the known structural artifact gates. Human listening is "
            "still required before v3 can become the persistent vocoder architecture."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=96)
    parser.add_argument("--train-items", type=int, default=16)
    parser.add_argument("--val-items", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--periodicity-weight", type=float, default=0.50)
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_periodicity_controlled_probe(
                args.root,
                segment_mel_frames=args.segment_mel_frames,
                train_items=args.train_items,
                val_items=args.val_items,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                seed=args.seed,
                periodicity_weight=args.periodicity_weight,
                time_budget_seconds=args.time_budget_seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
