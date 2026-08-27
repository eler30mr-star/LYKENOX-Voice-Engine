"""Persistent vocoder-training contract gate for LYKENOX Speech.

This is a bounded CPU smoke, not a long vocoder run. It validates the pieces required
before committing hours of training:

- deterministic train/validation mel-waveform segment pairing
- spectral reconstruction warm-up
- LYKENOX-owned multi-scale adversarial discriminator
- feature matching for perceptual pressure
- finite generator/discriminator updates
- held-out validation reconstruction measurement
- exact generator waveform-length contract
- resumable checkpoint/provenance round-trip for both networks and optimizers

A pass means the training recipe and artifact boundary are mechanically coherent. It does
not mean the resulting few-step generator sounds good.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderConfig,
    LykenoxVocoderGenerator,
)
from lykenox_voice_engine.training.speech_vocoder_artifact import (
    build_vocoder_training_provenance,
    load_vocoder_checkpoint,
    save_vocoder_checkpoint,
)
from lykenox_voice_engine.training.speech_vocoder_data import (
    VOCODER_SEGMENT_CONTRACT_VERSION,
    VocoderSegment,
    collect_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_losses import (
    VOCODER_LOSS_RECIPE_VERSION,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _reconstruction_value(
    generator: LykenoxVocoderGenerator,
    segments: list[VocoderSegment],
) -> float:
    generator.eval()
    values: list[float] = []
    with torch.no_grad():
        for segment in segments:
            prediction = generator(segment.mel.unsqueeze(0))
            target = segment.waveform.unsqueeze(0)
            loss = multi_resolution_reconstruction_loss(prediction, target).total
            values.append(float(loss.detach().cpu()))
    generator.train()
    return statistics.fmean(values)


def _exact_length_check(
    generator: LykenoxVocoderGenerator,
    segments: list[VocoderSegment],
) -> bool:
    generator.eval()
    with torch.no_grad():
        for segment in segments:
            waveform = generator(segment.mel.unsqueeze(0))
            expected = segment.mel_frames * generator.config.hop_length
            if tuple(waveform.shape) != (1, expected):
                generator.train()
                return False
    generator.train()
    return True


def _validation_determinism(
    root: Path,
    *,
    segment_mel_frames: int,
    max_items: int,
    seed: int,
    original: list[VocoderSegment],
) -> bool:
    repeated, _ = collect_vocoder_segments(
        root,
        "val",
        segment_mel_frames=segment_mel_frames,
        max_items=max_items,
        seed=seed,
    )
    if len(repeated) != len(original):
        return False
    for first, second in zip(original, repeated, strict=True):
        if (
            first.utterance_id != second.utterance_id
            or first.start_frame != second.start_frame
            or not torch.equal(first.mel, second.mel)
            or not torch.equal(first.waveform, second.waveform)
        ):
            return False
    return True


def run_vocoder_training_contract_smoke(
    root: Path,
    *,
    segment_mel_frames: int = 64,
    train_items: int = 2,
    val_items: int = 2,
    reconstruction_steps: int = 4,
    adversarial_steps: int = 2,
    seed: int = 1337,
) -> dict[str, object]:
    if train_items < 1 or val_items < 1:
        raise ValueError("train_items and val_items must be positive")
    if reconstruction_steps < 1 or adversarial_steps < 1:
        raise ValueError("both reconstruction and adversarial smoke stages must run")

    root = Path(root).resolve()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_segments, train_skipped = collect_vocoder_segments(
        root,
        "train",
        segment_mel_frames=segment_mel_frames,
        max_items=train_items,
        seed=seed,
    )
    val_segments, val_skipped = collect_vocoder_segments(
        root,
        "val",
        segment_mel_frames=segment_mel_frames,
        max_items=val_items,
        seed=seed,
    )
    validation_segment_deterministic = _validation_determinism(
        root,
        segment_mel_frames=segment_mel_frames,
        max_items=val_items,
        seed=seed,
        original=val_segments,
    )

    config = LykenoxVocoderConfig()
    generator = LykenoxVocoderGenerator(config).cpu().train()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=2e-4, weight_decay=1e-5
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=2e-4, weight_decay=1e-5
    )

    exact_waveform_length = _exact_length_check(
        generator,
        train_segments + val_segments,
    )
    train_probe_before = _reconstruction_value(generator, [train_segments[0]])
    validation_before = _reconstruction_value(generator, val_segments)

    reconstruction_losses: list[float] = []
    discriminator_losses: list[float] = []
    adversarial_generator_losses: list[float] = []
    feature_matching_losses: list[float] = []
    timings: list[float] = []
    max_generator_gradient_norm = 0.0
    max_discriminator_gradient_norm = 0.0

    # Stage A: stable spectral warm-up. Persistent training will use a longer warm-up
    # before enabling adversarial pressure.
    for step in range(reconstruction_steps):
        segment = train_segments[step % len(train_segments)]
        started = time.perf_counter()
        generator_optimizer.zero_grad(set_to_none=True)
        prediction = generator(segment.mel.unsqueeze(0))
        target = segment.waveform.unsqueeze(0)
        reconstruction = multi_resolution_reconstruction_loss(prediction, target)
        if not torch.isfinite(reconstruction.total):
            raise RuntimeError(f"Non-finite reconstruction loss at warm-up step {step}")
        reconstruction.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"Non-finite generator gradient at warm-up step {step}")
        generator_optimizer.step()
        timings.append(time.perf_counter() - started)
        reconstruction_losses.append(float(reconstruction.total.detach().cpu()))
        max_generator_gradient_norm = max(
            max_generator_gradient_norm,
            float(grad_norm),
        )

    # Stage B: small adversarial/feature-matching exercise. The reconstruction objective
    # remains dominant so this bounded gate does not destabilize the probe generator.
    for step in range(adversarial_steps):
        segment = train_segments[step % len(train_segments)]
        mel = segment.mel.unsqueeze(0)
        target = segment.waveform.unsqueeze(0)
        started = time.perf_counter()

        _set_requires_grad(discriminator, True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_detached = generator(mel)
        real_output = discriminator(target)
        fake_output = discriminator(fake_detached)
        discriminator_loss = discriminator_hinge_loss(real_output, fake_output)
        if not torch.isfinite(discriminator_loss):
            raise RuntimeError(f"Non-finite discriminator loss at step {step}")
        discriminator_loss.backward()
        discriminator_grad = torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(), 10.0
        )
        if not math.isfinite(float(discriminator_grad)):
            raise RuntimeError(f"Non-finite discriminator gradient at step {step}")
        discriminator_optimizer.step()

        _set_requires_grad(discriminator, False)
        generator_optimizer.zero_grad(set_to_none=True)
        fake = generator(mel)
        reconstruction = multi_resolution_reconstruction_loss(fake, target)
        with torch.no_grad():
            real_features = discriminator(target)
        fake_features = discriminator(fake)
        adversarial = generator_adversarial_loss(fake_features)
        feature_match = feature_matching_loss(real_features, fake_features)
        generator_loss = reconstruction.total + 0.10 * adversarial + 2.0 * feature_match
        if not torch.isfinite(generator_loss):
            raise RuntimeError(f"Non-finite adversarial generator loss at step {step}")
        generator_loss.backward()
        generator_grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
        if not math.isfinite(float(generator_grad)):
            raise RuntimeError(f"Non-finite adversarial generator gradient at step {step}")
        generator_optimizer.step()
        _set_requires_grad(discriminator, True)

        timings.append(time.perf_counter() - started)
        reconstruction_losses.append(float(reconstruction.total.detach().cpu()))
        discriminator_losses.append(float(discriminator_loss.detach().cpu()))
        adversarial_generator_losses.append(float(adversarial.detach().cpu()))
        feature_matching_losses.append(float(feature_match.detach().cpu()))
        max_discriminator_gradient_norm = max(
            max_discriminator_gradient_norm,
            float(discriminator_grad),
        )
        max_generator_gradient_norm = max(
            max_generator_gradient_norm,
            float(generator_grad),
        )

    train_probe_after = _reconstruction_value(generator, [train_segments[0]])
    validation_after = _reconstruction_value(generator, val_segments)
    train_probe_decreased = train_probe_after < train_probe_before
    validation_finite = math.isfinite(validation_before) and math.isfinite(validation_after)

    provenance = build_vocoder_training_provenance(
        root,
        segment_mel_frames=segment_mel_frames,
        seed=seed,
    )
    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_training_contract_smoke"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "roundtrip.pt"
    save_vocoder_checkpoint(
        checkpoint_path,
        generator,
        discriminator,
        epoch=0,
        global_step=reconstruction_steps + adversarial_steps,
        validation_reconstruction_loss=validation_after,
        training_provenance=provenance,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        training_metadata={
            "purpose": "vocoder_training_contract_smoke_only",
            "reconstruction_steps": reconstruction_steps,
            "adversarial_steps": adversarial_steps,
            "recipe": "spectral_warmup_then_light_adversarial_feature_matching",
        },
    )
    restored_generator, restored_discriminator, payload = load_vocoder_checkpoint(
        checkpoint_path
    )

    probe_mel = val_segments[0].mel.unsqueeze(0)
    probe_wave = val_segments[0].waveform.unsqueeze(0)
    generator.eval()
    discriminator.eval()
    restored_generator.eval()
    restored_discriminator.eval()
    with torch.no_grad():
        original_wave = generator(probe_mel)
        restored_wave = restored_generator(probe_mel)
        original_scores = discriminator(probe_wave).scores
        restored_scores = restored_discriminator(probe_wave).scores
    generator_roundtrip_delta = float(
        torch.max(torch.abs(original_wave - restored_wave)).detach().cpu()
    )
    discriminator_roundtrip_delta = max(
        float(torch.max(torch.abs(a - b)).detach().cpu())
        for a, b in zip(original_scores, restored_scores, strict=True)
    )
    checkpoint_roundtrip_exact = (
        generator_roundtrip_delta == 0.0 and discriminator_roundtrip_delta == 0.0
    )
    provenance_exact = payload.get("training_provenance") == provenance
    optimizer_states_present = (
        payload.get("generator_optimizer_state") is not None
        and payload.get("discriminator_optimizer_state") is not None
    )

    gate_pass = all(
        (
            exact_waveform_length,
            validation_segment_deterministic,
            train_probe_decreased,
            validation_finite,
            bool(discriminator_losses),
            checkpoint_roundtrip_exact,
            provenance_exact,
            optimizer_states_present,
        )
    )

    report = {
        "status": "pass" if gate_pass else "needs_review",
        "device": "cpu",
        "generator_architecture": "lykenox_compact_transposed_conv_v0",
        "generator_parameters": generator.parameter_count(),
        "discriminator_architecture": restored_discriminator.architecture,
        "discriminator_parameters": discriminator.parameter_count(),
        "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
        "loss_recipe_version": VOCODER_LOSS_RECIPE_VERSION,
        "segment_mel_frames": segment_mel_frames,
        "segment_audio_seconds": round(
            segment_mel_frames * config.hop_length / config.sample_rate,
            4,
        ),
        "train_segments": len(train_segments),
        "val_segments": len(val_segments),
        "train_skipped_before_selection": len(train_skipped),
        "val_skipped_before_selection": len(val_skipped),
        "validation_segment_deterministic": validation_segment_deterministic,
        "exact_waveform_length": exact_waveform_length,
        "reconstruction_steps": reconstruction_steps,
        "adversarial_steps": adversarial_steps,
        "train_probe_reconstruction_before": round(train_probe_before, 6),
        "train_probe_reconstruction_after": round(train_probe_after, 6),
        "train_probe_reconstruction_decreased": train_probe_decreased,
        "validation_reconstruction_before": round(validation_before, 6),
        "validation_reconstruction_after": round(validation_after, 6),
        "validation_reconstruction_finite": validation_finite,
        "first_step_reconstruction_loss": round(reconstruction_losses[0], 6),
        "last_step_reconstruction_loss": round(reconstruction_losses[-1], 6),
        "last_discriminator_loss": round(discriminator_losses[-1], 6),
        "last_generator_adversarial_loss": round(adversarial_generator_losses[-1], 6),
        "last_feature_matching_loss": round(feature_matching_losses[-1], 6),
        "mean_seconds_per_update_pair": round(statistics.fmean(timings), 4),
        "min_seconds_per_update_pair": round(min(timings), 4),
        "max_seconds_per_update_pair": round(max(timings), 4),
        "max_generator_gradient_norm": round(max_generator_gradient_norm, 6),
        "max_discriminator_gradient_norm": round(max_discriminator_gradient_norm, 6),
        "checkpoint": str(checkpoint_path),
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "generator_roundtrip_max_abs_delta": generator_roundtrip_delta,
        "discriminator_roundtrip_max_abs_delta": discriminator_roundtrip_delta,
        "checkpoint_provenance_exact": provenance_exact,
        "optimizer_states_present": optimizer_states_present,
        "training_recipe": {
            "stage_a": "multi_resolution_spectral_reconstruction_warmup",
            "stage_b": "reconstruction_plus_light_hinge_adversarial_plus_feature_matching",
            "validation_metric": "held_out_multi_resolution_reconstruction_loss",
            "best_checkpoint_rule": "lowest_held_out_reconstruction_loss_with_finite_adversarial_metrics",
            "runtime_requirement": "generator_only",
        },
        "next_gate": (
            "persistent_vocoder_short_training"
            if gate_pass
            else "fix_persistent_vocoder_training_contract"
        ),
        "warning": (
            "This gate validates the persistent training contract only. Perceptual quality "
            "must be judged after a bounded persistent train/validation run; the discriminator "
            "is training-only and will not ship in the LYKENOX runtime."
        ),
    }
    report_path = artifact_dir / "contract_smoke_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=64)
    parser.add_argument("--train-items", type=int, default=2)
    parser.add_argument("--val-items", type=int, default=2)
    parser.add_argument("--reconstruction-steps", type=int, default=4)
    parser.add_argument("--adversarial-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    print(
        json.dumps(
            run_vocoder_training_contract_smoke(
                args.root,
                segment_mel_frames=args.segment_mel_frames,
                train_items=args.train_items,
                val_items=args.val_items,
                reconstruction_steps=args.reconstruction_steps,
                adversarial_steps=args.adversarial_steps,
                seed=args.seed,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
