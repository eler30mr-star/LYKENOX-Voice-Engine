"""Bounded real-data architecture gate for the LYKENOX V7 vocoder.

This smoke performs no persistent training. It verifies the source-free structural contract,
finite gradients, direct mel-content trainability on one real held-out-compatible segment,
and checkpoint immutability before any resumable V7 trainer is allowed to exist.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV7,
    VOCODER_GENERATOR_V7_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_losses import (
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_v7_content_loss import (
    VOCODER_V7_CONTENT_LOSS_VERSION,
    V7MelContentConsistencyLoss,
)


SMOKE_VERSION = "vocoder-v7-source-free-content-smoke-v1"


def _sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_paths(root: Path) -> dict[str, Path]:
    base = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2_best": base / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_prior_last": base / "vocoder_direct_waveform_v6" / "last.pt",
        "v6_prior_best": base / "vocoder_direct_waveform_v6" / "best.pt",
        "v6_clarity_last": base / "vocoder_direct_waveform_v6_clarity_guard_v1" / "last.pt",
        "v6_clarity_best": base / "vocoder_direct_waveform_v6_clarity_guard_v1" / "best.pt",
    }


def _source_contract(model: LykenoxVocoderGeneratorV7) -> dict[str, bool]:
    implementation = (
        inspect.getsource(LykenoxVocoderGeneratorV7._frame_latent)
        + inspect.getsource(LykenoxVocoderGeneratorV7.forward)
    )
    forbidden_calls_absent = all(
        token not in implementation
        for token in (
            "cumsum(",
            "torch.sin(",
            "torch.cos(",
            "randn(",
            "torch.rand(",
            "remainder(",
        )
    )
    return {
        "source_free": bool(model.source_free),
        "no_explicit_source": not bool(model.explicit_source),
        "no_sinusoidal_carrier": not bool(model.explicit_sinusoidal_carrier),
        "no_harmonic_bank": int(model.deterministic_harmonics) == 0,
        "no_voiced_noise_source": not bool(model.voiced_noise_source),
        "no_deterministic_noise_conditioning": not bool(model.deterministic_noise_conditioning),
        "no_raw_source_bypass": not bool(model.raw_source_bypass),
        "no_sample_phase_conditioning": not bool(model.sample_phase_conditioning),
        "no_sample_rate_pitch_features": not bool(model.sample_rate_pitch_features),
        "frame_latent_pitch_only": model.pitch_conditioning_scope == "frame_latent_only",
        "no_local_unit_rms_shape_normalization": not bool(model.local_unit_rms_shape_normalization),
        "no_global_unit_rms_shape_normalization": not bool(model.global_unit_rms_shape_normalization),
        "no_level_rescue_branch": not bool(model.level_rescue_branch),
        "no_posthoc_gain_normalization": not bool(model.posthoc_gain_normalization),
        "forbidden_sample_source_calls_absent": forbidden_calls_absent,
    }


def _metrics(
    model: LykenoxVocoderGeneratorV7,
    content_loss: V7MelContentConsistencyLoss,
    mel: torch.Tensor,
    f0: torch.Tensor,
    voiced: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = model(mel, f0, voiced)
    reconstruction = multi_resolution_reconstruction_loss(prediction, target).total
    content = content_loss(prediction, mel)
    total = reconstruction + 0.75 * content.total
    return total, {
        "total": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "content": float(content.total.detach()),
        "content_log_mel": float(content.log_mel_l1.detach()),
        "content_centered_shape": float(content.centered_shape_l1.detach()),
        "content_spectral_delta": float(content.spectral_delta_l1.detach()),
        "content_temporal_delta": float(content.temporal_delta_l1.detach()),
        "content_temporal_acceleration": float(content.temporal_acceleration_l1.detach()),
        "prediction_rms": float(torch.sqrt(prediction.square().mean().clamp_min(1e-12)).detach()),
        "target_rms": float(torch.sqrt(target.square().mean().clamp_min(1e-12)).detach()),
    }


def run_v7_architecture_smoke(
    root: Path,
    *,
    segment_mel_frames: int = 24,
    updates: int = 6,
    seed: int = 77000,
) -> dict[str, object]:
    if segment_mel_frames < 16 or updates < 1:
        raise ValueError("invalid v7 smoke bounds")
    root = Path(root).resolve()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.manual_seed(seed)

    protected = _protected_paths(root)
    before = {name: _sha256_if_exists(path) for name, path in protected.items()}

    segments, skipped = collect_vocoder_segments(
        root,
        "train",
        segment_mel_frames=segment_mel_frames,
        max_items=1,
        seed=seed,
    )
    segment = segments[0]
    pitch = extract_pitch_frames(segment.waveform, frame_count=segment.mel_frames)
    mel = segment.mel.unsqueeze(0)
    f0 = pitch.f0_hz.unsqueeze(0)
    voiced = pitch.voiced.unsqueeze(0)
    target = segment.waveform.unsqueeze(0)

    model = LykenoxVocoderGeneratorV7(
        frame_channels=96,
        upsample_channels=(80, 56, 40),
        residual_kernels=(3, 7),
        residual_dilations=(1, 3),
    ).cpu()
    content_loss = V7MelContentConsistencyLoss().cpu()
    contract = _source_contract(model)

    model.eval()
    with torch.no_grad():
        initial_prediction = model(mel, f0, voiced)
        expected_samples = segment_mel_frames * model.config.hop_length
        output_length_exact = tuple(initial_prediction.shape) == (1, expected_samples)
        mel_changed = mel.roll(shifts=1, dims=1)
        changed_mel_prediction = model(mel_changed, f0, voiced)
        mel_dependency = float((initial_prediction - changed_mel_prediction).abs().mean())
        changed_f0_prediction = model(mel, f0 * 1.15, voiced)
        frame_pitch_dependency = float((initial_prediction - changed_f0_prediction).abs().mean())
        _, initial = _metrics(model, content_loss, mel, f0, voiced, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    best = dict(initial)
    finite_gradients = True
    model.train()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        total, _ = _metrics(model, content_loss, mel, f0, voiced, target)
        if not bool(torch.isfinite(total)):
            finite_gradients = False
            break
        total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            finite_gradients = False
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, current = _metrics(model, content_loss, mel, f0, voiced, target)
        model.train()
        if current["total"] < best["total"]:
            best = dict(current)

    model.eval()
    with torch.no_grad():
        started = time.perf_counter()
        benchmark_runs = 2
        for _ in range(benchmark_runs):
            model(mel, f0, voiced)
        elapsed = time.perf_counter() - started
    audio_seconds = target.shape[-1] / float(model.config.sample_rate)
    benchmark_rtf = (elapsed / benchmark_runs) / max(audio_seconds, 1e-8)

    after = {name: _sha256_if_exists(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    structural_gate_pass = all(contract.values()) and output_length_exact
    total_decreased = best["total"] < initial["total"]
    reconstruction_decreased = best["reconstruction"] < initial["reconstruction"]
    content_decreased = best["content"] < initial["content"]
    dependency_gate_pass = mel_dependency > 1e-7 and frame_pitch_dependency > 1e-9
    trainability_gate_pass = (
        finite_gradients
        and total_decreased
        and reconstruction_decreased
        and content_decreased
    )
    status = "pass" if (
        structural_gate_pass
        and dependency_gate_pass
        and trainability_gate_pass
        and checkpoints_unchanged
    ) else "fail"

    return {
        "status": status,
        "smoke_version": SMOKE_VERSION,
        "architecture": VOCODER_GENERATOR_V7_ARCHITECTURE,
        "content_loss_version": VOCODER_V7_CONTENT_LOSS_VERSION,
        **contract,
        "output_length_exact": output_length_exact,
        "finite_gradients": finite_gradients,
        "mel_dependency_mean_abs": round(mel_dependency, 9),
        "frame_pitch_dependency_mean_abs": round(frame_pitch_dependency, 9),
        "dependency_gate_pass": dependency_gate_pass,
        "initial": {key: round(value, 6) for key, value in initial.items()},
        "best": {key: round(value, 6) for key, value in best.items()},
        "total_decreased": total_decreased,
        "reconstruction_decreased": reconstruction_decreased,
        "content_decreased": content_decreased,
        "structural_gate_pass": structural_gate_pass,
        "trainability_gate_pass": trainability_gate_pass,
        "benchmark_rtf": round(float(benchmark_rtf), 4),
        "cpu_candidate": bool(benchmark_rtf < 2.5),
        "segment_utterance_id": segment.utterance_id,
        "segment_start_frame": segment.start_frame,
        "segment_mel_frames": segment.mel_frames,
        "skipped_items": len(skipped),
        "protected_checkpoints_present": {
            name: value is not None for name, value in before.items()
        },
        "protected_checkpoints_unchanged": checkpoints_unchanged,
        "persistent_training_started": False,
        "full_utterance_perceptual_acceptance": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "next_gate": (
            "build_v7_bounded_resumable_training_candidate"
            if status == "pass"
            else "reject_or_revise_v7_before_persistent_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=24)
    parser.add_argument("--updates", type=int, default=6)
    args = parser.parse_args()
    print(
        json.dumps(
            run_v7_architecture_smoke(
                args.root,
                segment_mel_frames=args.segment_mel_frames,
                updates=args.updates,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
