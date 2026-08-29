"""Bounded real-data architecture smoke for LYKENOX vocoder v4.4.

This gate follows the v4.3 perceptual rejection and carrier fine-structure ablation.  The
8-harmonic variant was the best tested carrier, but the radio-mistuned artifact remained,
so v4.4 combines that lower-coherence carrier with mel-selected convolutional filter bases
and an explicit aperiodic excitation state.

No persistent v4.4 training is started and no historical checkpoint is loaded or mutated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV44,
    VOCODER_GENERATOR_V4_4_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import LogMelEnvelopeLoss
from lykenox_voice_engine.training.speech_vocoder_harmonic_exposure_loss import (
    VOCODER_HARMONIC_EXPOSURE_VERSION,
    target_relative_harmonic_exposure_loss,
)
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
    target_relative_local_spectral_contrast_loss,
)
from lykenox_voice_engine.training.speech_vocoder_losses import multi_resolution_reconstruction_loss
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


SMOKE_VERSION = "vocoder-v4-4-architecture-smoke-v1"


def _finite_gradients(model: torch.nn.Module) -> bool:
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all()):
            return False
    return found


def _losses(
    model: LykenoxVocoderGeneratorV44,
    envelope_loss: LogMelEnvelopeLoss,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    target: torch.Tensor,
    *,
    envelope_weight: float,
    balance_weight: float,
    contrast_weight: float,
    harmonic_exposure_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = model(mel, f0_hz, voiced)
    reconstruction = multi_resolution_reconstruction_loss(prediction, target).total
    envelope = envelope_loss(prediction, target).total
    balance = target_relative_spectral_balance_loss(
        prediction,
        target,
        sample_rate=model.config.sample_rate,
    ).loss
    contrast = target_relative_local_spectral_contrast_loss(
        prediction,
        target,
        hop_length=model.config.hop_length,
    )
    harmonic = target_relative_harmonic_exposure_loss(
        prediction,
        target,
        f0_hz,
        sample_rate=model.config.sample_rate,
        hop_length=model.config.hop_length,
        harmonics=model.harmonics,
    )
    total = (
        reconstruction
        + envelope_weight * envelope
        + balance_weight * balance
        + contrast_weight * contrast.loss
        + harmonic_exposure_weight * harmonic.loss
    )
    return total, {
        "total": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "log_mel_envelope": float(envelope.detach()),
        "spectral_balance": float(balance.detach()),
        "local_spectral_contrast": float(contrast.loss.detach()),
        "harmonic_exposure": float(harmonic.loss.detach()),
        "prediction_mean_harmonic_exposure": float(harmonic.prediction_mean_exposure.detach()),
        "target_mean_harmonic_exposure": float(harmonic.target_mean_exposure.detach()),
        "harmonic_valid_fraction": float(harmonic.valid_fraction.detach()),
    }


def run_v4_4_architecture_smoke(
    root: Path,
    *,
    steps: int = 10,
    segment_mel_frames: int = 48,
    learning_rate: float = 4e-4,
    envelope_weight: float = 0.50,
    balance_weight: float = 0.25,
    contrast_weight: float = 0.15,
    harmonic_exposure_weight: float = 0.25,
    seed: int = 4444,
) -> dict[str, object]:
    if steps < 4 or steps > 20:
        raise ValueError("steps must be between 4 and 20")
    root = Path(root).resolve()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.manual_seed(seed)

    segments, skipped = collect_vocoder_segments(
        root,
        "train",
        segment_mel_frames=segment_mel_frames,
        max_items=1,
        seed=seed,
    )
    item = segments[0]
    pitch = extract_pitch_frames(item.waveform, frame_count=item.mel_frames)
    mel = item.mel.unsqueeze(0)
    f0_hz = pitch.f0_hz.unsqueeze(0)
    voiced = pitch.voiced.unsqueeze(0)
    target = item.waveform.unsqueeze(0)

    model = LykenoxVocoderGeneratorV44().cpu().train()
    envelope_loss = LogMelEnvelopeLoss().cpu()
    parameters = model.parameter_count()
    receptive_field_samples = model.sample_receptive_field()
    receptive_field_ms = 1000.0 * receptive_field_samples / model.config.sample_rate

    contract_frames = 17
    contract_mel = torch.randn(1, contract_frames, model.config.mel_bins)
    contract_f0 = torch.full((1, contract_frames), 120.0)
    contract_voiced = torch.ones(1, contract_frames)
    with torch.inference_mode():
        contract_wave = model(contract_mel, contract_f0, contract_voiced)
        zero_wave = model.diagnostic_zero_excitation(contract_mel)
    expected_samples = contract_frames * model.config.hop_length
    exact_length_contract = tuple(contract_wave.shape) == (1, expected_samples)
    structural_finite = bool(torch.isfinite(contract_wave).all())
    zero_excitation_max_abs = float(zero_wave.abs().max())
    no_mel_only_waveform = zero_excitation_max_abs == 0.0

    with torch.no_grad():
        _before_total, before = _losses(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            envelope_weight=envelope_weight,
            balance_weight=balance_weight,
            contrast_weight=contrast_weight,
            harmonic_exposure_weight=harmonic_exposure_weight,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    gradients_finite = True
    step_times: list[float] = []
    for _ in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total, _metrics = _losses(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            envelope_weight=envelope_weight,
            balance_weight=balance_weight,
            contrast_weight=contrast_weight,
            harmonic_exposure_weight=harmonic_exposure_weight,
        )
        total.backward()
        gradients_finite = gradients_finite and _finite_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        gradients_finite = gradients_finite and bool(torch.isfinite(norm))
        optimizer.step()
        step_times.append(time.perf_counter() - started)

    with torch.no_grad():
        _after_total, after = _losses(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            envelope_weight=envelope_weight,
            balance_weight=balance_weight,
            contrast_weight=contrast_weight,
            harmonic_exposure_weight=harmonic_exposure_weight,
        )

    benchmark_frames = 96
    benchmark_mel = mel[:, :1].expand(1, benchmark_frames, mel.shape[-1]).contiguous()
    voiced_values = f0_hz[f0_hz > 0.0]
    benchmark_f0_value = float(voiced_values.median()) if voiced_values.numel() else 120.0
    benchmark_f0 = torch.full((1, benchmark_frames), benchmark_f0_value)
    benchmark_voiced = torch.ones(1, benchmark_frames)
    model.eval()
    inference_times: list[float] = []
    with torch.inference_mode():
        _warmup = model(benchmark_mel, benchmark_f0, benchmark_voiced)
        for _ in range(3):
            t0 = time.perf_counter()
            benchmark_wave = model(benchmark_mel, benchmark_f0, benchmark_voiced)
            inference_times.append(time.perf_counter() - t0)
    inference_seconds = sorted(inference_times)[1]
    benchmark_audio_seconds = benchmark_frames * model.config.hop_length / model.config.sample_rate
    benchmark_rtf = inference_seconds / benchmark_audio_seconds

    total_decreased = after["total"] < before["total"]
    envelope_decreased = after["log_mel_envelope"] < before["log_mel_envelope"]
    harmonic_exposure_decreased = after["harmonic_exposure"] < before["harmonic_exposure"]
    parameter_budget_pass = parameters <= 750_000
    receptive_field_pass = receptive_field_ms >= 60.0
    benchmark_finite = bool(torch.isfinite(benchmark_wave).all())
    benchmark_exact_length = tuple(benchmark_wave.shape) == (
        1,
        benchmark_frames * model.config.hop_length,
    )
    status_pass = all(
        (
            exact_length_contract,
            structural_finite,
            no_mel_only_waveform,
            gradients_finite,
            total_decreased,
            envelope_decreased,
            harmonic_exposure_decreased,
            parameter_budget_pass,
            receptive_field_pass,
            benchmark_finite,
            benchmark_exact_length,
        )
    )

    return {
        "status": "pass" if status_pass else "fail",
        "smoke_version": SMOKE_VERSION,
        "device": "cpu",
        "architecture": VOCODER_GENERATOR_V4_4_ARCHITECTURE,
        "harmonics": model.harmonics,
        "filter_bases": model.filter_bases,
        "harmonic_exposure_loss_version": VOCODER_HARMONIC_EXPOSURE_VERSION,
        "persistent_training_started": False,
        "v4_3_checkpoint_mutated": False,
        "parameters": parameters,
        "parameter_budget_pass": parameter_budget_pass,
        "receptive_field_samples": receptive_field_samples,
        "receptive_field_ms": round(receptive_field_ms, 3),
        "receptive_field_pass": receptive_field_pass,
        "exact_length_contract": exact_length_contract,
        "structural_finite": structural_finite,
        "no_mel_only_waveform": no_mel_only_waveform,
        "zero_excitation_max_abs": zero_excitation_max_abs,
        "gradients_finite": gradients_finite,
        "steps": steps,
        "skipped_before_selection": len(skipped),
        "probe_before": {key: round(value, 6) for key, value in before.items()},
        "probe_after": {key: round(value, 6) for key, value in after.items()},
        "total_decreased": total_decreased,
        "envelope_decreased": envelope_decreased,
        "harmonic_exposure_decreased": harmonic_exposure_decreased,
        "mean_seconds_per_step": round(sum(step_times) / len(step_times), 4),
        "max_seconds_per_step": round(max(step_times), 4),
        "benchmark_audio_seconds": round(benchmark_audio_seconds, 4),
        "benchmark_inference_seconds_median": round(inference_seconds, 4),
        "benchmark_rtf": round(benchmark_rtf, 4),
        "benchmark_exact_length": benchmark_exact_length,
        "benchmark_finite": benchmark_finite,
        "reference_audio_required_for_product_inference": False,
        "next_gate": (
            "build_bounded_resumable_v4_4_training_candidate"
            if status_pass
            else "revise_v4_4_architecture_before_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run_v4_4_architecture_smoke(args.root, steps=args.steps), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
