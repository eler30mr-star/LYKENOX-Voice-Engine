"""Bounded real-data architecture smoke for the LYKENOX v4.2 vocoder candidate.

This is deliberately not persistent training.  It answers only the questions required
before spending CPU time on a new checkpoint:

* does the envelope-first v4.2 model preserve the exact waveform-length contract?;
* is the model finite and differentiable on real LYKENOX mel/F0/voicing?;
* can a tiny bounded overfit reduce both the composite reconstruction objective and the
  direct log-mel envelope error?;
* what are the local CPU step time, inference real-time factor and parameter count?;
* does the candidate retain a receptive field large enough to reshape source timbre over
  a meaningful speech window?

No existing v4.1 artifact is loaded or modified and no checkpoint is written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV42,
    VOCODER_GENERATOR_V4_2_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import (
    LogMelEnvelopeLoss,
    VOCODER_ENVELOPE_LOSS_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_losses import (
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


SMOKE_VERSION = "vocoder-v4-2-architecture-smoke-v1"


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
    model: LykenoxVocoderGeneratorV42,
    envelope_loss: LogMelEnvelopeLoss,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    target: torch.Tensor,
    *,
    envelope_weight: float,
    balance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = model(mel, f0_hz, voiced)
    reconstruction = multi_resolution_reconstruction_loss(prediction, target)
    envelope = envelope_loss(prediction, target)
    balance = target_relative_spectral_balance_loss(
        prediction,
        target,
        sample_rate=model.config.sample_rate,
    )
    total = (
        reconstruction.total
        + envelope_weight * envelope.total
        + balance_weight * balance.loss
    )
    metrics = {
        "total": float(total.detach()),
        "reconstruction": float(reconstruction.total.detach()),
        "log_mel_envelope": float(envelope.total.detach()),
        "log_mel_level": float(envelope.log_mel_l1.detach()),
        "spectral_slope": float(envelope.spectral_slope_l1.detach()),
        "temporal_delta": float(envelope.temporal_delta_l1.detach()),
        "spectral_balance": float(balance.loss.detach()),
    }
    return total, metrics


def run_v4_2_architecture_smoke(
    root: Path,
    *,
    steps: int = 8,
    segment_mel_frames: int = 48,
    envelope_weight: float = 0.50,
    balance_weight: float = 0.25,
    learning_rate: float = 4e-4,
    seed: int = 4242,
) -> dict[str, object]:
    if steps < 2 or steps > 32:
        raise ValueError("steps must be between 2 and 32")
    if segment_mel_frames < 32 or segment_mel_frames > 96:
        raise ValueError("segment_mel_frames must be between 32 and 96")

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
    pitch = extract_pitch_frames(
        item.waveform,
        frame_count=item.mel_frames,
    )
    mel = item.mel.unsqueeze(0)
    f0_hz = pitch.f0_hz.unsqueeze(0)
    voiced = pitch.voiced.unsqueeze(0)
    target = item.waveform.unsqueeze(0)

    model = LykenoxVocoderGeneratorV42().cpu().train()
    envelope_loss = LogMelEnvelopeLoss().cpu()
    parameter_count = model.parameter_count()
    receptive_field_samples = model.sample_receptive_field()
    receptive_field_ms = 1000.0 * receptive_field_samples / model.config.sample_rate

    # Structural contract on a shape different from the optimization probe.
    contract_frames = 17
    contract_mel = torch.randn(1, contract_frames, model.config.mel_bins)
    contract_f0 = torch.full((1, contract_frames), 120.0)
    contract_voiced = torch.ones(1, contract_frames)
    with torch.inference_mode():
        contract_wave = model(contract_mel, contract_f0, contract_voiced)
    expected_contract_samples = contract_frames * model.config.hop_length
    exact_length_contract = tuple(contract_wave.shape) == (1, expected_contract_samples)
    structural_finite = bool(torch.isfinite(contract_wave).all())

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
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    step_times: list[float] = []
    gradients_finite = True
    for _step in range(steps):
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
        )
        total.backward()
        gradients_finite = gradients_finite and _finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
        )

    # Short local inference benchmark; report RTF instead of making it a hard gate because
    # CPU scheduling varies between user sessions.
    benchmark_frames = 96
    benchmark_mel = mel[:, :1].expand(1, benchmark_frames, mel.shape[-1]).contiguous()
    voiced_f0_values = f0_hz[f0_hz > 0.0]
    benchmark_f0_value = (
        float(voiced_f0_values.median()) if voiced_f0_values.numel() else 120.0
    )
    benchmark_f0 = torch.full((1, benchmark_frames), benchmark_f0_value)
    benchmark_voiced = torch.ones(1, benchmark_frames)
    model.eval()
    with torch.inference_mode():
        started = time.perf_counter()
        benchmark_wave = model(benchmark_mel, benchmark_f0, benchmark_voiced)
        inference_seconds = time.perf_counter() - started
    benchmark_audio_seconds = (
        benchmark_frames * model.config.hop_length / model.config.sample_rate
    )
    inference_rtf = inference_seconds / benchmark_audio_seconds
    benchmark_exact_length = tuple(benchmark_wave.shape) == (
        1,
        benchmark_frames * model.config.hop_length,
    )
    benchmark_finite = bool(torch.isfinite(benchmark_wave).all())

    total_decreased = after["total"] < before["total"]
    envelope_decreased = after["log_mel_envelope"] < before["log_mel_envelope"]
    parameter_budget_pass = parameter_count <= 600_000
    receptive_field_pass = receptive_field_ms >= 40.0
    status_pass = all(
        (
            exact_length_contract,
            structural_finite,
            gradients_finite,
            benchmark_exact_length,
            benchmark_finite,
            total_decreased,
            envelope_decreased,
            parameter_budget_pass,
            receptive_field_pass,
        )
    )

    report: dict[str, object] = {
        "status": "pass" if status_pass else "fail",
        "smoke_version": SMOKE_VERSION,
        "device": "cpu",
        "architecture": VOCODER_GENERATOR_V4_2_ARCHITECTURE,
        "envelope_loss_version": VOCODER_ENVELOPE_LOSS_VERSION,
        "persistent_training_started": False,
        "v4_1_checkpoint_mutated": False,
        "utterance_id": item.utterance_id,
        "segment_mel_frames": segment_mel_frames,
        "skipped_before_selection": len(skipped),
        "parameters": parameter_count,
        "parameter_budget_pass": parameter_budget_pass,
        "receptive_field_samples": receptive_field_samples,
        "receptive_field_ms": round(receptive_field_ms, 3),
        "receptive_field_pass": receptive_field_pass,
        "exact_length_contract": exact_length_contract,
        "structural_finite": structural_finite,
        "gradients_finite": gradients_finite,
        "steps": steps,
        "envelope_weight": envelope_weight,
        "balance_weight": balance_weight,
        "probe_before": {key: round(value, 6) for key, value in before.items()},
        "probe_after": {key: round(value, 6) for key, value in after.items()},
        "total_decreased": total_decreased,
        "envelope_decreased": envelope_decreased,
        "mean_seconds_per_step": round(sum(step_times) / max(1, len(step_times)), 4),
        "max_seconds_per_step": round(max(step_times), 4),
        "benchmark_frames": benchmark_frames,
        "benchmark_audio_seconds": round(benchmark_audio_seconds, 4),
        "benchmark_inference_seconds": round(inference_seconds, 4),
        "benchmark_rtf": round(inference_rtf, 4),
        "benchmark_exact_length": benchmark_exact_length,
        "benchmark_finite": benchmark_finite,
        "reference_audio_required_for_product_inference": False,
        "source_speaker_required": False,
        "next_gate": (
            "build_bounded_resumable_v4_2_training_candidate"
            if status_pass
            else "revise_v4_2_architecture_before_training"
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            run_v4_2_architecture_smoke(args.root, steps=args.steps),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
