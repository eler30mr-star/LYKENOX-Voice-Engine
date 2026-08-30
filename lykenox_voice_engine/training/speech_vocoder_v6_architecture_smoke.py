"""Bounded architecture smoke for LYKENOX vocoder v6.

This gate is intentionally aimed at the two perceptual failures that blocked v5:

- source/noise coloration: v6 must have no explicit voiced source, harmonic carrier, pulse
  train or raw source bypass;
- weak/nasal perceived voice: the short real-data probe must demonstrate trainable target-
  relative level and spectral-presence control in addition to ordinary reconstruction.

Revision 2 also requires structural separation between waveform shape and output level, plus
a direct decrease in absolute RMS error. No persistent training is started and no historical
checkpoint is loaded or mutated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV6,
    VOCODER_GENERATOR_V6_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import LogMelEnvelopeLoss
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
    target_relative_level_loss,
    target_relative_presence_loss,
)
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
    target_relative_local_spectral_contrast_loss,
)
from lykenox_voice_engine.training.speech_vocoder_losses import multi_resolution_reconstruction_loss
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


SMOKE_VERSION = "vocoder-v6-direct-waveform-clarity-level-smoke-v2"


def _finite_gradients(model: torch.nn.Module) -> bool:
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all()):
            return False
    return found


def _unit_rms(waveform: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(waveform.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
    return waveform / rms


def _losses(
    model: LykenoxVocoderGeneratorV6,
    envelope_loss: LogMelEnvelopeLoss,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    target: torch.Tensor,
    *,
    envelope_weight: float,
    balance_weight: float,
    contrast_weight: float,
    level_weight: float,
    presence_weight: float,
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
    ).loss
    level = target_relative_level_loss(prediction, target)
    presence = target_relative_presence_loss(
        prediction,
        target,
        sample_rate=model.config.sample_rate,
        hop_length=model.config.hop_length,
    )
    total = (
        reconstruction
        + envelope_weight * envelope
        + balance_weight * balance
        + contrast_weight * contrast
        + level_weight * level.loss
        + presence_weight * presence.loss
    )
    fractions = presence.prediction_band_fractions.detach().cpu().tolist()
    target_fractions = presence.target_band_fractions.detach().cpu().tolist()
    return total, {
        "total": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "log_mel_envelope": float(envelope.detach()),
        "spectral_balance": float(balance.detach()),
        "local_spectral_contrast": float(contrast.detach()),
        "level": float(level.loss.detach()),
        "global_log_rms_loss": float(level.global_log_rms_loss.detach()),
        "frame_log_rms_loss": float(level.frame_log_rms_loss.detach()),
        "active_frame_fraction": float(level.active_frame_fraction.detach()),
        "presence": float(presence.loss.detach()),
        "prediction_rms_db": float(level.prediction_rms_db.detach()),
        "target_rms_db": float(level.target_rms_db.detach()),
        "rms_error_db": float(level.rms_error_db.detach()),
        "presence_1k_8k_error_db": float(presence.presence_1k_8k_error_db.detach()),
        "prediction_band_80_300": float(fractions[0]),
        "prediction_band_300_1000": float(fractions[1]),
        "prediction_band_1k_3k": float(fractions[2]),
        "prediction_band_3k_8k": float(fractions[3]),
        "target_band_80_300": float(target_fractions[0]),
        "target_band_300_1000": float(target_fractions[1]),
        "target_band_1k_3k": float(target_fractions[2]),
        "target_band_3k_8k": float(target_fractions[3]),
    }


def run_v6_architecture_smoke(
    root: Path,
    *,
    steps: int = 10,
    segment_mel_frames: int = 40,
    learning_rate: float = 4e-4,
    level_learning_rate_multiplier: float = 20.0,
    envelope_weight: float = 0.50,
    balance_weight: float = 0.20,
    contrast_weight: float = 0.15,
    level_weight: float = 0.75,
    presence_weight: float = 0.35,
    seed: int = 6600,
) -> dict[str, object]:
    if steps < 6 or steps > 16:
        raise ValueError("steps must be between 6 and 16")
    if not (1.0 <= level_learning_rate_multiplier <= 32.0):
        raise ValueError("level_learning_rate_multiplier must be between 1 and 32")
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

    model = LykenoxVocoderGeneratorV6().cpu().train()
    envelope_loss = LogMelEnvelopeLoss().cpu()
    parameters = model.parameter_count()
    receptive_field_samples = model.sample_decoder_receptive_field()
    receptive_field_ms = 1000.0 * receptive_field_samples / model.config.sample_rate

    contract_frames = 17
    contract_mel = torch.randn(1, contract_frames, model.config.mel_bins)
    contract_f0 = torch.full((1, contract_frames), 120.0)
    contract_voiced = torch.ones(1, contract_frames)
    with torch.inference_mode():
        contract_wave = model(contract_mel, contract_f0, contract_voiced)
        changed_mel_wave = model(contract_mel * 0.5, contract_f0, contract_voiced)
        low_f0_wave = model(
            contract_mel,
            torch.full_like(contract_f0, 90.0),
            contract_voiced,
        )
        high_f0_wave = model(
            contract_mel,
            torch.full_like(contract_f0, 180.0),
            contract_voiced,
        )
        unvoiced_wave = model(
            contract_mel,
            torch.zeros_like(contract_f0),
            torch.zeros_like(contract_voiced),
        )

        level_bias_before = model.level_logit_bias_parameter.detach().clone()
        model.level_logit_bias_parameter.add_(0.50)
        raised_level_wave = model(contract_mel, contract_f0, contract_voiced)
        model.level_logit_bias_parameter.copy_(level_bias_before)

    expected_samples = contract_frames * model.config.hop_length
    exact_length_contract = tuple(contract_wave.shape) == (1, expected_samples)
    structural_finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            contract_wave,
            changed_mel_wave,
            low_f0_wave,
            high_f0_wave,
            unvoiced_wave,
            raised_level_wave,
        )
    )
    mel_changes_waveform = float((contract_wave - changed_mel_wave).abs().max()) > 1e-8
    f0_changes_waveform = float((low_f0_wave - high_f0_wave).abs().max()) > 1e-8
    voicing_changes_waveform = float((contract_wave - unvoiced_wave).abs().max()) > 1e-8
    direct_waveform_contract = all(
        (
            model.source_family == "direct_conditional_waveform_decoder",
            model.explicit_source is False,
            model.explicit_sinusoidal_carrier is False,
            model.deterministic_harmonics == 0,
            model.voiced_noise_source is False,
            model.raw_source_bypass is False,
            model.conditioning_only_waveform is True,
            model.waveform_shape_level_decoupled is True,
            model.level_control_family == "mel_conditioned_frame_rms_envelope",
        )
    )

    contract_rms = torch.sqrt(contract_wave.square().mean().clamp_min(1e-8))
    raised_level_rms = torch.sqrt(raised_level_wave.square().mean().clamp_min(1e-8))
    level_control_raises_rms = float(raised_level_rms) > float(contract_rms) * 1.15
    normalized_shape_delta = float(
        (_unit_rms(contract_wave) - _unit_rms(raised_level_wave)).abs().mean()
    )
    level_control_shape_stable = normalized_shape_delta < 0.05

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
            level_weight=level_weight,
            presence_weight=presence_weight,
        )

    level_parameters = list(model.level_parameters())
    level_parameter_ids = {id(parameter) for parameter in level_parameters}
    shape_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in level_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": shape_parameters,
                "lr": learning_rate,
                "weight_decay": 1e-5,
            },
            {
                "params": level_parameters,
                "lr": learning_rate * level_learning_rate_multiplier,
                "weight_decay": 0.0,
            },
        ]
    )
    gradients_finite = True
    step_times: list[float] = []
    best_seen = dict(before)
    tracked_keys = (
        "total",
        "log_mel_envelope",
        "level",
        "rms_error_db",
        "presence",
        "presence_1k_8k_error_db",
    )
    for _ in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total, metrics = _losses(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            envelope_weight=envelope_weight,
            balance_weight=balance_weight,
            contrast_weight=contrast_weight,
            level_weight=level_weight,
            presence_weight=presence_weight,
        )
        total.backward()
        gradients_finite = gradients_finite and _finite_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        gradients_finite = gradients_finite and bool(torch.isfinite(norm))
        optimizer.step()
        step_times.append(time.perf_counter() - started)
        for key in tracked_keys:
            best_seen[key] = min(best_seen[key], metrics[key])

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
            level_weight=level_weight,
            presence_weight=presence_weight,
        )
    for key in tracked_keys:
        best_seen[key] = min(best_seen[key], after[key])

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
    benchmark_audio_seconds = (
        benchmark_frames * model.config.hop_length / model.config.sample_rate
    )
    benchmark_rtf = inference_seconds / benchmark_audio_seconds

    total_decreased = best_seen["total"] < before["total"]
    envelope_decreased = best_seen["log_mel_envelope"] < before["log_mel_envelope"]
    level_decreased = best_seen["level"] < before["level"]
    level_final_decreased = after["level"] < before["level"]
    rms_error_decreased = best_seen["rms_error_db"] < before["rms_error_db"]
    rms_error_final_decreased = after["rms_error_db"] < before["rms_error_db"]
    presence_decreased = best_seen["presence"] < before["presence"]
    presence_error_decreased = (
        best_seen["presence_1k_8k_error_db"] < before["presence_1k_8k_error_db"]
    )
    parameter_budget_pass = parameters <= 500_000
    receptive_field_pass = receptive_field_ms >= 60.0
    cpu_candidate_pass = benchmark_rtf <= 1.5
    benchmark_finite = bool(torch.isfinite(benchmark_wave).all())
    benchmark_exact_length = tuple(benchmark_wave.shape) == (
        1,
        benchmark_frames * model.config.hop_length,
    )
    status_pass = all(
        (
            exact_length_contract,
            structural_finite,
            direct_waveform_contract,
            mel_changes_waveform,
            f0_changes_waveform,
            voicing_changes_waveform,
            level_control_raises_rms,
            level_control_shape_stable,
            gradients_finite,
            total_decreased,
            envelope_decreased,
            level_decreased,
            level_final_decreased,
            rms_error_decreased,
            rms_error_final_decreased,
            presence_decreased,
            presence_error_decreased,
            parameter_budget_pass,
            receptive_field_pass,
            cpu_candidate_pass,
            benchmark_finite,
            benchmark_exact_length,
        )
    )

    return {
        "status": "pass" if status_pass else "fail",
        "smoke_version": SMOKE_VERSION,
        "level_presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
        "device": "cpu",
        "architecture": VOCODER_GENERATOR_V6_ARCHITECTURE,
        "source_family": model.source_family,
        "explicit_source": model.explicit_source,
        "explicit_sinusoidal_carrier": model.explicit_sinusoidal_carrier,
        "deterministic_harmonics": model.deterministic_harmonics,
        "voiced_noise_source": model.voiced_noise_source,
        "raw_source_bypass": model.raw_source_bypass,
        "conditioning_only_waveform": model.conditioning_only_waveform,
        "waveform_shape_level_decoupled": model.waveform_shape_level_decoupled,
        "level_control_family": model.level_control_family,
        "persistent_training_started": False,
        "historical_checkpoints_mutated": False,
        "parameters": parameters,
        "level_parameters": sum(parameter.numel() for parameter in level_parameters),
        "parameter_budget_pass": parameter_budget_pass,
        "sample_decoder_receptive_field_samples": receptive_field_samples,
        "sample_decoder_receptive_field_ms": round(receptive_field_ms, 3),
        "receptive_field_pass": receptive_field_pass,
        "exact_length_contract": exact_length_contract,
        "structural_finite": structural_finite,
        "direct_waveform_contract": direct_waveform_contract,
        "mel_changes_waveform": mel_changes_waveform,
        "f0_changes_waveform": f0_changes_waveform,
        "voicing_changes_waveform": voicing_changes_waveform,
        "level_control_raises_rms": level_control_raises_rms,
        "level_control_shape_stable": level_control_shape_stable,
        "level_control_normalized_shape_delta": round(normalized_shape_delta, 6),
        "gradients_finite": gradients_finite,
        "steps": steps,
        "learning_rate": learning_rate,
        "level_learning_rate": learning_rate * level_learning_rate_multiplier,
        "level_learning_rate_multiplier": level_learning_rate_multiplier,
        "skipped_before_selection": len(skipped),
        "probe_before": {key: round(value, 6) for key, value in before.items()},
        "probe_after": {key: round(value, 6) for key, value in after.items()},
        "probe_best_seen": {
            key: round(best_seen[key], 6)
            for key in tracked_keys
        },
        "total_decreased": total_decreased,
        "envelope_decreased": envelope_decreased,
        "level_decreased": level_decreased,
        "level_final_decreased": level_final_decreased,
        "rms_error_decreased": rms_error_decreased,
        "rms_error_final_decreased": rms_error_final_decreased,
        "presence_decreased": presence_decreased,
        "presence_error_decreased": presence_error_decreased,
        "nominal_output_rms": round(float(model.nominal_output_rms().detach()), 6),
        "mean_seconds_per_step": round(sum(step_times) / len(step_times), 4),
        "max_seconds_per_step": round(max(step_times), 4),
        "benchmark_audio_seconds": round(benchmark_audio_seconds, 4),
        "benchmark_inference_seconds_median": round(inference_seconds, 4),
        "benchmark_rtf": round(benchmark_rtf, 4),
        "cpu_candidate_pass": cpu_candidate_pass,
        "benchmark_exact_length": benchmark_exact_length,
        "benchmark_finite": benchmark_finite,
        "reference_audio_required_for_product_inference": False,
        "next_gate": (
            "build_bounded_resumable_v6_training_candidate"
            if status_pass
            else "revise_v6_direct_waveform_architecture_before_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            run_v6_architecture_smoke(args.root, steps=args.steps),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
