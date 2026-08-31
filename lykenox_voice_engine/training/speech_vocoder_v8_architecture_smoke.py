"""Bounded non-persistent architecture smoke for the V8 complex-spectral OLA vocoder.

This is a tiny in-memory overfit on one real aligned mel/waveform crop. It is not
persistent training and writes no checkpoint. The v1 smoke is intentionally superseded:
its absolute frame-grid flag could reject a numerically exact STFT/iSTFT round-trip of a
naturally periodic voiced crop, and its composite optimization could improve envelope
terms while the architecture's primary complex-spectrum objective regressed.

The v2 gate fixes both problems: frame-grid rejection is reference-relative, and the
optimization objective is the direct complex-spectral loss. Envelope and broad-band
balance remain diagnostics/rejection metrics; they do not override the representation
that V8 is explicitly designed to learn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV8,
    VOCODER_GENERATOR_V8_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import LogMelEnvelopeLoss
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    VOCODER_GRID_ARTIFACT_EXCESS_VERSION,
    frame_grid_artifact_excess_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)
from lykenox_voice_engine.training.speech_vocoder_v8_complex_spectral_loss import (
    V8_COMPLEX_SPECTRAL_LOSS_VERSION,
    v8_complex_spectral_loss,
)


PRIOR_SMOKE_VERSION = "vocoder-v8-complex-spectral-ola-smoke-v1"
SMOKE_VERSION = "vocoder-v8-complex-spectral-ola-smoke-v2"
PRIOR_SMOKE_INVALIDATED = True
DEFAULT_STEPS = 12
DEFAULT_SEGMENT_MEL_FRAMES = 48
DEFAULT_LEARNING_RATE = 3e-4


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v6_last": training / "vocoder_direct_waveform_v6" / "last.pt",
        "v6_clarity_best": training / "vocoder_direct_waveform_v6_clarity_guard_v1" / "best.pt",
        "v6_clarity_last": training / "vocoder_direct_waveform_v6_clarity_guard_v1" / "last.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
    }


def _forward_loss(
    model: LykenoxVocoderGeneratorV8,
    envelope_loss: LogMelEnvelopeLoss,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    target: torch.Tensor,
    target_spectrum: torch.Tensor,
):
    predicted_spectrum = model.predict_complex_spectrum(mel, f0_hz, voiced)
    waveform = model.synthesize_complex_spectrum(
        predicted_spectrum,
        samples=int(target.shape[-1]),
    )
    complex_loss = v8_complex_spectral_loss(
        predicted_spectrum,
        target_spectrum,
        waveform,
        target,
    )
    envelope = envelope_loss(waveform, target)
    balance = target_relative_spectral_balance_loss(
        waveform,
        target,
        sample_rate=model.config.sample_rate,
    )

    # Architecture smoke optimization is representation-first.  The prior v1 composite
    # objective could reduce envelope/balance enough to hide a regression in the direct
    # complex coefficients, which is precisely V8's core representation contract.
    optimization_total = complex_loss.total
    diagnostic_composite_total = (
        complex_loss.total + 0.50 * envelope.total + 0.25 * balance.loss
    )
    metrics = {
        "total": float(optimization_total.detach()),
        "diagnostic_composite_total": float(diagnostic_composite_total.detach()),
        "complex_relative_l1": float(complex_loss.complex_relative_l1.detach()),
        "log_magnitude_l1": float(complex_loss.log_magnitude_l1.detach()),
        "waveform_l1": float(complex_loss.waveform_l1.detach()),
        "envelope": float(envelope.total.detach()),
        "spectral_balance": float(balance.loss.detach()),
    }
    return optimization_total, waveform, metrics


def _finite_gradients(model: torch.nn.Module) -> bool:
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all()):
            return False
    return found


def run_v8_architecture_smoke(
    root: Path,
    *,
    steps: int = DEFAULT_STEPS,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 8080,
) -> dict[str, object]:
    if steps < 4 or steps > 24:
        raise ValueError("steps must be between 4 and 24")
    if segment_mel_frames < 32 or segment_mel_frames > 96:
        raise ValueError("segment_mel_frames must be between 32 and 96")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    root = Path(root).resolve()
    protected = _protected(root)
    before_hashes = {name: _sha256(path) for name, path in protected.items()}
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

    model = LykenoxVocoderGeneratorV8().cpu().train()
    envelope_loss = LogMelEnvelopeLoss().cpu()
    parameter_count = model.parameter_count()
    frame_receptive_field = model.frame_receptive_field()

    # Static shape/finite contract on a different frame count.
    contract_frames = 17
    contract_mel = torch.randn(1, contract_frames, model.config.mel_bins)
    contract_f0 = torch.full((1, contract_frames), 120.0)
    contract_voiced = torch.ones(1, contract_frames)
    with torch.inference_mode():
        contract_spectrum = model.predict_complex_spectrum(
            contract_mel, contract_f0, contract_voiced
        )
        contract_wave = model(contract_mel, contract_f0, contract_voiced)
    exact_spectrum_contract = tuple(contract_spectrum.shape) == (
        1,
        model.frequency_bins,
        contract_frames + 1,
    )
    exact_length_contract = tuple(contract_wave.shape) == (
        1,
        contract_frames * model.config.hop_length,
    )
    structural_finite = bool(torch.isfinite(contract_wave).all()) and bool(
        torch.isfinite(torch.view_as_real(contract_spectrum)).all()
    )

    # Prove the fixed analysis/synthesis geometry itself can reconstruct a real waveform.
    # no_grad is intentional rather than inference_mode because target_spectrum is reused
    # as a constant target inside subsequent differentiable loss computations.
    with torch.no_grad():
        target_spectrum = model.target_complex_spectrum(target)
        roundtrip = model.synthesize_complex_spectrum(
            target_spectrum,
            samples=int(target.shape[-1]),
        )
    target_spectrum_shape_exact = tuple(target_spectrum.shape) == (
        1,
        model.frequency_bins,
        segment_mel_frames + 1,
    )
    roundtrip_mae = float((roundtrip - target).abs().mean())
    roundtrip_max_abs = float((roundtrip - target).abs().max())
    roundtrip_exact_enough = roundtrip_mae < 1e-5
    roundtrip_grid = frame_grid_artifact_excess_metrics(
        roundtrip,
        target,
        sample_rate=model.config.sample_rate,
        hop_length=model.config.hop_length,
    )
    roundtrip_grid_failure = bool(roundtrip_grid.severe_grid_excess[0])

    with torch.no_grad():
        _initial_total_tensor, _initial_wave, initial = _forward_loss(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            target_spectrum,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    gradients_finite = True
    best_total = initial["total"]
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total, _waveform, _metrics = _forward_loss(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            target_spectrum,
        )
        if not bool(torch.isfinite(total)):
            raise RuntimeError("V8 smoke produced non-finite loss")
        total.backward()
        gradients_finite = gradients_finite and _finite_gradients(model)
        if not gradients_finite:
            raise RuntimeError("V8 smoke produced non-finite gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            _candidate_total, _candidate_wave, candidate_metrics = _forward_loss(
                model,
                envelope_loss,
                mel,
                f0_hz,
                voiced,
                target,
                target_spectrum,
            )
        if candidate_metrics["total"] < best_total:
            best_total = candidate_metrics["total"]
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _final_total_tensor, final_wave, final = _forward_loss(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            target_spectrum,
        )

    final_grid = frame_grid_artifact_excess_metrics(
        final_wave,
        target,
        sample_rate=model.config.sample_rate,
        hop_length=model.config.hop_length,
    )
    final_grid_failure = bool(final_grid.severe_grid_excess[0])
    total_decreased = final["total"] < initial["total"]
    complex_decreased = final["complex_relative_l1"] < initial["complex_relative_l1"]
    envelope_decreased = final["envelope"] < initial["envelope"]
    log_magnitude_decreased = final["log_magnitude_l1"] < initial["log_magnitude_l1"]
    waveform_decreased = final["waveform_l1"] < initial["waveform_l1"]
    parameter_budget_pass = parameter_count <= 650_000

    after_hashes = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before_hashes == after_hashes
    status_pass = all(
        (
            exact_spectrum_contract,
            exact_length_contract,
            structural_finite,
            target_spectrum_shape_exact,
            roundtrip_exact_enough,
            not roundtrip_grid_failure,
            gradients_finite,
            total_decreased,
            complex_decreased,
            envelope_decreased,
            log_magnitude_decreased,
            waveform_decreased,
            not final_grid_failure,
            parameter_budget_pass,
            checkpoints_unchanged,
        )
    )

    return {
        "status": "pass" if status_pass else "fail",
        "smoke_version": SMOKE_VERSION,
        "prior_smoke_version": PRIOR_SMOKE_VERSION,
        "prior_smoke_invalidated": PRIOR_SMOKE_INVALIDATED,
        "prior_smoke_invalidated_reason": (
            "absolute_grid_flag_false_positive_on_near_exact_natural_roundtrip_and_"
            "composite_objective_allowed_complex_regression"
        ),
        "architecture": VOCODER_GENERATOR_V8_ARCHITECTURE,
        "complex_loss_version": V8_COMPLEX_SPECTRAL_LOSS_VERSION,
        "grid_gate_version": VOCODER_GRID_ARTIFACT_EXCESS_VERSION,
        "grid_gate_mode": "paired_reference_relative_excess",
        "optimization_objective": "direct_complex_spectral_loss_only",
        "device": "cpu",
        "utterance_id": item.utterance_id,
        "steps": steps,
        "segment_mel_frames": segment_mel_frames,
        "skipped_before_selection": len(skipped),
        "parameters": parameter_count,
        "parameter_budget_pass": parameter_budget_pass,
        "frame_receptive_field": frame_receptive_field,
        "exact_spectrum_contract": exact_spectrum_contract,
        "exact_length_contract": exact_length_contract,
        "structural_finite": structural_finite,
        "target_spectrum_shape_exact": target_spectrum_shape_exact,
        "fixed_stft_istft_roundtrip_mae": round(roundtrip_mae, 9),
        "fixed_stft_istft_roundtrip_max_abs": round(roundtrip_max_abs, 9),
        "reference_grid_hop_autocorrelation": round(
            float(roundtrip_grid.reference.hop_autocorrelation[0]), 6
        ),
        "reference_grid_double_hop_autocorrelation": round(
            float(roundtrip_grid.reference.double_hop_autocorrelation[0]), 6
        ),
        "reference_grid_harmonic_power_fraction": round(
            float(roundtrip_grid.reference.grid_harmonic_power_fraction[0]), 6
        ),
        "fixed_roundtrip_grid_hop_excess": round(
            float(roundtrip_grid.hop_autocorrelation_excess[0]), 6
        ),
        "fixed_roundtrip_grid_double_hop_excess": round(
            float(roundtrip_grid.double_hop_autocorrelation_excess[0]), 6
        ),
        "fixed_roundtrip_grid_harmonic_power_excess": round(
            float(roundtrip_grid.grid_harmonic_power_fraction_excess[0]), 6
        ),
        "fixed_roundtrip_grid_failure": roundtrip_grid_failure,
        "gradients_finite": gradients_finite,
        "initial": {key: round(value, 6) for key, value in initial.items()},
        "final": {key: round(value, 6) for key, value in final.items()},
        "total_decreased": total_decreased,
        "complex_decreased": complex_decreased,
        "log_magnitude_decreased": log_magnitude_decreased,
        "waveform_decreased": waveform_decreased,
        "envelope_decreased": envelope_decreased,
        "final_grid_hop_autocorrelation": round(
            float(final_grid.candidate.hop_autocorrelation[0]), 6
        ),
        "final_grid_double_hop_autocorrelation": round(
            float(final_grid.candidate.double_hop_autocorrelation[0]), 6
        ),
        "final_grid_harmonic_power_fraction": round(
            float(final_grid.candidate.grid_harmonic_power_fraction[0]), 6
        ),
        "final_grid_hop_excess": round(
            float(final_grid.hop_autocorrelation_excess[0]), 6
        ),
        "final_grid_double_hop_excess": round(
            float(final_grid.double_hop_autocorrelation_excess[0]), 6
        ),
        "final_grid_harmonic_power_excess": round(
            float(final_grid.grid_harmonic_power_fraction_excess[0]), 6
        ),
        "final_grid_failure": final_grid_failure,
        "source_free": model.source_free,
        "explicit_sample_rate_source": model.explicit_sample_rate_source,
        "learned_sample_rate_upsampling": model.learned_sample_rate_upsampling,
        "synthesis": model.synthesis,
        "protected_checkpoints_unchanged": checkpoints_unchanged,
        "persistent_training_started": False,
        "persistent_training_authorized": False,
        "metrics_can_accept_voice_quality": False,
        "audible_full_utterance_acceptance_required": True,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "next_gate": (
            "build_exact_resume_v8_first_epoch_candidate"
            if status_pass
            else "reject_or_revise_v8_before_persistent_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_v8_architecture_smoke(args.root, steps=args.steps),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
