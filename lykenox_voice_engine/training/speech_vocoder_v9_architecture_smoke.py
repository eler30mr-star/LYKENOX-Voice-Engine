"""Bounded non-persistent architecture smoke for the V9 phase-increment vocoder.

V8 learned hop-locked repetition while predicting absolute complex STFT frames. V9 instead
supervises magnitude and inter-frame phase advance. This smoke first proves that the V9
factorization itself reconstructs a real waveform, then performs a tiny in-memory overfit
and rejects any learned frame-grid excess relative to the paired reference.

The historical v1 gate is superseded. It required raw sample-domain waveform L1 to decrease
while assigning that term only 0.05 weight inside the optimization objective. That was an
internally inconsistent gate: the optimizer could rationally improve the spectral
representation and envelope while slightly worsening a phase-sensitive sample L1, then be
rejected for the term it was barely asked to optimize. V2 uses representation +
multi-resolution spectral reconstruction + envelope + target-relative presence as the
optimization/rejection surface. Raw waveform L1 remains diagnostic, never an acceptance
metric. If the structural smoke passes, it writes reference/final WAVs for an auditory gate
before any persistent checkpoint may be built.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV9,
    VOCODER_GENERATOR_V9_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_envelope_loss import LogMelEnvelopeLoss
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    VOCODER_GRID_ARTIFACT_EXCESS_VERSION,
    frame_grid_artifact_excess_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
    target_relative_presence_loss,
)
from lykenox_voice_engine.training.speech_vocoder_losses import (
    VOCODER_LOSS_RECIPE_VERSION,
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_v9_phase_increment_loss import (
    PRIOR_V9_PHASE_INCREMENT_LOSS_INVALIDATED,
    PRIOR_V9_PHASE_INCREMENT_LOSS_VERSION,
    V9_PHASE_INCREMENT_LOSS_VERSION,
    v9_phase_increment_loss,
)


PRIOR_SMOKE_VERSION = "vocoder-v9-phase-increment-ola-smoke-v1"
SMOKE_VERSION = "vocoder-v9-phase-increment-ola-smoke-v2"
PRIOR_SMOKE_INVALIDATED = True
DEFAULT_STEPS = 16
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


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _protected(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v6_clarity_best": training / "vocoder_direct_waveform_v6_clarity_guard_v1" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
    }


def _finite_gradients(model: torch.nn.Module) -> bool:
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all()):
            return False
    return found


def _forward_loss(
    model: LykenoxVocoderGeneratorV9,
    envelope_loss: LogMelEnvelopeLoss,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    target: torch.Tensor,
    target_magnitude: torch.Tensor,
    target_residual_phase: torch.Tensor,
):
    magnitude, residual_phase = model.predict_spectral_factors(mel, f0_hz, voiced)
    spectrum = model.spectrum_from_factors(magnitude, residual_phase)
    waveform = model.synthesize_complex_spectrum(spectrum, samples=int(target.shape[-1]))
    representation = v9_phase_increment_loss(
        magnitude,
        target_magnitude,
        residual_phase,
        target_residual_phase,
        waveform,
        target,
    )
    reconstruction = multi_resolution_reconstruction_loss(waveform, target)
    envelope = envelope_loss(waveform, target)
    presence = target_relative_presence_loss(
        waveform,
        target,
        sample_rate=model.config.sample_rate,
        hop_length=model.config.hop_length,
    )
    optimization_total = (
        representation.total
        + 0.50 * reconstruction.total
        + 0.25 * envelope.total
        + 0.25 * presence.loss
    )
    metrics = {
        "total": float(optimization_total.detach()),
        "representation_total": float(representation.total.detach()),
        "log_magnitude_l1": float(representation.log_magnitude_l1.detach()),
        "phase_increment_circular": float(representation.phase_increment_circular.detach()),
        "waveform_l1": float(representation.waveform_l1.detach()),
        "reconstruction": float(reconstruction.total.detach()),
        "reconstruction_spectral_convergence": float(reconstruction.spectral_convergence.detach()),
        "reconstruction_log_magnitude": float(reconstruction.log_magnitude.detach()),
        "envelope": float(envelope.total.detach()),
        "presence": float(presence.loss.detach()),
        "presence_1k_8k_error_db": float(presence.presence_1k_8k_error_db.detach()),
    }
    return optimization_total, waveform, metrics


def run_v9_architecture_smoke(
    root: Path,
    *,
    steps: int = DEFAULT_STEPS,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 9090,
) -> dict[str, object]:
    if steps < 8 or steps > 32:
        raise ValueError("steps must be between 8 and 32")
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

    model = LykenoxVocoderGeneratorV9().cpu().train()
    envelope_loss = LogMelEnvelopeLoss().cpu()
    parameter_count = model.parameter_count()

    contract_frames = 17
    contract_mel = torch.randn(1, contract_frames, model.config.mel_bins)
    contract_f0 = torch.full((1, contract_frames), 120.0)
    contract_voiced = torch.ones(1, contract_frames)
    with torch.inference_mode():
        contract_magnitude, contract_residual = model.predict_spectral_factors(
            contract_mel, contract_f0, contract_voiced
        )
        contract_spectrum = model.spectrum_from_factors(contract_magnitude, contract_residual)
        contract_wave = model(contract_mel, contract_f0, contract_voiced)
    expected_spectrum_shape = (1, model.frequency_bins, contract_frames + 1)
    exact_factor_contract = (
        tuple(contract_magnitude.shape) == expected_spectrum_shape
        and tuple(contract_residual.shape) == expected_spectrum_shape
        and tuple(contract_spectrum.shape) == expected_spectrum_shape
    )
    exact_length_contract = tuple(contract_wave.shape) == (
        1,
        contract_frames * model.config.hop_length,
    )
    structural_finite = bool(torch.isfinite(contract_wave).all()) and bool(
        torch.isfinite(contract_magnitude).all()
    ) and bool(torch.isfinite(torch.view_as_real(contract_residual)).all())

    with torch.no_grad():
        target_spectrum = model.target_complex_spectrum(target)
        target_magnitude, target_residual_phase = model.factorize_target_spectrum(target_spectrum)
        refactorized_spectrum = model.spectrum_from_factors(
            target_magnitude,
            target_residual_phase,
        )
        factorized_roundtrip = model.synthesize_complex_spectrum(
            refactorized_spectrum,
            samples=int(target.shape[-1]),
        )
    factorized_spectrum_mae = float((refactorized_spectrum - target_spectrum).abs().mean())
    factorized_waveform_mae = float((factorized_roundtrip - target).abs().mean())
    factorization_exact_enough = (
        factorized_spectrum_mae < 1e-4 and factorized_waveform_mae < 1e-5
    )
    factorization_grid = frame_grid_artifact_excess_metrics(
        factorized_roundtrip,
        target,
        sample_rate=model.config.sample_rate,
        hop_length=model.config.hop_length,
    )
    factorization_grid_failure = bool(factorization_grid.severe_grid_excess[0])

    with torch.no_grad():
        _initial_total, initial_wave, initial = _forward_loss(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            target_magnitude,
            target_residual_phase,
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
            target_magnitude,
            target_residual_phase,
        )
        if not bool(torch.isfinite(total)):
            raise RuntimeError("V9 smoke produced non-finite loss")
        total.backward()
        gradients_finite = gradients_finite and _finite_gradients(model)
        if not gradients_finite:
            raise RuntimeError("V9 smoke produced non-finite gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            _candidate_total, _candidate_wave, candidate = _forward_loss(
                model,
                envelope_loss,
                mel,
                f0_hz,
                voiced,
                target,
                target_magnitude,
                target_residual_phase,
            )
        if candidate["total"] < best_total:
            best_total = candidate["total"]
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _final_total, final_wave, final = _forward_loss(
            model,
            envelope_loss,
            mel,
            f0_hz,
            voiced,
            target,
            target_magnitude,
            target_residual_phase,
        )

    final_grid = frame_grid_artifact_excess_metrics(
        final_wave,
        target,
        sample_rate=model.config.sample_rate,
        hop_length=model.config.hop_length,
    )
    final_grid_failure = bool(final_grid.severe_grid_excess[0])
    total_decreased = final["total"] < initial["total"]
    representation_decreased = final["representation_total"] < initial["representation_total"]
    magnitude_decreased = final["log_magnitude_l1"] < initial["log_magnitude_l1"]
    phase_increment_decreased = (
        final["phase_increment_circular"] < initial["phase_increment_circular"]
    )
    reconstruction_decreased = final["reconstruction"] < initial["reconstruction"]
    envelope_decreased = final["envelope"] < initial["envelope"]
    presence_decreased = final["presence"] < initial["presence"]
    presence_error_decreased = (
        final["presence_1k_8k_error_db"] < initial["presence_1k_8k_error_db"]
    )
    parameter_budget_pass = parameter_count <= 650_000

    after_hashes = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before_hashes == after_hashes
    status_pass = all(
        (
            exact_factor_contract,
            exact_length_contract,
            structural_finite,
            factorization_exact_enough,
            not factorization_grid_failure,
            gradients_finite,
            total_decreased,
            representation_decreased,
            magnitude_decreased,
            phase_increment_decreased,
            reconstruction_decreased,
            envelope_decreased,
            presence_decreased,
            presence_error_decreased,
            not final_grid_failure,
            parameter_budget_pass,
            checkpoints_unchanged,
        )
    )

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_v9_bounded_oracle_smoke_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "reference.wav"
    initial_path = output_dir / "initial_v9.wav"
    final_path = output_dir / "final_v9.wav"
    sf.write(reference_path, target[0].detach().cpu().numpy(), model.config.sample_rate, subtype="FLOAT")
    sf.write(initial_path, initial_wave[0].detach().cpu().numpy(), model.config.sample_rate, subtype="FLOAT")
    sf.write(final_path, final_wave[0].detach().cpu().numpy(), model.config.sample_rate, subtype="FLOAT")

    report: dict[str, object] = {
        "status": "needs_listening" if status_pass else "fail",
        "structural_smoke_pass": status_pass,
        "smoke_version": SMOKE_VERSION,
        "prior_smoke_version": PRIOR_SMOKE_VERSION,
        "prior_smoke_invalidated": PRIOR_SMOKE_INVALIDATED,
        "prior_smoke_invalidated_reason": (
            "raw_waveform_l1_was_a_hard_gate_despite_only_0_05_objective_weight"
        ),
        "loss_version": V9_PHASE_INCREMENT_LOSS_VERSION,
        "prior_loss_version": PRIOR_V9_PHASE_INCREMENT_LOSS_VERSION,
        "prior_loss_invalidated": PRIOR_V9_PHASE_INCREMENT_LOSS_INVALIDATED,
        "phase_loss_semantics": "inter_frame_increments_only_frame_zero_anchor_excluded",
        "architecture": VOCODER_GENERATOR_V9_ARCHITECTURE,
        "reconstruction_loss_version": VOCODER_LOSS_RECIPE_VERSION,
        "presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
        "grid_gate_version": VOCODER_GRID_ARTIFACT_EXCESS_VERSION,
        "grid_gate_mode": "paired_reference_relative_excess",
        "optimization_objective": (
            "v9_representation_plus_0.50_mrstft_plus_0.25_envelope_plus_0.25_presence"
        ),
        "raw_waveform_l1_is_hard_gate": False,
        "metrics_can_accept_voice_quality": False,
        "audible_acceptance_required_before_persistence": True,
        "device": "cpu",
        "utterance_id": item.utterance_id,
        "steps": steps,
        "segment_mel_frames": segment_mel_frames,
        "skipped_before_selection": len(skipped),
        "parameters": parameter_count,
        "parameter_budget_pass": parameter_budget_pass,
        "exact_factor_contract": exact_factor_contract,
        "exact_length_contract": exact_length_contract,
        "structural_finite": structural_finite,
        "factorized_spectrum_mae": round(factorized_spectrum_mae, 9),
        "factorized_waveform_mae": round(factorized_waveform_mae, 9),
        "factorization_grid_failure": factorization_grid_failure,
        "gradients_finite": gradients_finite,
        "initial": {key: round(value, 6) for key, value in initial.items()},
        "final": {key: round(value, 6) for key, value in final.items()},
        "total_decreased": total_decreased,
        "representation_decreased": representation_decreased,
        "magnitude_decreased": magnitude_decreased,
        "phase_increment_decreased": phase_increment_decreased,
        "reconstruction_decreased": reconstruction_decreased,
        "envelope_decreased": envelope_decreased,
        "presence_decreased": presence_decreased,
        "presence_error_decreased": presence_error_decreased,
        "waveform_l1_diagnostic_initial": round(initial["waveform_l1"], 6),
        "waveform_l1_diagnostic_final": round(final["waveform_l1"], 6),
        "final_grid_hop_excess": round(float(final_grid.hop_autocorrelation_excess[0]), 6),
        "final_grid_double_hop_excess": round(
            float(final_grid.double_hop_autocorrelation_excess[0]), 6
        ),
        "final_grid_harmonic_power_excess": round(
            float(final_grid.grid_harmonic_power_fraction_excess[0]), 6
        ),
        "final_grid_failure": final_grid_failure,
        "reference_wav": str(reference_path),
        "initial_wav": str(initial_path),
        "final_wav": str(final_path),
        "audio_was_gain_normalized": False,
        "audio_was_eq_processed": False,
        "audio_was_denoised": False,
        "protected_checkpoints_unchanged": checkpoints_unchanged,
        "persistent_training_started": False,
        "persistent_training_authorized": False,
        "predicted_duration_modified": False,
        "next_gate": (
            "listen_v9_bounded_oracle_reference_vs_final_before_persistence"
            if status_pass
            else "reject_or_revise_v9_before_persistent_training"
        ),
    }
    report_path = output_dir / "smoke_report.json"
    report["report_path"] = str(report_path)
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_v9_architecture_smoke(args.root, steps=args.steps),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
