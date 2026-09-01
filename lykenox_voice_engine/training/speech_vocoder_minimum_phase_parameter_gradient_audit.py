"""Read-only parameter-space authority audit for the owned minimum-phase vocoder path.

The frozen Loss V2 weights were calibrated in waveform space before an architecture existed.
After the frame-rate cepstral predictor and fixed renderer passed their structural gates, the
first two-update real-data optimizer smoke showed valid descent but also an approximately
580x raw gradient norm relative to the clip threshold and a tiny envelope increase.  Before
any longer trainability smoke is authorized, this audit measures how the four frozen
objectives are transformed by the renderer/predictor Jacobians.

No optimizer is created.  No parameter update is executed.  No checkpoint is loaded or
saved.  The audit instantiates the owned predictor on small deterministic train/val V2
segments and measures objective gradients in two spaces:

* cepstrum space: after the predictor, before the fixed renderer;
* predictor parameter space: the actual gradient seen by trainable parameters.

Two model states are diagnostic only: exact neutral initialization and a deterministic
1e-4 output-head perturbation that exposes the already-proven connected upstream Jacobian.
The perturbation is not an optimizer step and is never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder.network_minimum_phase_v1 import (
    PREDICTOR_ARCHITECTURE,
    LykenoxFrameRateCepstralPredictorV1,
)
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as renderer
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    collect_owned_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    frame_grid_artifact_excess_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
    OWNED_VOCODER_LOSS_V2_VERSION,
    valid_context_multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2_weight_contract import (
    FROZEN_WEIGHTS,
    OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION,
    combine_owned_vocoder_loss_v2,
)
from lykenox_voice_engine.training.speech_vocoder_presence_v2 import (
    OWNED_VOCODER_PRESENCE_V2_VERSION,
    target_relative_presence_loss_v2,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


AUDIT_VERSION = "owned-minimum-phase-parameter-gradient-authority-audit-v1"
ARCHITECTURE_CONTRACT_VERSION = "owned-vocoder-architecture-contract-v1"
RENDERER_VERSION = "owned-minimum-phase-time-varying-renderer-v1"
SEGMENT_MEL_FRAMES = 32
ITEMS_PER_SPLIT = 2
SPLITS = ("train", "val")
STATES = ("neutral", "connected_probe")
OBJECTIVES = ("reconstruction", "envelope", "presence", "spectral_balance")
PAIR_KEYS = (
    ("reconstruction", "envelope"),
    ("reconstruction", "presence"),
    ("reconstruction", "spectral_balance"),
    ("envelope", "presence"),
    ("envelope", "spectral_balance"),
    ("presence", "spectral_balance"),
)
DATA_SEED = 20260901
MODEL_SEED = 20260903
NOISE_SEED = 97
CONNECTED_HEAD_SCALE = 1.0e-4
REFERENCE_MAX_GRAD_NORM = 1.0
OUTPUT_DIR_NAME = "owned_minimum_phase_parameter_gradient_authority_audit_v1"

OPTIMIZER_CREATED = False
PARAMETER_UPDATE_EXECUTED = False
TRAINER_INSTANTIATED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False
EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED = False


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
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _flatten(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _norm(values: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(values.detach()))


def _finite_nonzero(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all()) and _norm(values) > 0.0


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    ).clamp_min(1e-12)
    return float(torch.sum(first * second) / denominator)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _make_model(state: str) -> LykenoxFrameRateCepstralPredictorV1:
    if state not in STATES:
        raise ValueError(f"unknown diagnostic state: {state}")
    torch.manual_seed(MODEL_SEED)
    model = LykenoxFrameRateCepstralPredictorV1().cpu()
    if state == "connected_probe":
        with torch.no_grad():
            index = torch.arange(
                model.cepstral_projection.weight.numel(), dtype=torch.float32
            )
            pattern = CONNECTED_HEAD_SCALE * torch.sin(index * math.sqrt(2.0))
            model.cepstral_projection.weight.copy_(
                pattern.view_as(model.cepstral_projection.weight)
            )
    return model


def _objective_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    conditioning_log_mel: torch.Tensor,
    envelope_objective: ConditioningAlignedLogMelEnvelopeLossV2,
) -> dict[str, torch.Tensor]:
    reconstruction = valid_context_multi_resolution_reconstruction_loss(
        prediction, target
    ).total
    envelope = envelope_objective(prediction, conditioning_log_mel).total
    presence = target_relative_presence_loss_v2(
        prediction,
        target,
        sample_rate=renderer.SAMPLE_RATE,
    ).loss
    spectral_balance = target_relative_spectral_balance_loss(
        prediction,
        target,
        sample_rate=renderer.SAMPLE_RATE,
    ).loss
    return {
        "reconstruction": reconstruction,
        "envelope": envelope,
        "presence": presence,
        "spectral_balance": spectral_balance,
    }


def _gradient_vectors(
    losses: dict[str, torch.Tensor],
    *,
    cepstrum: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cepstrum_gradients: dict[str, torch.Tensor] = {}
    parameter_gradients: dict[str, torch.Tensor] = {}
    for name in OBJECTIVES:
        cepstrum_gradients[name] = torch.autograd.grad(
            losses[name],
            cepstrum,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0].detach()
        parameter_gradient_tensors = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        parameter_gradients[name] = _flatten(parameter_gradient_tensors).detach()
    return cepstrum_gradients, parameter_gradients


def _space_metrics(
    gradients: dict[str, torch.Tensor],
    *,
    direct_combined: torch.Tensor | None = None,
) -> dict[str, object]:
    weights = FROZEN_WEIGHTS.as_dict()
    raw_norms = {name: _norm(gradients[name]) for name in OBJECTIVES}
    weighted = {
        name: gradients[name] * float(weights[name]) for name in OBJECTIVES
    }
    weighted_norms = {name: _norm(weighted[name]) for name in OBJECTIVES}
    norm_sum = max(sum(weighted_norms.values()), 1e-12)
    shares = {name: weighted_norms[name] / norm_sum for name in OBJECTIVES}
    combined = sum(weighted.values())
    combined_norm = _norm(combined)
    pairwise = {
        f"{first}_vs_{second}": _cosine(gradients[first], gradients[second])
        for first, second in PAIR_KEYS
    }
    alignments = {
        name: _cosine(gradients[name], combined) for name in OBJECTIVES
    }
    descent_dots = {
        name: float(torch.sum(gradients[name] * combined)) for name in OBJECTIVES
    }
    linearity_relative_error = 0.0
    if direct_combined is not None:
        denominator = torch.linalg.vector_norm(direct_combined).clamp_min(1e-12)
        linearity_relative_error = float(
            torch.linalg.vector_norm(direct_combined - combined) / denominator
        )
    clip_scale = min(1.0, REFERENCE_MAX_GRAD_NORM / max(combined_norm, 1e-12))
    return {
        "raw_gradient_norms": raw_norms,
        "weighted_gradient_norms": weighted_norms,
        "weighted_gradient_norm_shares": shares,
        "pairwise_gradient_cosines": pairwise,
        "combined_gradient_alignment_cosines": alignments,
        "first_order_descent_dots": descent_dots,
        "combined_gradient_norm": combined_norm,
        "clip_scale_if_max_norm_1": clip_scale,
        "maximum_weighted_gradient_norm_share": max(shares.values()),
        "all_objective_gradients_finite_nonzero": all(
            _finite_nonzero(gradients[name]) for name in OBJECTIVES
        ),
        "combined_gradient_finite_nonzero": _finite_nonzero(combined),
        "combined_gradient_linearity_relative_error": linearity_relative_error,
    }


def _probe(
    segment: Any,
    *,
    state: str,
    envelope_objective: ConditioningAlignedLogMelEnvelopeLossV2,
) -> dict[str, object]:
    model = _make_model(state)
    parameters = tuple(model.parameters())
    mel = segment.mel.unsqueeze(0).cpu()
    f0_hz = segment.f0_hz.unsqueeze(0).cpu()
    voiced = segment.voiced.unsqueeze(0).cpu()
    periodicity = segment.periodicity.unsqueeze(0).cpu()
    target = segment.waveform.unsqueeze(0).cpu()

    cepstrum = model(mel, f0_hz, voiced, periodicity)
    prediction, excitation = renderer.render_owned_minimum_phase_vocoder_path(
        cepstrum,
        f0_hz,
        voiced,
        periodicity,
        noise_seed=NOISE_SEED,
    )
    losses = _objective_tensors(prediction, target, mel, envelope_objective)
    cepstrum_gradients, parameter_gradients = _gradient_vectors(
        losses,
        cepstrum=cepstrum,
        parameters=parameters,
    )

    combined_loss = combine_owned_vocoder_loss_v2(**losses)
    direct_parameter_gradient = _flatten(
        torch.autograd.grad(
            combined_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
    ).detach()
    direct_cepstrum_gradient = torch.autograd.grad(
        combined_loss,
        cepstrum,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0].detach()

    parameter_space = _space_metrics(
        parameter_gradients,
        direct_combined=direct_parameter_gradient,
    )
    cepstrum_space = _space_metrics(
        cepstrum_gradients,
        direct_combined=direct_cepstrum_gradient,
    )
    grid = frame_grid_artifact_excess_metrics(
        prediction,
        excitation,
        sample_rate=renderer.SAMPLE_RATE,
        hop_length=renderer.HOP_LENGTH,
    )
    return {
        "split": segment.split,
        "utterance_id": segment.utterance_id,
        "start_frame": segment.start_frame,
        "state": state,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "total_loss": float(combined_loss.detach()),
        "parameter_space": parameter_space,
        "cepstrum_space": cepstrum_space,
        "prediction_samples": int(prediction.shape[-1]),
        "expected_samples": int(segment.mel_frames * renderer.HOP_LENGTH),
        "exact_output_length": int(prediction.shape[-1])
        == int(segment.mel_frames * renderer.HOP_LENGTH),
        "hop_autocorrelation_excess": float(grid.hop_autocorrelation_excess.max()),
        "double_hop_autocorrelation_excess": float(
            grid.double_hop_autocorrelation_excess.max()
        ),
        "grid_harmonic_power_fraction_excess": float(
            grid.grid_harmonic_power_fraction_excess.max()
        ),
        "severe_grid_excess": bool(grid.severe_grid_excess.any()),
    }


def _summarize_space(probes: list[dict[str, object]], space: str) -> dict[str, object]:
    summary: dict[str, object] = {}
    for state in STATES:
        state_probes = [probe for probe in probes if probe["state"] == state]
        raw_norms: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
        shares: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
        alignments: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
        dots: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
        pairs: dict[str, list[float]] = {
            f"{first}_vs_{second}": [] for first, second in PAIR_KEYS
        }
        combined_norms: list[float] = []
        clip_scales: list[float] = []
        linearity_errors: list[float] = []
        maximum_shares: list[float] = []
        for probe in state_probes:
            metrics = probe[space]
            for name in OBJECTIVES:
                raw_norms[name].append(float(metrics["raw_gradient_norms"][name]))
                shares[name].append(float(metrics["weighted_gradient_norm_shares"][name]))
                alignments[name].append(
                    float(metrics["combined_gradient_alignment_cosines"][name])
                )
                dots[name].append(float(metrics["first_order_descent_dots"][name]))
            for key, value in metrics["pairwise_gradient_cosines"].items():
                pairs[key].append(float(value))
            combined_norms.append(float(metrics["combined_gradient_norm"]))
            clip_scales.append(float(metrics["clip_scale_if_max_norm_1"]))
            linearity_errors.append(
                float(metrics["combined_gradient_linearity_relative_error"])
            )
            maximum_shares.append(float(metrics["maximum_weighted_gradient_norm_share"]))

        summary[state] = {
            "probe_count": len(state_probes),
            "mean_raw_gradient_norms": {
                name: _mean(raw_norms[name]) for name in OBJECTIVES
            },
            "mean_weighted_gradient_norm_shares": {
                name: _mean(shares[name]) for name in OBJECTIVES
            },
            "minimum_weighted_gradient_norm_shares": {
                name: min(shares[name], default=0.0) for name in OBJECTIVES
            },
            "maximum_weighted_gradient_norm_shares": {
                name: max(shares[name], default=0.0) for name in OBJECTIVES
            },
            "mean_pairwise_gradient_cosines": {
                key: _mean(values) for key, values in pairs.items()
            },
            "minimum_pairwise_gradient_cosines": {
                key: min(values, default=0.0) for key, values in pairs.items()
            },
            "mean_combined_gradient_alignment_cosines": {
                name: _mean(alignments[name]) for name in OBJECTIVES
            },
            "minimum_combined_gradient_alignment_cosines": {
                name: min(alignments[name], default=0.0) for name in OBJECTIVES
            },
            "minimum_first_order_descent_dots": {
                name: min(dots[name], default=0.0) for name in OBJECTIVES
            },
            "mean_combined_gradient_norm": _mean(combined_norms),
            "minimum_clip_scale_if_max_norm_1": min(clip_scales, default=0.0),
            "mean_clip_scale_if_max_norm_1": _mean(clip_scales),
            "maximum_weighted_gradient_norm_share": max(maximum_shares, default=0.0),
            "maximum_combined_gradient_linearity_relative_error": max(
                linearity_errors, default=0.0
            ),
        }
    return summary


def run_audit(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected(root)
    checkpoints_before = {name: _sha256(path) for name, path in protected.items()}
    envelope_objective = ConditioningAlignedLogMelEnvelopeLossV2(
        LykenoxSpeechConfig()
    ).cpu()

    probes: list[dict[str, object]] = []
    skipped_count = 0
    for split in SPLITS:
        segments, skipped = collect_owned_vocoder_segments(
            root,
            split,
            segment_mel_frames=SEGMENT_MEL_FRAMES,
            max_items=ITEMS_PER_SPLIT,
            seed=DATA_SEED,
        )
        skipped_count += len(skipped)
        for segment in segments:
            if segment.conditioning_contract_version != OWNED_VOCODER_SEGMENT_CONTRACT_VERSION:
                raise RuntimeError("parameter-gradient audit received the wrong data contract")
            for state in STATES:
                probes.append(
                    _probe(
                        segment,
                        state=state,
                        envelope_objective=envelope_objective,
                    )
                )

    checkpoints_after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = checkpoints_before == checkpoints_after
    all_space_metrics = [
        probe[space]
        for probe in probes
        for space in ("parameter_space", "cepstrum_space")
    ]
    all_finite_nonzero = all(
        bool(metrics["all_objective_gradients_finite_nonzero"])
        and bool(metrics["combined_gradient_finite_nonzero"])
        for metrics in all_space_metrics
    )
    linearity_exact = all(
        float(metrics["combined_gradient_linearity_relative_error"]) <= 1e-5
        for metrics in all_space_metrics
    )
    exact_lengths = all(bool(probe["exact_output_length"]) for probe in probes)
    no_severe_grid = all(not bool(probe["severe_grid_excess"]) for probe in probes)
    status_pass = (
        bool(probes)
        and all_finite_nonzero
        and linearity_exact
        and exact_lengths
        and no_severe_grid
        and checkpoints_unchanged
    )

    report: dict[str, object] = {
        "status": "pass" if status_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "architecture_contract_version": ARCHITECTURE_CONTRACT_VERSION,
        "renderer_version": RENDERER_VERSION,
        "predictor_architecture": PREDICTOR_ARCHITECTURE,
        "data_contract_version": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "loss_contract_version": OWNED_VOCODER_LOSS_V2_VERSION,
        "presence_contract_version": OWNED_VOCODER_PRESENCE_V2_VERSION,
        "loss_weight_contract_version": OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION,
        "frozen_weights": FROZEN_WEIGHTS.as_dict(),
        "contract": {
            "splits": list(SPLITS),
            "states": list(STATES),
            "segment_mel_frames": SEGMENT_MEL_FRAMES,
            "items_per_split": ITEMS_PER_SPLIT,
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "noise_seed": NOISE_SEED,
            "connected_head_scale": CONNECTED_HEAD_SCALE,
            "reference_max_grad_norm": REFERENCE_MAX_GRAD_NORM,
        },
        "summary": {
            "parameter_space": _summarize_space(probes, "parameter_space"),
            "cepstrum_space": _summarize_space(probes, "cepstrum_space"),
        },
        "probes": probes,
        "probe_count": len(probes),
        "skipped_item_count": skipped_count,
        "all_objective_gradients_finite_nonzero": all_finite_nonzero,
        "combined_gradient_linearity_exact": linearity_exact,
        "exact_output_length_all_probes": exact_lengths,
        "no_severe_grid_excess_all_probes": no_severe_grid,
        "checkpoints_unchanged": checkpoints_unchanged,
        "model_instantiated": True,
        "optimizer_created": False,
        "parameter_update_executed": False,
        "trainer_instantiated": False,
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "persistent_training_started": False,
        "persistent_training_authorized": False,
        "new_vocoder_checkpoint_authorized": False,
        "extended_trainability_smoke_authorized": False,
        "metrics_accept_voice_quality": False,
        "third_party_model_used": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "next_gate": (
            "review_predictor_parameter_space_loss_v2_authority_and_clip_regime_before_extended_trainability"
            if status_pass
            else "revise_predictor_or_objective_parameter_space_contract_before_extended_trainability"
        ),
    }
    report_path = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / OUTPUT_DIR_NAME
        / "parameter_gradient_authority_audit.json"
    )
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_audit(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
