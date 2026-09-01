"""Read-only architecture-coupled Loss V2 weight recalibration for minimum-phase vocoder.

Loss-weight contract v1 was correctly calibrated in waveform space before a vocoder
architecture existed.  The later parameter-authority audit proved that the fixed
minimum-phase renderer changes objective authority before the predictor itself: under v1,
spectral balance owns about 75.6% of cepstrum-space weighted gradient authority while the
envelope objective owns about 0.8% and can be anti-aligned with the combined direction.

This audit therefore derives a *candidate* architecture-coupled weight vector from the
actual trainable representation, the frame-rate real cepstrum.  It does not mutate or
reinterpret v1 and it does not authorize the candidate.  The derivation rule remains the
same transparent gradient equalization rule used previously: reconstruction is the unit
weight and every other objective is scaled by mean reconstruction-gradient norm divided by
that objective's mean gradient norm.

The cepstrum-space derivation is the candidate because it captures renderer-induced
objective geometry while remaining independent of a particular predictor parameterization.
A parameter-space derivation is reported only as a cross-check.  The cepstrum-derived
candidate is then evaluated in both spaces and in both diagnostic model states.

No optimizer is created, no parameter update is executed, and no checkpoint is loaded or
saved.  Passing this audit means only that the measurements are finite and internally
consistent.  Sensitivity and explicit review are still required before any v2 weight
contract can be frozen or any longer trainability smoke can be authorized.
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
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as renderer
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    collect_owned_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
    OWNED_VOCODER_LOSS_V2_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2_weight_contract import (
    FROZEN_WEIGHTS as WAVEFORM_SPACE_V1_WEIGHTS,
    OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION as WAVEFORM_SPACE_V1_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_parameter_gradient_audit import (
    ARCHITECTURE_CONTRACT_VERSION,
    CONNECTED_HEAD_SCALE,
    DATA_SEED,
    ITEMS_PER_SPLIT,
    MODEL_SEED,
    NOISE_SEED,
    OBJECTIVES,
    PAIR_KEYS,
    RENDERER_VERSION,
    SEGMENT_MEL_FRAMES,
    SPLITS,
    STATES,
    _flatten,
    _gradient_vectors,
    _make_model,
    _objective_tensors,
)
from lykenox_voice_engine.training.speech_vocoder_presence_v2 import (
    OWNED_VOCODER_PRESENCE_V2_VERSION,
)


AUDIT_VERSION = "owned-minimum-phase-architecture-weight-recalibration-audit-v1"
CANDIDATE_CONTRACT_VERSION = "owned-vocoder-loss-v2-minimum-phase-weight-contract-v2-candidate"
DERIVATION_SPACE = "cepstrum_space"
CROSS_CHECK_SPACE = "parameter_space"
OUTPUT_DIR_NAME = "owned_minimum_phase_architecture_weight_recalibration_audit_v1"

OPTIMIZER_CREATED = False
PARAMETER_UPDATE_EXECUTED = False
TRAINER_INSTANTIATED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False
EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED = False
WEIGHT_CONTRACT_V2_AUTHORIZED = False


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


def _norm(values: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(values.detach()))


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    ).clamp_min(1e-12)
    return float(torch.sum(first * second) / denominator)


def _finite_nonzero(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all()) and _norm(values) > 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _derive_equalized_weights(mean_norms: dict[str, float]) -> dict[str, float]:
    reference = float(mean_norms["reconstruction"])
    if not math.isfinite(reference) or reference <= 0.0:
        raise RuntimeError("reconstruction mean gradient norm is not calibratable")
    result = {"reconstruction": 1.0}
    for name in OBJECTIVES[1:]:
        norm = float(mean_norms[name])
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"{name} mean gradient norm is not calibratable")
        result[name] = reference / norm
    return result


def _space_metrics(
    gradients: dict[str, torch.Tensor],
    *,
    weights: dict[str, float],
) -> dict[str, object]:
    raw_norms = {name: _norm(gradients[name]) for name in OBJECTIVES}
    weighted = {name: gradients[name] * float(weights[name]) for name in OBJECTIVES}
    weighted_norms = {name: _norm(weighted[name]) for name in OBJECTIVES}
    norm_sum = max(sum(weighted_norms.values()), 1e-12)
    shares = {name: weighted_norms[name] / norm_sum for name in OBJECTIVES}
    combined = sum(weighted.values())
    combined_norm = _norm(combined)
    pairwise = {
        f"{first}_vs_{second}": _cosine(gradients[first], gradients[second])
        for first, second in PAIR_KEYS
    }
    alignments = {name: _cosine(gradients[name], combined) for name in OBJECTIVES}
    descent_dots = {
        name: float(torch.sum(gradients[name] * combined)) for name in OBJECTIVES
    }
    return {
        "raw_gradient_norms": raw_norms,
        "weighted_gradient_norm_shares": shares,
        "pairwise_gradient_cosines": pairwise,
        "combined_gradient_alignment_cosines": alignments,
        "first_order_descent_dots": descent_dots,
        "combined_gradient_norm": combined_norm,
        "maximum_weighted_gradient_norm_share": max(shares.values()),
        "all_objective_gradients_finite_nonzero": all(
            _finite_nonzero(gradients[name]) for name in OBJECTIVES
        ),
        "combined_gradient_finite_nonzero": _finite_nonzero(combined),
    }


def _probe_gradients(
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
    prediction, _excitation = renderer.render_owned_minimum_phase_vocoder_path(
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
    return {
        "split": segment.split,
        "utterance_id": segment.utterance_id,
        "start_frame": segment.start_frame,
        "state": state,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "cepstrum_gradients": cepstrum_gradients,
        "parameter_gradients": parameter_gradients,
        "exact_output_length": int(prediction.shape[-1])
        == int(segment.mel_frames * renderer.HOP_LENGTH),
    }


def _summarize(
    probes: list[dict[str, object]],
    *,
    gradient_key: str,
    weights: dict[str, float],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for state in STATES:
        state_probes = [probe for probe in probes if probe["state"] == state]
        metrics = [
            _space_metrics(probe[gradient_key], weights=weights) for probe in state_probes
        ]
        result[state] = {
            "probe_count": len(state_probes),
            "mean_weighted_gradient_norm_shares": {
                name: _mean(
                    [float(item["weighted_gradient_norm_shares"][name]) for item in metrics]
                )
                for name in OBJECTIVES
            },
            "minimum_weighted_gradient_norm_shares": {
                name: min(
                    [float(item["weighted_gradient_norm_shares"][name]) for item in metrics],
                    default=0.0,
                )
                for name in OBJECTIVES
            },
            "maximum_weighted_gradient_norm_shares": {
                name: max(
                    [float(item["weighted_gradient_norm_shares"][name]) for item in metrics],
                    default=0.0,
                )
                for name in OBJECTIVES
            },
            "minimum_combined_gradient_alignment_cosines": {
                name: min(
                    [
                        float(item["combined_gradient_alignment_cosines"][name])
                        for item in metrics
                    ],
                    default=0.0,
                )
                for name in OBJECTIVES
            },
            "mean_combined_gradient_alignment_cosines": {
                name: _mean(
                    [
                        float(item["combined_gradient_alignment_cosines"][name])
                        for item in metrics
                    ]
                )
                for name in OBJECTIVES
            },
            "minimum_first_order_descent_dots": {
                name: min(
                    [float(item["first_order_descent_dots"][name]) for item in metrics],
                    default=0.0,
                )
                for name in OBJECTIVES
            },
            "mean_combined_gradient_norm": _mean(
                [float(item["combined_gradient_norm"]) for item in metrics]
            ),
            "maximum_weighted_gradient_norm_share": max(
                [float(item["maximum_weighted_gradient_norm_share"]) for item in metrics],
                default=0.0,
            ),
        }
    return result


def run_audit(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected(root)
    before = {name: _sha256(path) for name, path in protected.items()}
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
                raise RuntimeError("weight recalibration received the wrong data contract")
            for state in STATES:
                probes.append(
                    _probe_gradients(
                        segment,
                        state=state,
                        envelope_objective=envelope_objective,
                    )
                )

    mean_cepstrum_norms = {
        name: _mean([_norm(probe["cepstrum_gradients"][name]) for probe in probes])
        for name in OBJECTIVES
    }
    mean_parameter_norms = {
        name: _mean([_norm(probe["parameter_gradients"][name]) for probe in probes])
        for name in OBJECTIVES
    }
    cepstrum_weights = _derive_equalized_weights(mean_cepstrum_norms)
    parameter_cross_check_weights = _derive_equalized_weights(mean_parameter_norms)
    relative_cross_check_difference = {
        name: abs(parameter_cross_check_weights[name] - cepstrum_weights[name])
        / max(abs(cepstrum_weights[name]), 1e-12)
        for name in OBJECTIVES
    }

    cepstrum_summary = _summarize(
        probes,
        gradient_key="cepstrum_gradients",
        weights=cepstrum_weights,
    )
    parameter_summary = _summarize(
        probes,
        gradient_key="parameter_gradients",
        weights=cepstrum_weights,
    )
    all_gradients_valid = all(
        _finite_nonzero(probe[gradient_key][name])
        for probe in probes
        for gradient_key in ("cepstrum_gradients", "parameter_gradients")
        for name in OBJECTIVES
    )
    exact_lengths = all(bool(probe["exact_output_length"]) for probe in probes)
    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    status_pass = (
        bool(probes)
        and all_gradients_valid
        and exact_lengths
        and checkpoints_unchanged
        and all(math.isfinite(value) and value > 0.0 for value in cepstrum_weights.values())
        and all(
            math.isfinite(value) and value > 0.0
            for value in parameter_cross_check_weights.values()
        )
    )

    public_probes = [
        {
            "split": probe["split"],
            "utterance_id": probe["utterance_id"],
            "start_frame": probe["start_frame"],
            "state": probe["state"],
            "losses": probe["losses"],
            "exact_output_length": probe["exact_output_length"],
            "cepstrum_space_candidate_metrics": _space_metrics(
                probe["cepstrum_gradients"], weights=cepstrum_weights
            ),
            "parameter_space_candidate_metrics": _space_metrics(
                probe["parameter_gradients"], weights=cepstrum_weights
            ),
        }
        for probe in probes
    ]

    report: dict[str, object] = {
        "status": "pass" if status_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "architecture_contract_version": ARCHITECTURE_CONTRACT_VERSION,
        "renderer_version": RENDERER_VERSION,
        "waveform_space_v1_contract_version": WAVEFORM_SPACE_V1_VERSION,
        "waveform_space_v1_weights": WAVEFORM_SPACE_V1_WEIGHTS.as_dict(),
        "data_contract_version": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "loss_contract_version": OWNED_VOCODER_LOSS_V2_VERSION,
        "presence_contract_version": OWNED_VOCODER_PRESENCE_V2_VERSION,
        "derivation_rule": (
            "cepstrum-space reconstruction weight=1; every other weight="
            "mean reconstruction cepstrum-gradient norm / mean objective cepstrum-gradient norm"
        ),
        "derivation_space": DERIVATION_SPACE,
        "cross_check_space": CROSS_CHECK_SPACE,
        "contract": {
            "splits": list(SPLITS),
            "states": list(STATES),
            "segment_mel_frames": SEGMENT_MEL_FRAMES,
            "items_per_split": ITEMS_PER_SPLIT,
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "noise_seed": NOISE_SEED,
            "connected_head_scale": CONNECTED_HEAD_SCALE,
        },
        "summary": {
            "mean_cepstrum_space_raw_gradient_norms": mean_cepstrum_norms,
            "mean_parameter_space_raw_gradient_norms": mean_parameter_norms,
            "derived_cepstrum_space_candidate_weights": cepstrum_weights,
            "derived_parameter_space_cross_check_weights": parameter_cross_check_weights,
            "parameter_vs_cepstrum_weight_relative_difference": relative_cross_check_difference,
            "candidate_in_cepstrum_space": cepstrum_summary,
            "candidate_in_parameter_space": parameter_summary,
        },
        "probes": public_probes,
        "probe_count": len(probes),
        "skipped_item_count": skipped_count,
        "all_objective_gradients_finite_nonzero": all_gradients_valid,
        "exact_output_length_all_probes": exact_lengths,
        "checkpoints_unchanged": checkpoints_unchanged,
        "optimizer_created": False,
        "parameter_update_executed": False,
        "trainer_instantiated": False,
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "persistent_training_started": False,
        "persistent_training_authorized": False,
        "new_vocoder_checkpoint_authorized": False,
        "extended_trainability_smoke_authorized": False,
        "weight_contract_v2_authorized": False,
        "metrics_accept_voice_quality": False,
        "third_party_model_used": False,
        "next_gate": (
            "review_architecture_coupled_weight_candidate_before_sensitivity_and_freeze"
            if status_pass
            else "revise_architecture_coupled_weight_recalibration_before_trainability"
        ),
    }
    report_path = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / OUTPUT_DIR_NAME
        / "architecture_weight_recalibration_audit.json"
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
