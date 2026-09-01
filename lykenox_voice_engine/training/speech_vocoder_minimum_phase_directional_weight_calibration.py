"""Deterministic pre-training fixed-weight calibration for the owned minimum-phase vocoder.

The previous minimum-phase v2 contract equalized gradient *norms* but ignored gradient
*direction*.  That is insufficient: equally sized objectives can still point against one
another.  This module calibrates one fixed positive loss-weight vector before model training
by maximizing the worst common-descent cosine across deterministic owned gradient Gram
matrices.  The selected weights are then frozen for the whole run and exact-resume contract.

This is not adaptive training-time reweighting.  It creates no model optimizer, performs no
parameter update, loads no third-party model and changes no duration or audio postprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


CALIBRATION_VERSION = "owned-minimum-phase-directional-fixed-weight-calibration-v1"
OBJECTIVES = ("reconstruction", "envelope", "presence", "spectral_balance")
REFERENCE_OBJECTIVE = "reconstruction"
GRID_LOG10_MIN = -4.0
GRID_LOG10_MAX = 4.0
GRID_POINTS = 41
MIN_WORST_ALIGNMENT = 1.0e-4
MIN_MEAN_AUTHORITY_SHARE = 0.02
MAX_MEAN_AUTHORITY_SHARE = 0.80

MODEL_OPTIMIZER_CREATED = False
MODEL_PARAMETER_UPDATE_EXECUTED = False
ADAPTIVE_DURING_TRAINING = False
RUNTIME_REDERIVATION_AFTER_OPTIMIZER_CREATION = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_USED = False


@dataclass(frozen=True)
class DirectionalFixedWeights:
    reconstruction: float
    envelope: float
    presence: float
    spectral_balance: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reconstruction": float(self.reconstruction),
            "envelope": float(self.envelope),
            "presence": float(self.presence),
            "spectral_balance": float(self.spectral_balance),
        }

    def as_tensor(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return torch.tensor(
            [self.reconstruction, self.envelope, self.presence, self.spectral_balance],
            dtype=dtype,
        )


@dataclass(frozen=True)
class GradientGramProbe:
    probe_id: str
    space: str
    state: str
    gram: torch.Tensor

    def validated_gram(self) -> torch.Tensor:
        value = self.gram.detach().cpu().to(torch.float64)
        if tuple(value.shape) != (4, 4):
            raise ValueError("gradient Gram matrix must be 4x4")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("gradient Gram matrix must be finite")
        value = (value + value.T) * 0.5
        diagonal = torch.diagonal(value)
        if bool((diagonal <= 0.0).any()):
            raise ValueError("every objective gradient must have positive squared norm")
        return value


def gradient_gram(gradients: dict[str, torch.Tensor]) -> torch.Tensor:
    vectors = []
    for name in OBJECTIVES:
        value = gradients[name].detach().reshape(-1).to(torch.float64).cpu()
        if not bool(torch.isfinite(value).all()) or float(torch.linalg.vector_norm(value)) <= 0.0:
            raise ValueError(f"objective gradient is non-finite or zero: {name}")
        vectors.append(value)
    return torch.stack(
        [torch.stack([torch.dot(first, second) for second in vectors]) for first in vectors]
    )


def _candidate_matrix() -> torch.Tensor:
    axis = torch.logspace(
        GRID_LOG10_MIN,
        GRID_LOG10_MAX,
        GRID_POINTS,
        base=10.0,
        dtype=torch.float64,
    )
    envelope, presence, balance = torch.meshgrid(axis, axis, axis, indexing="ij")
    ones = torch.ones_like(envelope)
    return torch.stack((ones, envelope, presence, balance), dim=-1).reshape(-1, 4)


def _evaluate_candidates(
    candidates: torch.Tensor,
    probes: list[GradientGramProbe],
) -> dict[str, torch.Tensor]:
    if not probes:
        raise ValueError("at least one gradient Gram probe is required")
    count = candidates.shape[0]
    worst_alignment = torch.full((count,), float("inf"), dtype=torch.float64)
    share_sum = torch.zeros((count, 4), dtype=torch.float64)
    minimum_descent_dot = torch.full((count,), float("inf"), dtype=torch.float64)
    for probe in probes:
        gram = probe.validated_gram()
        dots = candidates @ gram.T
        combined_squared_norm = (candidates * dots).sum(dim=1).clamp_min(1.0e-30)
        combined_norm = torch.sqrt(combined_squared_norm)
        objective_norms = torch.sqrt(torch.diagonal(gram)).clamp_min(1.0e-15)
        alignments = dots / (combined_norm.unsqueeze(1) * objective_norms.unsqueeze(0))
        worst_alignment = torch.minimum(worst_alignment, alignments.min(dim=1).values)
        minimum_descent_dot = torch.minimum(minimum_descent_dot, dots.min(dim=1).values)
        weighted_norms = candidates * objective_norms.unsqueeze(0)
        shares = weighted_norms / weighted_norms.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
        share_sum += shares
    mean_shares = share_sum / float(len(probes))
    return {
        "worst_alignment": worst_alignment,
        "minimum_descent_dot": minimum_descent_dot,
        "mean_shares": mean_shares,
        "minimum_mean_share": mean_shares.min(dim=1).values,
        "maximum_mean_share": mean_shares.max(dim=1).values,
        "share_spread": mean_shares.max(dim=1).values - mean_shares.min(dim=1).values,
    }


def summarize_weights(
    weights: DirectionalFixedWeights,
    probes: list[GradientGramProbe],
) -> dict[str, object]:
    candidate = weights.as_tensor().view(1, 4)
    measured = _evaluate_candidates(candidate, probes)
    mean_shares = measured["mean_shares"][0]
    return {
        "weights": weights.as_dict(),
        "worst_alignment": float(measured["worst_alignment"][0]),
        "minimum_descent_dot": float(measured["minimum_descent_dot"][0]),
        "mean_weighted_gradient_norm_shares": {
            name: float(mean_shares[index]) for index, name in enumerate(OBJECTIVES)
        },
        "minimum_mean_weighted_gradient_norm_share": float(
            measured["minimum_mean_share"][0]
        ),
        "maximum_mean_weighted_gradient_norm_share": float(
            measured["maximum_mean_share"][0]
        ),
    }


def calibrate_directional_fixed_weights(
    probes: list[GradientGramProbe],
) -> tuple[DirectionalFixedWeights | None, dict[str, object]]:
    candidates = _candidate_matrix()
    measured = _evaluate_candidates(candidates, probes)
    feasible = (
        (measured["worst_alignment"] >= MIN_WORST_ALIGNMENT)
        & (measured["minimum_descent_dot"] > 0.0)
        & (measured["minimum_mean_share"] >= MIN_MEAN_AUTHORITY_SHARE)
        & (measured["maximum_mean_share"] <= MAX_MEAN_AUTHORITY_SHARE)
    )
    feasible_count = int(feasible.sum())
    if feasible_count == 0:
        best_index = int(torch.argmax(measured["worst_alignment"]))
        best = candidates[best_index]
        diagnostic = DirectionalFixedWeights(*[float(value) for value in best])
        return None, {
            "status": "no_static_positive_weight_solution",
            "calibration_version": CALIBRATION_VERSION,
            "candidate_count": int(candidates.shape[0]),
            "feasible_candidate_count": 0,
            "best_unconstrained_candidate": summarize_weights(diagnostic, probes),
            "model_optimizer_created": False,
            "model_parameter_update_executed": False,
            "adaptive_during_training": False,
        }

    # Maximize the weakest objective alignment first; use authority spread only as a
    # deterministic tie-breaker.  Scale is fixed by reconstruction=1.
    score = measured["worst_alignment"] - 1.0e-3 * measured["share_spread"]
    score = torch.where(feasible, score, torch.full_like(score, -float("inf")))
    index = int(torch.argmax(score))
    selected = candidates[index]
    weights = DirectionalFixedWeights(*[float(value) for value in selected])
    summary = summarize_weights(weights, probes)
    return weights, {
        "status": "pass",
        "calibration_version": CALIBRATION_VERSION,
        "candidate_count": int(candidates.shape[0]),
        "feasible_candidate_count": feasible_count,
        "selected": summary,
        "model_optimizer_created": False,
        "model_parameter_update_executed": False,
        "adaptive_during_training": False,
        "runtime_rederivation_after_optimizer_creation": False,
    }


def fixed_weights_from_mapping(values: dict[str, object]) -> DirectionalFixedWeights:
    missing = [name for name in OBJECTIVES if name not in values]
    if missing:
        raise ValueError(f"fixed weight mapping is missing objectives: {missing}")
    numbers = [float(values[name]) for name in OBJECTIVES]
    if any(not math.isfinite(value) or value <= 0.0 for value in numbers):
        raise ValueError("fixed loss weights must be finite and positive")
    if abs(numbers[0] - 1.0) > 1.0e-12:
        raise ValueError("directional fixed weights must normalize reconstruction to 1")
    return DirectionalFixedWeights(*numbers)


__all__ = [
    "CALIBRATION_VERSION",
    "OBJECTIVES",
    "DirectionalFixedWeights",
    "GradientGramProbe",
    "gradient_gram",
    "calibrate_directional_fixed_weights",
    "summarize_weights",
    "fixed_weights_from_mapping",
]
