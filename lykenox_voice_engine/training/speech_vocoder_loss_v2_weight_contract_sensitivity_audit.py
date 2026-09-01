"""Read-only robustness audit for the proposed owned LYKENOX Loss V2 weight contract.

The preceding four-objective calibration established that reconstruction, envelope,
valid-context presence, and target-relative spectral balance can all receive material
waveform-gradient authority without destructive first-order conflict.  This audit asks a
narrower question before freezing those weights: is the calibrated direction robust to
small changes in the relative weights, or does it only work at one knife-edge vector?

No vocoder model, optimizer, trainer, checkpoint mutation, duration modification, or
post-hoc audio processing is involved.  Gradients are taken only with respect to controlled
waveform perturbations around real owned speech segments.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_vocoder_data import collect_owned_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2_gradient_balance_audit import (
    _diagnostic_candidates,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2_weight_calibration_audit import (
    OBJECTIVES,
    _derive_equalized_weights,
    _finite_nonzero,
    _norm,
    _objective_gradients,
)


AUDIT_VERSION = "owned-vocoder-loss-v2-weight-contract-sensitivity-audit-v1"
DEFAULT_SEGMENT_MEL_FRAMES = 64
DEFAULT_ITEMS_PER_SPLIT = 3
SPLITS = ("train", "val")
OUTPUT_DIR_NAME = "owned_vocoder_loss_v2_weight_contract_sensitivity_v1"

# Human-readable rounding of the successful four-objective calibration.  These values are
# only a candidate until this sensitivity audit passes and the project decision explicitly
# freezes them.
CANDIDATE_WEIGHTS = {
    "reconstruction": 1.0,
    "envelope": 3.1475,
    "presence": 19.3369,
    "spectral_balance": 60.9496,
}

RELATIVE_PERTURBATION = 0.10
MAX_CANDIDATE_DERIVATION_RELATIVE_ERROR = 0.005
AUTHORITY_RETENTION_FRACTION = 0.75
ALIGNMENT_RETENTION_FRACTION = 0.75
MAX_DOMINANCE_EXPANSION = 1.20


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    ).clamp_min(1e-12)
    return float(torch.sum(first * second) / denominator)


def _normalize_to_reconstruction(weights: dict[str, float]) -> dict[str, float]:
    reference = float(weights["reconstruction"])
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("reconstruction weight must be finite and positive")
    normalized = {name: float(weights[name]) / reference for name in OBJECTIVES}
    normalized["reconstruction"] = 1.0
    return normalized


def _scenario_key(weights: dict[str, float]) -> tuple[float, ...]:
    return tuple(round(float(weights[name]), 10) for name in OBJECTIVES)


def build_weight_scenarios() -> dict[str, dict[str, float]]:
    """Return baseline, one-at-a-time, and all-corner +/-10% relative-weight probes."""

    scenarios: dict[str, dict[str, float]] = {
        "baseline": dict(CANDIDATE_WEIGHTS),
    }
    seen = {_scenario_key(CANDIDATE_WEIGHTS)}

    for name in OBJECTIVES:
        for sign, factor in (("minus", 1.0 - RELATIVE_PERTURBATION), ("plus", 1.0 + RELATIVE_PERTURBATION)):
            raw = dict(CANDIDATE_WEIGHTS)
            raw[name] *= factor
            normalized = _normalize_to_reconstruction(raw)
            key = _scenario_key(normalized)
            if key not in seen:
                scenarios[f"{name}_{sign}_10pct"] = normalized
                seen.add(key)

    factors = (1.0 - RELATIVE_PERTURBATION, 1.0 + RELATIVE_PERTURBATION)
    for index, combination in enumerate(itertools.product(factors, repeat=len(OBJECTIVES))):
        raw = {
            name: float(CANDIDATE_WEIGHTS[name]) * float(factor)
            for name, factor in zip(OBJECTIVES, combination)
        }
        normalized = _normalize_to_reconstruction(raw)
        key = _scenario_key(normalized)
        if key not in seen:
            scenarios[f"corner_{index:02d}"] = normalized
            seen.add(key)

    return scenarios


def _evaluate_weights(
    gradients: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> dict[str, Any]:
    weighted = {
        name: gradients[name] * float(weights[name])
        for name in OBJECTIVES
    }
    combined = sum(weighted.values())
    weighted_norms = {name: _norm(weighted[name]) for name in OBJECTIVES}
    norm_sum = max(sum(weighted_norms.values()), 1e-12)
    shares = {name: weighted_norms[name] / norm_sum for name in OBJECTIVES}
    alignments = {name: _cosine(gradients[name], combined) for name in OBJECTIVES}
    descent_dots = {
        name: float(torch.sum(gradients[name] * combined)) for name in OBJECTIVES
    }
    return {
        "combined_finite_nonzero": _finite_nonzero(combined),
        "shares": shares,
        "alignments": alignments,
        "descent_dots": descent_dots,
        "combined_norm": _norm(combined),
    }


def run_owned_vocoder_loss_v2_weight_contract_sensitivity_audit(
    root: Path,
    *,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    items_per_split: int = DEFAULT_ITEMS_PER_SPLIT,
    seed: int = 4242,
) -> dict[str, object]:
    root = Path(root).resolve()
    config = LykenoxSpeechConfig()
    envelope_objective = ConditioningAlignedLogMelEnvelopeLossV2(config).cpu()
    protected = _protected(root)
    before = {name: _sha256(path) for name, path in protected.items()}

    raw_probes: list[dict[str, Any]] = []
    norm_values: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
    all_objective_gradients_valid = True

    for split in SPLITS:
        segments, _skipped = collect_owned_vocoder_segments(
            root,
            split,
            segment_mel_frames=segment_mel_frames,
            max_items=items_per_split,
            seed=seed,
        )
        for segment in segments:
            target = segment.waveform.unsqueeze(0)
            conditioning = segment.mel.unsqueeze(0)
            for perturbation_name, candidate in _diagnostic_candidates(
                target,
                sample_rate=config.sample_rate,
            ).items():
                losses, gradients, diagnostics = _objective_gradients(
                    candidate,
                    target,
                    conditioning,
                    envelope_objective,
                    sample_rate=config.sample_rate,
                )
                valid = all(_finite_nonzero(gradients[name]) for name in OBJECTIVES)
                all_objective_gradients_valid = all_objective_gradients_valid and valid
                for name in OBJECTIVES:
                    norm_values[name].append(_norm(gradients[name]))
                raw_probes.append(
                    {
                        "split": split,
                        "utterance_id": segment.utterance_id,
                        "start_frame": segment.start_frame,
                        "perturbation": perturbation_name,
                        "losses": losses,
                        "diagnostics": diagnostics,
                        "gradients": gradients,
                    }
                )

    mean_norms = {name: _mean(norm_values[name]) for name in OBJECTIVES}
    rederived_weights = _derive_equalized_weights(mean_norms)
    candidate_relative_errors = {
        name: abs(float(CANDIDATE_WEIGHTS[name]) - float(rederived_weights[name]))
        / max(abs(float(rederived_weights[name])), 1e-12)
        for name in OBJECTIVES
    }
    candidate_tracks_derivation = max(candidate_relative_errors.values(), default=float("inf")) <= MAX_CANDIDATE_DERIVATION_RELATIVE_ERROR

    scenarios = build_weight_scenarios()
    scenario_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in scenarios}
    all_combined_valid = True

    for scenario_name, weights in scenarios.items():
        for probe in raw_probes:
            result = _evaluate_weights(probe["gradients"], weights)
            all_combined_valid = all_combined_valid and bool(result["combined_finite_nonzero"])
            scenario_rows[scenario_name].append(result)

    baseline_rows = scenario_rows["baseline"]
    baseline_min_shares = {
        name: min((row["shares"][name] for row in baseline_rows), default=0.0)
        for name in OBJECTIVES
    }
    baseline_min_alignments = {
        name: min((row["alignments"][name] for row in baseline_rows), default=0.0)
        for name in OBJECTIVES
    }
    baseline_max_share = max(
        (
            row["shares"][name]
            for row in baseline_rows
            for name in OBJECTIVES
        ),
        default=0.0,
    )

    all_min_shares = {
        name: min(
            (
                row["shares"][name]
                for rows in scenario_rows.values()
                for row in rows
            ),
            default=0.0,
        )
        for name in OBJECTIVES
    }
    all_min_alignments = {
        name: min(
            (
                row["alignments"][name]
                for rows in scenario_rows.values()
                for row in rows
            ),
            default=0.0,
        )
        for name in OBJECTIVES
    }
    all_min_descent_dots = {
        name: min(
            (
                row["descent_dots"][name]
                for rows in scenario_rows.values()
                for row in rows
            ),
            default=0.0,
        )
        for name in OBJECTIVES
    }
    all_max_share = max(
        (
            row["shares"][name]
            for rows in scenario_rows.values()
            for row in rows
            for name in OBJECTIVES
        ),
        default=0.0,
    )

    authority_retained = all(
        all_min_shares[name] >= AUTHORITY_RETENTION_FRACTION * baseline_min_shares[name]
        for name in OBJECTIVES
    )
    alignment_positive = all(value > 0.0 for value in all_min_alignments.values())
    alignment_retained = all(
        all_min_alignments[name] >= ALIGNMENT_RETENTION_FRACTION * baseline_min_alignments[name]
        for name in OBJECTIVES
    )
    descent_positive = all(value > 0.0 for value in all_min_descent_dots.values())
    dominance_bounded = all_max_share <= MAX_DOMINANCE_EXPANSION * baseline_max_share

    scenario_summary: dict[str, object] = {}
    for scenario_name, rows in scenario_rows.items():
        scenario_summary[scenario_name] = {
            "weights": {
                name: round(float(scenarios[scenario_name][name]), 10)
                for name in OBJECTIVES
            },
            "minimum_weighted_gradient_norm_shares": {
                name: round(min((row["shares"][name] for row in rows), default=0.0), 6)
                for name in OBJECTIVES
            },
            "maximum_weighted_gradient_norm_shares": {
                name: round(max((row["shares"][name] for row in rows), default=0.0), 6)
                for name in OBJECTIVES
            },
            "minimum_combined_gradient_alignment_cosines": {
                name: round(min((row["alignments"][name] for row in rows), default=0.0), 6)
                for name in OBJECTIVES
            },
            "minimum_first_order_descent_dots": {
                name: round(min((row["descent_dots"][name] for row in rows), default=0.0), 10)
                for name in OBJECTIVES
            },
        }

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    robustness_pass = (
        bool(raw_probes)
        and all_objective_gradients_valid
        and all_combined_valid
        and candidate_tracks_derivation
        and authority_retained
        and alignment_positive
        and alignment_retained
        and descent_positive
        and dominance_bounded
        and checkpoints_unchanged
    )

    summary = {
        "candidate_weights": {
            name: round(float(CANDIDATE_WEIGHTS[name]), 10) for name in OBJECTIVES
        },
        "rederived_equalization_weights": {
            name: round(float(rederived_weights[name]), 10) for name in OBJECTIVES
        },
        "candidate_derivation_relative_errors": {
            name: round(float(candidate_relative_errors[name]), 8) for name in OBJECTIVES
        },
        "scenario_count": len(scenarios),
        "relative_weight_perturbation": RELATIVE_PERTURBATION,
        "baseline_minimum_weighted_gradient_norm_shares": {
            name: round(float(baseline_min_shares[name]), 6) for name in OBJECTIVES
        },
        "all_scenarios_minimum_weighted_gradient_norm_shares": {
            name: round(float(all_min_shares[name]), 6) for name in OBJECTIVES
        },
        "baseline_minimum_combined_gradient_alignment_cosines": {
            name: round(float(baseline_min_alignments[name]), 6) for name in OBJECTIVES
        },
        "all_scenarios_minimum_combined_gradient_alignment_cosines": {
            name: round(float(all_min_alignments[name]), 6) for name in OBJECTIVES
        },
        "all_scenarios_minimum_first_order_descent_dots": {
            name: round(float(all_min_descent_dots[name]), 10) for name in OBJECTIVES
        },
        "baseline_maximum_weighted_gradient_norm_share": round(float(baseline_max_share), 6),
        "all_scenarios_maximum_weighted_gradient_norm_share": round(float(all_max_share), 6),
        "candidate_tracks_derivation": candidate_tracks_derivation,
        "authority_retained": authority_retained,
        "alignment_positive": alignment_positive,
        "alignment_retained": alignment_retained,
        "descent_positive": descent_positive,
        "dominance_bounded": dominance_bounded,
    }

    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    report_path = output_dir / "loss_v2_weight_contract_sensitivity_audit.json"
    report: dict[str, object] = {
        "status": "pass" if robustness_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "summary": summary,
        "scenarios": scenario_summary,
        "all_objective_gradients_finite_nonzero": all_objective_gradients_valid,
        "all_combined_gradients_finite_nonzero": all_combined_valid,
        "checkpoints_unchanged": checkpoints_unchanged,
        "training_started": False,
        "optimizer_created": False,
        "model_instantiated": False,
        "persistent_training_authorized": False,
        "loss_weight_contract_authorized": False,
        "new_vocoder_architecture_authorized": False,
        "third_party_model_used": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "report_path": str(report_path),
        "next_gate": (
            "freeze_owned_vocoder_loss_v2_weight_contract_if_reviewed"
            if robustness_pass
            else "reject_or_recalibrate_owned_vocoder_loss_v2_weight_candidate"
        ),
    }
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_owned_vocoder_loss_v2_weight_contract_sensitivity_audit(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
