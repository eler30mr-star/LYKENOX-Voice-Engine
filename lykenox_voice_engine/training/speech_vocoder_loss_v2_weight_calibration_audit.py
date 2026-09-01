"""Read-only four-objective gradient calibration for owned LYKENOX vocoder Loss V2.

This audit runs before any new vocoder architecture is selected.  The differentiable
variable is a controlled perturbation of a real LYKENOX waveform segment, never model
parameters.  It measures reconstruction V2, conditioning-aligned envelope V2, valid-context
presence V2, and target-relative broad spectral balance.

Candidate weights are derived from real gradient norms instead of chosen by hand:
reconstruction is the unit weight and each other objective is scaled by
mean_norm(reconstruction) / mean_norm(objective).  This merely proposes a gradient-scale
calibration.  It does not authorize those weights, an architecture, or persistent training.
The report also checks pairwise directions, combined-gradient alignment, and first-order
descent dot products so norm equalization cannot hide destructive objective conflict.
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
from lykenox_voice_engine.training.speech_vocoder_data import collect_owned_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
    valid_context_multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2_gradient_balance_audit import (
    _diagnostic_candidates,
)
from lykenox_voice_engine.training.speech_vocoder_presence_v2 import (
    target_relative_presence_loss_v2,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


AUDIT_VERSION = "owned-vocoder-loss-v2-four-objective-weight-calibration-audit-v1"
DEFAULT_SEGMENT_MEL_FRAMES = 64
DEFAULT_ITEMS_PER_SPLIT = 3
SPLITS = ("train", "val")
OBJECTIVES = ("reconstruction", "envelope", "presence", "spectral_balance")
PAIR_KEYS = (
    ("reconstruction", "envelope"),
    ("reconstruction", "presence"),
    ("reconstruction", "spectral_balance"),
    ("envelope", "presence"),
    ("envelope", "spectral_balance"),
    ("presence", "spectral_balance"),
)
OUTPUT_DIR_NAME = "owned_vocoder_loss_v2_four_objective_weight_calibration_v1"


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
    return float(torch.linalg.vector_norm(values))


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    ).clamp_min(1e-12)
    return float(torch.sum(first * second) / denominator)


def _gradient(loss: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        loss,
        candidate,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0].detach()


def _finite_nonzero(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all()) and _norm(values) > 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _objective_gradients(
    candidate_values: torch.Tensor,
    target: torch.Tensor,
    conditioning: torch.Tensor,
    envelope_objective: ConditioningAlignedLogMelEnvelopeLossV2,
    *,
    sample_rate: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor], dict[str, float]]:
    candidate = candidate_values.detach().clone().requires_grad_(True)
    reconstruction = valid_context_multi_resolution_reconstruction_loss(candidate, target)
    envelope = envelope_objective(candidate, conditioning)
    presence = target_relative_presence_loss_v2(
        candidate,
        target,
        sample_rate=sample_rate,
    )
    spectral_balance = target_relative_spectral_balance_loss(
        candidate,
        target,
        sample_rate=sample_rate,
    )
    tensors = {
        "reconstruction": reconstruction.total,
        "envelope": envelope.total,
        "presence": presence.loss,
        "spectral_balance": spectral_balance.loss,
    }
    gradients = {name: _gradient(loss, candidate) for name, loss in tensors.items()}
    losses = {name: float(loss.detach()) for name, loss in tensors.items()}
    diagnostics = {
        "presence_1k_8k_error_db": float(presence.presence_1k_8k_error_db.detach()),
        "presence_valid_frame_count": float(presence.valid_frame_count),
        "presence_analysis_frame_count": float(presence.analysis_frame_count),
    }
    return losses, gradients, diagnostics


def _derive_equalized_weights(mean_norms: dict[str, float]) -> dict[str, float]:
    reference = float(mean_norms["reconstruction"])
    if not math.isfinite(reference) or reference <= 0.0:
        raise RuntimeError("reconstruction mean gradient norm is not calibratable")
    weights = {"reconstruction": 1.0}
    for name in OBJECTIVES[1:]:
        norm = float(mean_norms[name])
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"{name} mean gradient norm is not calibratable")
        weights[name] = reference / norm
    return weights


def run_owned_vocoder_loss_v2_weight_calibration_audit(
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

    probes: list[dict[str, Any]] = []
    norm_values: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
    all_gradients_valid = True

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
                all_gradients_valid = all_gradients_valid and valid
                norms = {name: _norm(gradients[name]) for name in OBJECTIVES}
                for name in OBJECTIVES:
                    norm_values[name].append(norms[name])
                pairwise = {
                    f"{first}_vs_{second}": _cosine(gradients[first], gradients[second])
                    for first, second in PAIR_KEYS
                }
                probes.append(
                    {
                        "split": split,
                        "utterance_id": segment.utterance_id,
                        "start_frame": segment.start_frame,
                        "perturbation": perturbation_name,
                        "losses": losses,
                        "gradient_norms": norms,
                        "pairwise_gradient_cosines": pairwise,
                        "diagnostics": diagnostics,
                        "gradients": gradients,
                    }
                )

    mean_raw_norms = {name: _mean(norm_values[name]) for name in OBJECTIVES}
    derived_weights = _derive_equalized_weights(mean_raw_norms)

    share_values: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
    alignment_values: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
    descent_dot_values: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
    pair_values: dict[str, list[float]] = {
        f"{first}_vs_{second}": [] for first, second in PAIR_KEYS
    }
    public_probes: list[dict[str, Any]] = []
    combined_valid = True

    for probe in probes:
        gradients = probe["gradients"]
        weighted = {
            name: gradients[name] * float(derived_weights[name])
            for name in OBJECTIVES
        }
        combined = sum(weighted.values())
        combined_valid = combined_valid and _finite_nonzero(combined)
        weighted_norms = {name: _norm(weighted[name]) for name in OBJECTIVES}
        norm_sum = max(sum(weighted_norms.values()), 1e-12)
        shares = {name: weighted_norms[name] / norm_sum for name in OBJECTIVES}
        alignments = {name: _cosine(gradients[name], combined) for name in OBJECTIVES}
        descent_dots = {
            name: float(torch.sum(gradients[name] * combined)) for name in OBJECTIVES
        }
        for name in OBJECTIVES:
            share_values[name].append(shares[name])
            alignment_values[name].append(alignments[name])
            descent_dot_values[name].append(descent_dots[name])
        for key, value in probe["pairwise_gradient_cosines"].items():
            pair_values[key].append(float(value))

        public_probes.append(
            {
                "split": probe["split"],
                "utterance_id": probe["utterance_id"],
                "start_frame": probe["start_frame"],
                "perturbation": probe["perturbation"],
                "losses": {k: round(float(v), 10) for k, v in probe["losses"].items()},
                "gradient_norms": {
                    k: round(float(v), 10) for k, v in probe["gradient_norms"].items()
                },
                "pairwise_gradient_cosines": {
                    k: round(float(v), 6)
                    for k, v in probe["pairwise_gradient_cosines"].items()
                },
                "derived_weighted_gradient_norm_shares": {
                    k: round(float(v), 6) for k, v in shares.items()
                },
                "combined_gradient_alignment_cosines": {
                    k: round(float(v), 6) for k, v in alignments.items()
                },
                "first_order_descent_dots": {
                    k: round(float(v), 10) for k, v in descent_dots.items()
                },
                "diagnostics": {
                    k: round(float(v), 8) for k, v in probe["diagnostics"].items()
                },
            }
        )

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    status_pass = (
        bool(probes)
        and all_gradients_valid
        and combined_valid
        and checkpoints_unchanged
        and all(math.isfinite(value) and value > 0.0 for value in derived_weights.values())
    )

    summary = {
        "mean_raw_gradient_norms": {
            name: round(mean_raw_norms[name], 10) for name in OBJECTIVES
        },
        "derived_gradient_equalization_weights": {
            name: round(float(derived_weights[name]), 10) for name in OBJECTIVES
        },
        "mean_derived_weighted_gradient_norm_shares": {
            name: round(_mean(share_values[name]), 6) for name in OBJECTIVES
        },
        "minimum_derived_weighted_gradient_norm_shares": {
            name: round(min(share_values[name], default=0.0), 6) for name in OBJECTIVES
        },
        "maximum_derived_weighted_gradient_norm_shares": {
            name: round(max(share_values[name], default=0.0), 6) for name in OBJECTIVES
        },
        "mean_pairwise_gradient_cosines": {
            key: round(_mean(values), 6) for key, values in pair_values.items()
        },
        "minimum_pairwise_gradient_cosines": {
            key: round(min(values, default=0.0), 6) for key, values in pair_values.items()
        },
        "mean_combined_gradient_alignment_cosines": {
            name: round(_mean(alignment_values[name]), 6) for name in OBJECTIVES
        },
        "minimum_combined_gradient_alignment_cosines": {
            name: round(min(alignment_values[name], default=0.0), 6) for name in OBJECTIVES
        },
        "minimum_first_order_descent_dots": {
            name: round(min(descent_dot_values[name], default=0.0), 10)
            for name in OBJECTIVES
        },
        "maximum_derived_weighted_gradient_norm_share": round(
            max(
                (max(values, default=0.0) for values in share_values.values()),
                default=0.0,
            ),
            6,
        ),
    }

    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    report_path = output_dir / "loss_v2_weight_calibration_audit.json"
    report: dict[str, object] = {
        "status": "pass" if status_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "objective_names": list(OBJECTIVES),
        "calibration_rule": (
            "reconstruction_weight=1; other_weight="
            "mean_reconstruction_gradient_norm/mean_objective_gradient_norm"
        ),
        "summary": summary,
        "probes": public_probes,
        "all_objective_gradients_finite_nonzero": all_gradients_valid,
        "combined_gradient_finite_nonzero": combined_valid,
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
            "review_four_objective_gradient_calibration_before_freezing_weight_contract"
            if status_pass
            else "fix_four_objective_gradient_calibration_before_model_work"
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
            run_owned_vocoder_loss_v2_weight_calibration_audit(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
