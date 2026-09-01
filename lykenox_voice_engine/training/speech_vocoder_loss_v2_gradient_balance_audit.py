"""Read-only gradient-balance audit for the owned LYKENOX vocoder Loss V2.

This audit intentionally has no vocoder architecture.  It treats a waveform candidate as
the differentiable variable and measures how the already validated owned objectives act on
controlled errors around real held-out/train speech segments.

The goal is diagnostic, not acceptance by metric:
- reconstruction V2 uses only valid crop context;
- envelope V2 is aligned to the owned cached conditioning mel;
- target-relative spectral balance is measured as the third historical objective;
- gradients are obtained with ``torch.autograd.grad`` with respect to the waveform only;
- no optimizer, model parameters, checkpoint write, duration change, or post-hoc output
  processing is involved.

The synthetic perturbations below are audit probes only.  They are never an inference or
training-time audio-processing path.
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
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    collect_owned_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
    OWNED_VOCODER_LOSS_V2_VERSION,
    valid_context_multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    VOCODER_SOURCE_BALANCE_VERSION,
    target_relative_spectral_balance_loss,
)


AUDIT_VERSION = "owned-vocoder-loss-v2-gradient-balance-audit-v1"
DEFAULT_SEGMENT_MEL_FRAMES = 64
DEFAULT_ITEMS_PER_SPLIT = 3
SPLITS = ("train", "val")
OUTPUT_DIR_NAME = "owned_vocoder_loss_v2_gradient_balance_audit_v1"

# These are the historical v4.2 objective weights.  They are measured here only as a
# diagnostic reference; this audit does not authorize retaining them for a future model.
REFERENCE_OBJECTIVE_WEIGHTS = {
    "reconstruction": 1.0,
    "envelope": 0.50,
    "spectral_balance": 0.25,
}


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


def _spectral_gain_candidate(
    target: torch.Tensor,
    *,
    sample_rate: int,
    low_band_gain: float = 1.0,
    mid_band_gain: float = 1.0,
    high_band_gain: float = 1.0,
) -> torch.Tensor:
    """Create a deterministic diagnostic spectral-color perturbation."""

    if target.ndim != 2 or int(target.shape[0]) != 1:
        raise ValueError("target must have shape [1, samples]")
    spectrum = torch.fft.rfft(target, dim=1)
    frequencies = torch.fft.rfftfreq(
        int(target.shape[1]),
        d=1.0 / float(sample_rate),
        device=target.device,
        dtype=target.dtype,
    )
    gain = torch.ones_like(frequencies)
    gain[(frequencies >= 80.0) & (frequencies < 300.0)] = float(low_band_gain)
    gain[(frequencies >= 300.0) & (frequencies < 3000.0)] = float(mid_band_gain)
    gain[frequencies >= 3000.0] = float(high_band_gain)
    return torch.fft.irfft(
        spectrum * gain.unsqueeze(0),
        n=int(target.shape[1]),
        dim=1,
    ).to(target.dtype)


def _diagnostic_candidates(
    target: torch.Tensor,
    *,
    sample_rate: int,
) -> dict[str, torch.Tensor]:
    """Return controlled non-product perturbations representing known failure modes."""

    # Approximate the measured v4.2 coloration direction: slightly too much low-band
    # authority and reduced formant/presence energy above 300 Hz.
    colored = _spectral_gain_candidate(
        target,
        sample_rate=sample_rate,
        low_band_gain=1.035,
        mid_band_gain=0.89,
        high_band_gain=0.91,
    )
    low_excess = _spectral_gain_candidate(
        target,
        sample_rate=sample_rate,
        low_band_gain=1.12,
        mid_band_gain=1.0,
        high_band_gain=1.0,
    )
    shifted = torch.roll(target, shifts=1, dims=1)
    shifted[:, 0] = target[:, 0]
    phase_smear = 0.88 * target + 0.12 * shifted
    return {
        "v4_2_like_color": colored.detach(),
        "low_band_excess": low_excess.detach(),
        "one_sample_phase_smear": phase_smear.detach(),
    }


def _gradient(loss: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        loss,
        candidate,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    return gradient.detach()


def _norm(values: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(values))


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)
    denominator = (first_norm * second_norm).clamp_min(1e-12)
    return float(torch.sum(first * second) / denominator)


def _finite_nonzero(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all()) and _norm(values) > 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def _min(rows: list[dict[str, Any]], key: str) -> float:
    return min((float(row[key]) for row in rows), default=0.0)


def _max(rows: list[dict[str, Any]], key: str) -> float:
    return max((float(row[key]) for row in rows), default=0.0)


def _objective_probe(
    candidate_values: torch.Tensor,
    target: torch.Tensor,
    conditioning: torch.Tensor,
    envelope_objective: ConditioningAlignedLogMelEnvelopeLossV2,
    *,
    sample_rate: int,
) -> dict[str, Any]:
    candidate = candidate_values.detach().clone().requires_grad_(True)
    reconstruction = valid_context_multi_resolution_reconstruction_loss(
        candidate,
        target,
    )
    envelope = envelope_objective(candidate, conditioning)
    spectral_balance = target_relative_spectral_balance_loss(
        candidate,
        target,
        sample_rate=sample_rate,
    )

    losses = {
        "reconstruction": reconstruction.total,
        "envelope": envelope.total,
        "spectral_balance": spectral_balance.loss,
    }
    gradients = {
        name: _gradient(loss, candidate)
        for name, loss in losses.items()
    }
    gradients_finite_nonzero = all(_finite_nonzero(values) for values in gradients.values())

    weighted_gradients = {
        name: values * float(REFERENCE_OBJECTIVE_WEIGHTS[name])
        for name, values in gradients.items()
    }
    combined = sum(weighted_gradients.values())
    combined_finite_nonzero = _finite_nonzero(combined)

    weighted_norms = {name: _norm(values) for name, values in weighted_gradients.items()}
    weighted_norm_sum = max(sum(weighted_norms.values()), 1e-12)
    weighted_shares = {
        name: value / weighted_norm_sum
        for name, value in weighted_norms.items()
    }

    pairwise = {
        "reconstruction_vs_envelope": _cosine(
            gradients["reconstruction"], gradients["envelope"]
        ),
        "reconstruction_vs_spectral_balance": _cosine(
            gradients["reconstruction"], gradients["spectral_balance"]
        ),
        "envelope_vs_spectral_balance": _cosine(
            gradients["envelope"], gradients["spectral_balance"]
        ),
    }
    combined_alignment = {
        name: _cosine(values, combined)
        for name, values in gradients.items()
    }

    return {
        "losses": {
            "reconstruction": float(losses["reconstruction"].detach()),
            "reconstruction_waveform_l1": float(reconstruction.waveform_l1.detach()),
            "reconstruction_spectral_convergence": float(
                reconstruction.spectral_convergence.detach()
            ),
            "reconstruction_log_magnitude": float(reconstruction.log_magnitude.detach()),
            "envelope": float(losses["envelope"].detach()),
            "envelope_log_mel_l1": float(envelope.log_mel_l1.detach()),
            "envelope_spectral_slope_l1": float(envelope.spectral_slope_l1.detach()),
            "envelope_temporal_delta_l1": float(envelope.temporal_delta_l1.detach()),
            "spectral_balance": float(losses["spectral_balance"].detach()),
        },
        "gradient_norms": {
            name: _norm(values) for name, values in gradients.items()
        },
        "reference_weighted_gradient_norms": weighted_norms,
        "reference_weighted_gradient_norm_shares": weighted_shares,
        "pairwise_gradient_cosines": pairwise,
        "combined_gradient_alignment_cosines": combined_alignment,
        "combined_gradient_norm": _norm(combined),
        "gradients_finite_nonzero": gradients_finite_nonzero,
        "combined_gradient_finite_nonzero": combined_finite_nonzero,
    }


def run_owned_vocoder_loss_v2_gradient_balance_audit(
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

    items: list[dict[str, object]] = []
    flat_rows: list[dict[str, Any]] = []
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
            perturbations = _diagnostic_candidates(
                target,
                sample_rate=config.sample_rate,
            )
            probes: dict[str, object] = {}
            for perturbation_name, candidate in perturbations.items():
                probe = _objective_probe(
                    candidate,
                    target,
                    conditioning,
                    envelope_objective,
                    sample_rate=config.sample_rate,
                )
                valid = bool(probe["gradients_finite_nonzero"]) and bool(
                    probe["combined_gradient_finite_nonzero"]
                )
                all_gradients_valid = all_gradients_valid and valid
                probes[perturbation_name] = probe

                pairwise = probe["pairwise_gradient_cosines"]
                combined_alignment = probe["combined_gradient_alignment_cosines"]
                shares = probe["reference_weighted_gradient_norm_shares"]
                norms = probe["gradient_norms"]
                flat_rows.append(
                    {
                        "reconstruction_norm": norms["reconstruction"],
                        "envelope_norm": norms["envelope"],
                        "spectral_balance_norm": norms["spectral_balance"],
                        "reconstruction_share": shares["reconstruction"],
                        "envelope_share": shares["envelope"],
                        "spectral_balance_share": shares["spectral_balance"],
                        "reconstruction_vs_envelope": pairwise[
                            "reconstruction_vs_envelope"
                        ],
                        "reconstruction_vs_spectral_balance": pairwise[
                            "reconstruction_vs_spectral_balance"
                        ],
                        "envelope_vs_spectral_balance": pairwise[
                            "envelope_vs_spectral_balance"
                        ],
                        "combined_to_reconstruction": combined_alignment[
                            "reconstruction"
                        ],
                        "combined_to_envelope": combined_alignment["envelope"],
                        "combined_to_spectral_balance": combined_alignment[
                            "spectral_balance"
                        ],
                    }
                )

            items.append(
                {
                    "split": split,
                    "utterance_id": segment.utterance_id,
                    "start_frame": segment.start_frame,
                    "probes": probes,
                }
            )

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    status_pass = all_gradients_valid and checkpoints_unchanged and bool(flat_rows)

    summary = {
        "mean_gradient_norms": {
            "reconstruction": round(_mean(flat_rows, "reconstruction_norm"), 10),
            "envelope": round(_mean(flat_rows, "envelope_norm"), 10),
            "spectral_balance": round(_mean(flat_rows, "spectral_balance_norm"), 10),
        },
        "mean_reference_weighted_gradient_norm_shares": {
            "reconstruction": round(_mean(flat_rows, "reconstruction_share"), 6),
            "envelope": round(_mean(flat_rows, "envelope_share"), 6),
            "spectral_balance": round(_mean(flat_rows, "spectral_balance_share"), 6),
        },
        "mean_pairwise_gradient_cosines": {
            "reconstruction_vs_envelope": round(
                _mean(flat_rows, "reconstruction_vs_envelope"), 6
            ),
            "reconstruction_vs_spectral_balance": round(
                _mean(flat_rows, "reconstruction_vs_spectral_balance"), 6
            ),
            "envelope_vs_spectral_balance": round(
                _mean(flat_rows, "envelope_vs_spectral_balance"), 6
            ),
        },
        "minimum_pairwise_gradient_cosine": round(
            min(
                _min(flat_rows, "reconstruction_vs_envelope"),
                _min(flat_rows, "reconstruction_vs_spectral_balance"),
                _min(flat_rows, "envelope_vs_spectral_balance"),
            ),
            6,
        ),
        "mean_combined_gradient_alignment_cosines": {
            "reconstruction": round(_mean(flat_rows, "combined_to_reconstruction"), 6),
            "envelope": round(_mean(flat_rows, "combined_to_envelope"), 6),
            "spectral_balance": round(
                _mean(flat_rows, "combined_to_spectral_balance"), 6
            ),
        },
        "minimum_combined_gradient_alignment_cosine": round(
            min(
                _min(flat_rows, "combined_to_reconstruction"),
                _min(flat_rows, "combined_to_envelope"),
                _min(flat_rows, "combined_to_spectral_balance"),
            ),
            6,
        ),
        "maximum_reference_weighted_gradient_norm_share": round(
            max(
                _max(flat_rows, "reconstruction_share"),
                _max(flat_rows, "envelope_share"),
                _max(flat_rows, "spectral_balance_share"),
            ),
            6,
        ),
    }

    output_dir = (
        root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    )
    report_path = output_dir / "loss_v2_gradient_balance_audit.json"
    report: dict[str, object] = {
        "status": "pass" if status_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "owned_segment_contract": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "owned_loss_contract": OWNED_VOCODER_LOSS_V2_VERSION,
        "spectral_balance_contract": VOCODER_SOURCE_BALANCE_VERSION,
        "segment_mel_frames": segment_mel_frames,
        "items_per_split": items_per_split,
        "diagnostic_perturbations": [
            "v4_2_like_color",
            "low_band_excess",
            "one_sample_phase_smear",
        ],
        "reference_objective_weights_are_diagnostic_only": True,
        "reference_objective_weights": dict(REFERENCE_OBJECTIVE_WEIGHTS),
        "all_objective_gradients_finite_nonzero": all_gradients_valid,
        "summary": summary,
        "items": items,
        "checkpoints_unchanged": checkpoints_unchanged,
        "training_started": False,
        "optimizer_created": False,
        "model_instantiated": False,
        "persistent_training_authorized": False,
        "new_vocoder_architecture_authorized": False,
        "loss_weight_contract_authorized": False,
        "third_party_model_used": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "report_path": str(report_path),
        "next_gate": (
            "review_owned_vocoder_loss_v2_gradient_balance_before_weight_or_architecture_contract"
            if status_pass
            else "fix_owned_vocoder_loss_v2_gradient_semantics_before_model_work"
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
            run_owned_vocoder_loss_v2_gradient_balance_audit(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
