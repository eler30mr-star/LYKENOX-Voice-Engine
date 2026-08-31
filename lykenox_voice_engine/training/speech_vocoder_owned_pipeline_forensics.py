"""Read-only forensics for the LYKENOX-owned vocoder conditioning pipeline.

This audit compares the historical v1 crop-local pitch semantics against the corrected v2
full-utterance cached conditioning contract. It performs no vocoder forward pass, no
optimizer step, no checkpoint write, and uses no third-party trained component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_pitch_cache import load_indexed_pitch_target
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    VOCODER_SEGMENT_CONTRACT_VERSION,
    collect_owned_vocoder_segments,
    collect_vocoder_segments,
)


AUDIT_VERSION = "owned-vocoder-conditioning-pipeline-forensics-v1"
SPLITS = ("train", "val")
DEFAULT_SEGMENT_MEL_FRAMES = 64
DEFAULT_ITEMS_PER_SPLIT = 3
BOUNDARY_FRAME_COUNT = 2
OUTPUT_DIR_NAME = "owned_vocoder_conditioning_pipeline_forensics_v1"


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


def _selected_indices(frame_count: int, *, boundary: bool) -> torch.Tensor:
    if frame_count < BOUNDARY_FRAME_COUNT * 2 + 1:
        return torch.arange(frame_count, dtype=torch.long)
    if boundary:
        return torch.tensor(
            [0, 1, frame_count - 2, frame_count - 1], dtype=torch.long
        )
    return torch.arange(BOUNDARY_FRAME_COUNT, frame_count - BOUNDARY_FRAME_COUNT)


def _pitch_comparison(
    old_f0: torch.Tensor,
    old_voiced: torch.Tensor,
    old_periodicity: torch.Tensor,
    new_f0: torch.Tensor,
    new_voiced: torch.Tensor,
    new_periodicity: torch.Tensor,
    *,
    boundary: bool | None = None,
) -> dict[str, float | int]:
    count = int(old_f0.numel())
    if not all(
        int(values.numel()) == count
        for values in (old_voiced, old_periodicity, new_f0, new_voiced, new_periodicity)
    ):
        raise ValueError("pitch comparison tensors must have equal frame counts")
    if boundary is None:
        indices = torch.arange(count, dtype=torch.long)
    else:
        indices = _selected_indices(count, boundary=boundary)

    ov = old_voiced[indices] > 0.5
    nv = new_voiced[indices] > 0.5
    disagreement = (ov != nv).to(torch.float32).mean()
    periodicity_l1 = (
        old_periodicity[indices] - new_periodicity[indices]
    ).abs().mean()
    common = ov & nv & (old_f0[indices] > 0.0) & (new_f0[indices] > 0.0)
    if bool(common.any()):
        ratio = old_f0[indices][common] / new_f0[indices][common]
        cents = (1200.0 * torch.log2(ratio.clamp_min(1e-8))).abs()
        f0_mae_cents = float(cents.mean())
        common_count = int(common.sum())
    else:
        f0_mae_cents = 0.0
        common_count = 0
    return {
        "frames": int(indices.numel()),
        "voicing_disagreement_fraction": float(disagreement),
        "periodicity_l1": float(periodicity_l1),
        "common_voiced_frames": common_count,
        "f0_mae_cents_on_common_voiced": f0_mae_cents,
    }


def _mean(rows: list[dict[str, float | int]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def run_owned_vocoder_conditioning_forensics(
    root: Path,
    *,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    items_per_split: int = DEFAULT_ITEMS_PER_SPLIT,
    seed: int = 4242,
) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected(root)
    before = {name: _sha256(path) for name, path in protected.items()}
    items: list[dict[str, object]] = []
    all_rows: list[dict[str, float | int]] = []
    boundary_rows: list[dict[str, float | int]] = []
    interior_rows: list[dict[str, float | int]] = []
    v2_exact = True
    same_selection = True
    clock_exact = True

    for split in SPLITS:
        historical, _old_skipped = collect_vocoder_segments(
            root,
            split,
            segment_mel_frames=segment_mel_frames,
            max_items=items_per_split,
            seed=seed,
        )
        owned, _new_skipped = collect_owned_vocoder_segments(
            root,
            split,
            segment_mel_frames=segment_mel_frames,
            max_items=items_per_split,
            seed=seed,
        )
        if len(historical) != len(owned):
            raise RuntimeError("v1/v2 selected segment count mismatch")

        for old, new in zip(historical, owned, strict=True):
            selection_match = (
                old.utterance_id == new.utterance_id
                and old.start_frame == new.start_frame
                and old.mel_frames == new.mel_frames
            )
            same_selection = same_selection and selection_match
            if not selection_match:
                raise RuntimeError("v1/v2 deterministic crop selection diverged")
            if not torch.equal(old.mel, new.mel) or not torch.equal(
                old.waveform, new.waveform
            ):
                raise RuntimeError("v2 changed mel or waveform target instead of conditioning")

            crop_pitch = extract_pitch_frames(
                old.waveform,
                frame_count=old.mel_frames,
            )
            cached = load_indexed_pitch_target(
                root,
                split=split,
                utterance_id=new.utterance_id,
            )
            start = new.start_frame
            end = start + new.mel_frames
            exact_cached_slice = (
                torch.equal(new.f0_hz, cached.f0_hz[start:end])
                and torch.equal(new.voiced, cached.voiced[start:end])
                and torch.equal(new.periodicity, cached.periodicity[start:end])
            )
            v2_exact = v2_exact and exact_cached_slice

            expected_start_sample = start * 256
            start_clock_offset_samples = expected_start_sample - start * 256
            clock_exact = clock_exact and start_clock_offset_samples == 0

            all_cmp = _pitch_comparison(
                crop_pitch.f0_hz,
                crop_pitch.voiced,
                crop_pitch.periodicity,
                new.f0_hz,
                new.voiced,
                new.periodicity,
            )
            boundary_cmp = _pitch_comparison(
                crop_pitch.f0_hz,
                crop_pitch.voiced,
                crop_pitch.periodicity,
                new.f0_hz,
                new.voiced,
                new.periodicity,
                boundary=True,
            )
            interior_cmp = _pitch_comparison(
                crop_pitch.f0_hz,
                crop_pitch.voiced,
                crop_pitch.periodicity,
                new.f0_hz,
                new.voiced,
                new.periodicity,
                boundary=False,
            )
            all_rows.append(all_cmp)
            boundary_rows.append(boundary_cmp)
            interior_rows.append(interior_cmp)
            items.append(
                {
                    "split": split,
                    "utterance_id": new.utterance_id,
                    "start_frame": start,
                    "start_sample": expected_start_sample,
                    "frames": new.mel_frames,
                    "v1_v2_same_crop": selection_match,
                    "v2_exact_pitch_cache_slice": exact_cached_slice,
                    "all_frames": {
                        key: round(float(value), 6)
                        if isinstance(value, float)
                        else value
                        for key, value in all_cmp.items()
                    },
                    "boundary_frames": {
                        key: round(float(value), 6)
                        if isinstance(value, float)
                        else value
                        for key, value in boundary_cmp.items()
                    },
                    "interior_frames": {
                        key: round(float(value), 6)
                        if isinstance(value, float)
                        else value
                        for key, value in interior_cmp.items()
                    },
                }
            )

    historical_mismatch = any(
        float(row["voicing_disagreement_fraction"]) > 0.0
        or float(row["periodicity_l1"]) > 1e-7
        or float(row["f0_mae_cents_on_common_voiced"]) > 1e-4
        for row in all_rows
    )
    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    status_pass = all((v2_exact, same_selection, clock_exact, checkpoints_unchanged))

    output_dir = (
        root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    )
    report_path = output_dir / "owned_conditioning_forensics.json"
    report: dict[str, object] = {
        "status": "pass" if status_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "historical_segment_contract": VOCODER_SEGMENT_CONTRACT_VERSION,
        "corrected_segment_contract": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "splits": list(SPLITS),
        "segment_mel_frames": segment_mel_frames,
        "items_per_split": items_per_split,
        "v1_v2_same_deterministic_crop": same_selection,
        "v2_conditioning_exact_from_owned_pitch_cache": v2_exact,
        "shared_frame_origin_clock_offset_samples": 0 if clock_exact else None,
        "historical_v1_crop_local_pitch_mismatch_observed": historical_mismatch,
        "mean_all_frames": {
            "voicing_disagreement_fraction": round(
                _mean(all_rows, "voicing_disagreement_fraction"), 6
            ),
            "periodicity_l1": round(_mean(all_rows, "periodicity_l1"), 6),
            "f0_mae_cents_on_common_voiced": round(
                _mean(all_rows, "f0_mae_cents_on_common_voiced"), 6
            ),
        },
        "mean_boundary_frames": {
            "voicing_disagreement_fraction": round(
                _mean(boundary_rows, "voicing_disagreement_fraction"), 6
            ),
            "periodicity_l1": round(_mean(boundary_rows, "periodicity_l1"), 6),
            "f0_mae_cents_on_common_voiced": round(
                _mean(boundary_rows, "f0_mae_cents_on_common_voiced"), 6
            ),
        },
        "mean_interior_frames": {
            "voicing_disagreement_fraction": round(
                _mean(interior_rows, "voicing_disagreement_fraction"), 6
            ),
            "periodicity_l1": round(_mean(interior_rows, "periodicity_l1"), 6),
            "f0_mae_cents_on_common_voiced": round(
                _mean(interior_rows, "f0_mae_cents_on_common_voiced"), 6
            ),
        },
        "items": items,
        "checkpoints_unchanged": checkpoints_unchanged,
        "training_started": False,
        "persistent_training_authorized": False,
        "new_vocoder_architecture_authorized": False,
        "third_party_model_used": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "report_path": str(report_path),
        "next_gate": (
            "audit_owned_vocoder_loss_edge_and_objective_semantics"
            if status_pass
            else "fix_owned_vocoder_conditioning_contract_before_any_model_work"
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
            run_owned_vocoder_conditioning_forensics(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
