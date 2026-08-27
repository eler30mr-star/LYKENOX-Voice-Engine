"""Targeted forensic review of residual LYKENOX speech duration outliers.

This command performs no training and does not regenerate the corpus. It reruns
only the few utterances already flagged by alignment-v2 and decomposes each long
phoneme duration into direct CTC target occupancy versus interior blank frames
assigned by the duration policy.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from lykenox_voice_engine.core.ctc_alignment import (
    ctc_targets,
    forced_alignment_durations,
)
from lykenox_voice_engine.core.ctc_alignment_diagnostics import (
    ctc_frame_ownership_breakdown,
)
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.alignment_artifact import load_aligner_checkpoint
from lykenox_voice_engine.training.speech_aligner_train import _dataset
from lykenox_voice_engine.training.speech_duration_outlier_review import (
    PAUSE_TOKENS,
    _boundary_role,
    _latest_duration_root,
    _record_paths,
    review_duration_outliers,
)


def classify_interior_mechanism(
    direct_target_frames: int,
    allocated_blank_frames: int,
) -> str:
    """Classify the mechanical source of one long interior duration."""

    total = direct_target_frames + allocated_blank_frames
    if total <= 0:
        return "invalid_zero_duration"
    blank_fraction = allocated_blank_frames / total
    direct_fraction = direct_target_frames / total
    if blank_fraction >= 0.60:
        return "interior_blank_allocation_dominant"
    if direct_fraction >= 0.60:
        return "direct_ctc_occupancy_dominant"
    return "mixed_ctc_occupancy"


def classify_interior_set(mechanisms: list[str]) -> tuple[str, str]:
    """Return aggregate diagnosis and the next safe engineering gate."""

    if not mechanisms:
        return "no_interior_outliers", "review_residual_boundary_outliers_or_aligned_smoke"
    counts = Counter(mechanisms)
    total = len(mechanisms)
    blank_dominant = counts["interior_blank_allocation_dominant"]
    direct_dominant = counts["direct_ctc_occupancy_dominant"]
    if blank_dominant / total >= 0.60:
        return "interior_blank_allocation_dominant", "fix_interior_blank_assignment"
    if direct_dominant / total >= 0.60:
        return "direct_ctc_occupancy_dominant", "inspect_transcript_or_acoustic_mismatch"
    return "mixed_interior_alignment_mechanisms", "review_mixed_interior_outliers"


def _mean_log_mel(mel: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any().item()):
        return None
    return float(mel[mask].mean().detach().cpu())


def review_interior_alignments(
    root: Path,
    *,
    duration_root: Path | None = None,
    threshold_frames: int = 100,
) -> dict[str, object]:
    if threshold_frames < 1:
        raise ValueError("threshold_frames must be >= 1")

    root = Path(root).resolve()
    duration_root = (
        Path(duration_root).resolve()
        if duration_root is not None
        else _latest_duration_root(root)
    )
    quick_review = review_duration_outliers(
        root,
        duration_root=duration_root,
        threshold_frames=threshold_frames,
    )
    if str(quick_review.get("cache_version")) != "alignment-v2":
        raise RuntimeError("Interior forensic review requires an alignment-v2 cache")

    frontend = SpanishTextFrontend()
    checkpoint = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "speech_aligner"
        / frontend.version
        / "best.pt"
    )
    model, checkpoint_payload = load_aligner_checkpoint(checkpoint)
    model.eval()
    speech_config = LykenoxSpeechConfig()
    frame_ms = speech_config.hop_length / speech_config.sample_rate * 1000.0

    cached_records: dict[tuple[str, str], dict[str, object]] = {}
    flagged: list[dict[str, object]] = []
    for split, path in _record_paths(duration_root):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid duration record: {path}")
        utterance_id = str(record.get("utterance_id"))
        cached_records[(split, utterance_id)] = record
        content = list(record.get("content", []))
        for content_index, row in enumerate(content):
            token = str(row.get("token"))
            duration = int(row.get("duration_frames", 0))
            if token in PAUSE_TOKENS or duration <= threshold_frames:
                continue
            flagged.append(
                {
                    "split": split,
                    "utterance_id": utterance_id,
                    "content_index": content_index,
                    "token": token,
                    "duration_frames": duration,
                    "boundary_role": _boundary_role(content, content_index),
                }
            )

    if not flagged:
        report = {
            "status": "pass_no_outliers",
            "duration_root": str(duration_root),
            "outliers_analyzed": 0,
            "interior_outliers_analyzed": 0,
            "diagnosis": "no_interior_outliers",
            "next_gate": "aligned_acoustic_smoke",
        }
        report_path = duration_root / "interior_alignment_review.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    datasets = {
        split: _dataset(root, split, speech_config)
        for split in {str(row["split"]) for row in flagged}
    }
    index_lookup: dict[str, dict[str, int]] = {}
    for split, dataset in datasets.items():
        index_lookup[split] = {
            str(row.utterance_id): index
            for index, row in enumerate(dataset.rows)
        }

    by_utterance: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in flagged:
        key = (str(row["split"]), str(row["utterance_id"]))
        by_utterance.setdefault(key, []).append(row)

    token_stats = dict(quick_review.get("token_stats", {}))
    details: list[dict[str, object]] = []
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    with torch.no_grad():
        for (split, utterance_id), rows in sorted(by_utterance.items()):
            dataset = datasets[split]
            try:
                item_index = index_lookup[split][utterance_id]
            except KeyError as error:
                raise RuntimeError(
                    f"Flagged utterance {utterance_id} not found in {split} dataset"
                ) from error
            item = dataset[item_index]
            mel = item["mel"]
            token_ids = item["token_ids"]
            targets, positions = ctc_targets(token_ids)
            logits = model(mel.unsqueeze(0)).squeeze(0)
            log_probs = F.log_softmax(logits, dim=-1)
            alignment = forced_alignment_durations(
                log_probs,
                targets,
                model.config.blank_id,
                mel_frames=int(mel.shape[0]),
                frame_stride=model.config.frame_stride,
            )
            ownership = ctc_frame_ownership_breakdown(
                alignment,
                target_steps=int(targets.numel()),
                frame_stride=model.config.frame_stride,
            )

            cached_record = cached_records[(split, utterance_id)]
            content = list(cached_record.get("content", []))
            for row in rows:
                target_index = int(row["content_index"])
                if target_index >= int(targets.numel()):
                    raise RuntimeError(
                        f"Content index {target_index} exceeds targets for {utterance_id}"
                    )
                cached_duration = int(row["duration_frames"])
                recomputed_duration = int(alignment.target_durations[target_index].item())
                if cached_duration != recomputed_duration:
                    raise RuntimeError(
                        f"Stale or inconsistent duration cache for {utterance_id}: "
                        f"cached={cached_duration}, recomputed={recomputed_duration}"
                    )

                direct_frames = int(ownership.direct_target_frames[target_index].item())
                blank_frames = int(ownership.allocated_blank_frames[target_index].item())
                blank_fraction = blank_frames / max(1, recomputed_duration)
                direct_fraction = direct_frames / max(1, recomputed_duration)
                owner_mask = ownership.frame_owners == target_index
                direct_mask = owner_mask & ownership.frame_is_direct_target
                blank_mask = owner_mask & (~ownership.frame_is_direct_target)
                direct_log_mel = _mean_log_mel(mel, direct_mask)
                blank_log_mel = _mean_log_mel(mel, blank_mask)
                blank_minus_direct = (
                    blank_log_mel - direct_log_mel
                    if blank_log_mel is not None and direct_log_mel is not None
                    else None
                )

                token = str(row["token"])
                stats = dict(token_stats.get(token, {}))
                median = float(stats.get("median_frames", 0.0) or 0.0)
                content_row = dict(content[target_index])
                previous_token = (
                    str(content[target_index - 1].get("token"))
                    if target_index > 0
                    else None
                )
                next_token = (
                    str(content[target_index + 1].get("token"))
                    if target_index + 1 < len(content)
                    else None
                )
                mechanism = classify_interior_mechanism(direct_frames, blank_frames)
                details.append(
                    {
                        "split": split,
                        "utterance_id": utterance_id,
                        "text": str(cached_record.get("text", "")),
                        "token": token,
                        "content_index": target_index,
                        "model_token_position": int(positions[target_index]),
                        "boundary_role": str(row["boundary_role"]),
                        "previous_token": previous_token,
                        "next_token": next_token,
                        "duration_frames": recomputed_duration,
                        "duration_ms": round(recomputed_duration * frame_ms, 2),
                        "direct_target_frames": direct_frames,
                        "allocated_blank_frames": blank_frames,
                        "direct_fraction": round(direct_fraction, 4),
                        "blank_fraction": round(blank_fraction, 4),
                        "direct_mean_log_mel": (
                            round(direct_log_mel, 5) if direct_log_mel is not None else None
                        ),
                        "blank_mean_log_mel": (
                            round(blank_log_mel, 5) if blank_log_mel is not None else None
                        ),
                        "blank_minus_direct_mean_log_mel": (
                            round(blank_minus_direct, 5)
                            if blank_minus_direct is not None
                            else None
                        ),
                        "token_median_frames": stats.get("median_frames"),
                        "token_p95_frames": stats.get("p95_frames"),
                        "ratio_to_token_median": (
                            round(recomputed_duration / median, 2) if median > 0 else None
                        ),
                        "alignment_score_per_step": round(
                            float(cached_record.get("alignment_score_per_step", 0.0)), 6
                        ),
                        "mechanism": mechanism,
                        "cached_token_id": content_row.get("token_id"),
                    }
                )

    details.sort(key=lambda value: int(value["duration_frames"]), reverse=True)
    interior = [row for row in details if row["boundary_role"] == "interior"]
    residual_boundary = [row for row in details if row["boundary_role"] != "interior"]
    mechanisms = [str(row["mechanism"]) for row in interior]
    mechanism_counts = Counter(mechanisms)
    diagnosis, next_gate = classify_interior_set(mechanisms)

    report = {
        "status": "review_complete",
        "duration_root": str(duration_root),
        "cache_version": "alignment-v2",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "threshold_frames": threshold_frames,
        "threshold_ms": round(threshold_frames * frame_ms, 2),
        "outliers_analyzed": len(details),
        "interior_outliers_analyzed": len(interior),
        "residual_boundary_outliers_analyzed": len(residual_boundary),
        "interior_mechanism_counts": dict(mechanism_counts),
        "interior_blank_fraction_median": (
            round(statistics.median(float(row["blank_fraction"]) for row in interior), 4)
            if interior
            else None
        ),
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "interior_outliers": interior,
        "residual_boundary_outliers": residual_boundary,
        "interpretation": (
            "A high blank fraction means the long duration is mostly created by the "
            "interior CTC blank-allocation policy, not by direct phoneme occupancy. "
            "A high direct fraction points instead toward the aligner acoustically "
            "holding that phoneme region, transcript mismatch, or genuinely long speech."
        ),
    }
    report_path = duration_root / "interior_alignment_review.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--duration-root", type=Path)
    parser.add_argument("--threshold-frames", type=int, default=100)
    args = parser.parse_args()
    result = review_interior_alignments(
        args.root,
        duration_root=args.duration_root,
        threshold_frames=args.threshold_frames,
    )
    compact = {
        "status": result["status"],
        "cache_version": result.get("cache_version"),
        "checkpoint_epoch": result.get("checkpoint_epoch"),
        "outliers_analyzed": result.get("outliers_analyzed"),
        "interior_outliers_analyzed": result.get("interior_outliers_analyzed"),
        "residual_boundary_outliers_analyzed": result.get("residual_boundary_outliers_analyzed"),
        "interior_mechanism_counts": result.get("interior_mechanism_counts"),
        "interior_blank_fraction_median": result.get("interior_blank_fraction_median"),
        "diagnosis": result.get("diagnosis"),
        "next_gate": result.get("next_gate"),
        "interior_outliers": result.get("interior_outliers"),
        "report_path": result.get("report_path"),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
