"""Regenerate LYKENOX speech durations with word-boundary-aware interior blank timing.

This command never trains the aligner. It reuses the validated ``best.pt`` checkpoint,
generates the versioned ``alignment-v3`` cache, and immediately re-runs the duration
outlier gate. The generator remains resumable and bounded for short local executors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.training.speech_duration_cache import generate_duration_cache
from lykenox_voice_engine.training.speech_duration_outlier_review import review_duration_outliers


DEFAULT_TIME_BUDGET_SECONDS = 85.0


def regenerate_interior_safe_durations(
    root: Path,
    *,
    threshold_frames: int = 100,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    root = Path(root).resolve()
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
    if not checkpoint.exists():
        raise FileNotFoundError(f"Validated LYKENOX best aligner not found: {checkpoint}")

    durations = generate_duration_cache(
        root,
        checkpoint,
        nonpause_warn_frames=threshold_frames,
        time_budget_seconds=time_budget_seconds,
        resume=True,
    )
    duration_root = Path(str(durations["duration_cache_root"]))

    if durations["status"] == "incomplete":
        result = {
            "status": "incomplete",
            "duration_cache_version": durations.get("duration_cache_version"),
            "duration_cache_root": str(duration_root),
            "reused_records": durations.get("reused_records"),
            "new_records_generated": durations.get("new_records_generated"),
            "pending_item_count": durations.get("pending_item_count"),
            "elapsed_seconds": durations.get("elapsed_seconds"),
            "next_gate": "rerun_same_command_to_resume",
        }
        (duration_root / "interior_fix_regeneration_report.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    if durations["status"] != "pass":
        return {
            "status": "duration_generation_failed",
            "duration_cache_version": durations.get("duration_cache_version"),
            "duration_cache_root": str(duration_root),
            "failures": durations.get("failures", []),
            "next_gate": "review_failed_alignments",
        }

    review = review_duration_outliers(
        root,
        duration_root=duration_root,
        threshold_frames=threshold_frames,
    )
    status = "pass" if review["status"] == "pass" else "review_required"
    next_gate = "aligned_acoustic_smoke" if status == "pass" else str(review["next_gate"])

    result = {
        "status": status,
        "checkpoint_epoch": durations.get("checkpoint_epoch"),
        "duration_cache_version": durations.get("duration_cache_version"),
        "boundary_blank_policy": durations.get("boundary_blank_policy"),
        "interior_blank_policy": durations.get("interior_blank_policy"),
        "duration_cache_root": str(duration_root),
        "elapsed_seconds": durations.get("elapsed_seconds"),
        "reused_records": durations.get("reused_records"),
        "new_records_generated": durations.get("new_records_generated"),
        "train_generated": durations["splits"]["train"]["generated"],
        "train_items": durations["splits"]["train"]["items"],
        "val_generated": durations["splits"]["val"]["generated"],
        "val_items": durations["splits"]["val"]["items"],
        "nonpause_duration_frames": durations.get("nonpause_duration_frames"),
        "word_boundary_blank_frames": durations.get("word_boundary_blank_frames"),
        "pause_blank_frames": durations.get("pause_blank_frames"),
        "neighbor_split_blank_frames": durations.get("neighbor_split_blank_frames"),
        "outlier_token_count": review.get("outlier_token_count"),
        "outlier_utterance_count": review.get("outlier_utterance_count"),
        "boundary_outlier_token_count": review.get("boundary_outlier_token_count"),
        "interior_outlier_token_count": review.get("interior_outlier_token_count"),
        "diagnosis": review.get("diagnosis"),
        "review_report": review.get("report_path"),
        "next_gate": next_gate,
    }
    report_path = duration_root / "interior_fix_regeneration_report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["report_path"] = str(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--threshold-frames", type=int, default=100)
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    print(
        json.dumps(
            regenerate_interior_safe_durations(
                args.root,
                threshold_frames=args.threshold_frames,
                time_budget_seconds=args.time_budget_seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
