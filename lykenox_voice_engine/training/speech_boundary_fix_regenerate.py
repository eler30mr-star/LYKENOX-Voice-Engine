"""Regenerate boundary-safe LYKENOX duration caches from the validated best aligner.

This command does not train or modify the aligner checkpoint. It reuses ``best.pt``,
generates ``alignment-v2`` durations with leading/trailing CTC blanks mapped to BOS/EOS,
then runs the fast outlier review and prints only a compact gate summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.training.speech_duration_cache import generate_duration_cache
from lykenox_voice_engine.training.speech_duration_outlier_review import review_duration_outliers


def regenerate_boundary_safe_durations(
    root: Path,
    *,
    threshold_frames: int = 100,
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

    duration_report = generate_duration_cache(
        root,
        checkpoint,
        nonpause_warn_frames=threshold_frames,
    )
    duration_root = Path(str(duration_report["duration_cache_root"]))
    if duration_report["status"] != "pass":
        return {
            "status": "duration_generation_failed",
            "checkpoint": str(checkpoint),
            "duration_cache_root": str(duration_root),
            "failures": duration_report.get("failures", []),
            "next_gate": "review_failed_alignments",
        }

    review = review_duration_outliers(
        root,
        duration_root=duration_root,
        threshold_frames=threshold_frames,
    )
    if review["status"] == "pass":
        status = "pass"
        next_gate = "aligned_acoustic_smoke"
    else:
        status = "review_required"
        next_gate = str(review["next_gate"])

    report = {
        "status": status,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": duration_report.get("checkpoint_epoch"),
        "duration_cache_version": duration_report.get("duration_cache_version"),
        "boundary_blank_policy": duration_report.get("boundary_blank_policy"),
        "duration_cache_root": str(duration_root),
        "train_generated": duration_report["splits"]["train"]["generated"],
        "train_items": duration_report["splits"]["train"]["items"],
        "val_generated": duration_report["splits"]["val"]["generated"],
        "val_items": duration_report["splits"]["val"]["items"],
        "content_duration_frames": duration_report["content_duration_frames"],
        "nonpause_duration_frames": duration_report["nonpause_duration_frames"],
        "leading_boundary_frames": duration_report["leading_boundary_frames"],
        "trailing_boundary_frames": duration_report["trailing_boundary_frames"],
        "outlier_token_count": review["outlier_token_count"],
        "outlier_utterance_count": review["outlier_utterance_count"],
        "boundary_outlier_token_count": review["boundary_outlier_token_count"],
        "interior_outlier_token_count": review["interior_outlier_token_count"],
        "boundary_fraction": review["boundary_fraction"],
        "diagnosis": review["diagnosis"],
        "review_report": review["report_path"],
        "next_gate": next_gate,
    }
    report_path = duration_root / "boundary_fix_regeneration_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--threshold-frames", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            regenerate_boundary_safe_durations(
                args.root,
                threshold_frames=args.threshold_frames,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
