"""Controlled LYKENOX aligner training/recovery and duration audit pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.training.speech_aligner_recover import recover_pipeline
from lykenox_voice_engine.training.speech_aligner_train import train_persistent_aligner
from lykenox_voice_engine.training.speech_duration_cache import generate_duration_cache


def _best_checkpoint(root: Path) -> Path:
    frontend = SpanishTextFrontend()
    return (
        Path(root)
        / "models"
        / "lykenox_identity"
        / "training"
        / "speech_aligner"
        / frontend.version
        / "best.pt"
    )


def run_pipeline(
    root: Path,
    *,
    epochs: int = 20,
    patience: int = 4,
    min_delta: float = 0.01,
    max_mel_frames: int = 1800,
    seed: int = 1337,
    nonpause_warn_frames: int = 100,
    recover_existing: bool = True,
) -> dict[str, object]:
    root = Path(root).resolve()

    # A killed terminal/executor can interrupt after best.pt was already written.
    # Never throw that work away: audit it first and only retrain if it fails.
    if recover_existing and _best_checkpoint(root).exists():
        recovered = recover_pipeline(
            root,
            max_mel_frames=max_mel_frames,
            seed=seed,
            nonpause_warn_frames=nonpause_warn_frames,
        )
        if recovered["status"] in {"pass", "duration_review_required"}:
            return {
                "status": recovered["status"],
                "mode": "recovered_existing_checkpoint",
                "training": recovered["recovery"],
                "durations": recovered["durations"],
                "next_gate": recovered["next_gate"],
            }
        if recovered["status"] == "duration_gate_failed":
            return {
                "status": "duration_gate_failed",
                "mode": "recovered_existing_checkpoint",
                "training": recovered["recovery"],
                "durations": recovered["durations"],
                "next_gate": recovered["next_gate"],
            }

    training = train_persistent_aligner(
        root,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        max_mel_frames=max_mel_frames,
        seed=seed,
    )
    if training["status"] != "pass":
        return {
            "status": "training_gate_failed",
            "mode": "trained_new_checkpoint",
            "training": training,
            "durations": None,
            "next_gate": "review_alignment_training",
        }

    durations = generate_duration_cache(
        root,
        Path(str(training["best_checkpoint"])),
        nonpause_warn_frames=nonpause_warn_frames,
    )
    suspicious = int(durations.get("suspicious_utterance_count", 0))
    if durations["status"] != "pass":
        status = "duration_gate_failed"
        next_gate = "review_failed_alignments"
    elif suspicious > 0:
        status = "duration_review_required"
        next_gate = "review_duration_outliers"
    else:
        status = "pass"
        next_gate = "aligned_acoustic_smoke"

    result = {
        "status": status,
        "mode": "trained_new_checkpoint",
        "training": training,
        "durations": durations,
        "next_gate": next_gate,
    }

    report_dir = Path(str(training["best_checkpoint"])).parent
    (report_dir / "alignment_pipeline_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--max-mel-frames", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--nonpause-warn-frames", type=int, default=100)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore an existing best.pt and start a new aligner run.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(
                args.root,
                epochs=args.epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                max_mel_frames=args.max_mel_frames,
                seed=args.seed,
                nonpause_warn_frames=args.nonpause_warn_frames,
                recover_existing=not args.force_retrain,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
