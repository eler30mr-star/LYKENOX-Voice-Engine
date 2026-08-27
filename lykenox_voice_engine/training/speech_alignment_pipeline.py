"""One controlled command for persistent LYKENOX aligner training and duration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.training.speech_aligner_train import train_persistent_aligner
from lykenox_voice_engine.training.speech_duration_cache import generate_duration_cache


def run_pipeline(
    root: Path,
    *,
    epochs: int = 20,
    patience: int = 4,
    min_delta: float = 0.01,
    max_mel_frames: int = 1800,
    seed: int = 1337,
    nonpause_warn_frames: int = 100,
) -> dict[str, object]:
    root = Path(root).resolve()
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
            "training": training,
            "durations": None,
        }

    durations = generate_duration_cache(
        root,
        Path(str(training["best_checkpoint"])),
        nonpause_warn_frames=nonpause_warn_frames,
    )
    status = "pass" if durations["status"] == "pass" else "duration_gate_failed"
    result = {
        "status": status,
        "training": training,
        "durations": durations,
        "next_gate": (
            "aligned_acoustic_smoke"
            if status == "pass"
            else "review_duration_audit"
        ),
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
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
