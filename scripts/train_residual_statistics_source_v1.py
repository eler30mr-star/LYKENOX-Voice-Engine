"""One-command train-and-render entrypoint for residual-statistics source V1.

Persistent source training is policy-blocked until CLEAN_V1 is active, every stale acoustic
feature/target/cache has been regenerated from the clean WAVs, and the post-clean GOLD oracle gate
has passed. Policy: LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.identity_voice_clean_v1 import require_clean_v1_training_ready
from lykenox_voice_engine.training.speech_vocoder_residual_statistics_source_train_v1 import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_UPDATES,
    DEFAULT_SEED,
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_TRAIN_ITEMS,
    DEFAULT_VAL_ITEMS,
    train_residual_statistics_source_v1,
)
from scripts.render_residual_statistics_source_v1 import render_heldout_residual_statistics_source_v1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--val-items", type=int, default=DEFAULT_VAL_ITEMS)
    parser.add_argument("--segment-frames", type=int, default=DEFAULT_SEGMENT_FRAMES)
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    require_clean_v1_training_ready(
        args.root,
        purpose="residual-statistics source persistent training",
    )

    training = train_residual_statistics_source_v1(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        segment_frames=args.segment_frames,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    )
    listening = render_heldout_residual_statistics_source_v1(
        args.root,
        heldout_items=args.val_items,
        checkpoint=Path(str(training["best_checkpoint"])),
    )
    print(json.dumps({
        "status": "residual_statistics_source_train_and_gate_complete",
        "root_correction": "replace_deterministic_frame_waveform_regression_with_source_statistics_and_continuous_carrier",
        "training": training,
        "listening": listening,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
