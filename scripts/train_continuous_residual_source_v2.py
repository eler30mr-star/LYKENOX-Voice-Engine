"""One-command train-and-render entrypoint for the active LYKENOX continuous residual source V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v2 import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_UPDATES,
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_SEED,
    DEFAULT_TRAIN_ITEMS,
    DEFAULT_VAL_ITEMS,
    train_continuous_residual_source_v2,
)
from scripts.render_continuous_residual_source_v2 import render_heldout_v2


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

    training = train_continuous_residual_source_v2(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        segment_frames=args.segment_frames,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    )
    listening = render_heldout_v2(
        args.root,
        heldout_items=args.val_items,
        checkpoint=Path(str(training["best_checkpoint"])),
    )
    print(
        json.dumps(
            {
                "status": "continuous_residual_source_v2_train_and_listen_complete",
                "root_fix": "explicit_residual_shape_and_level_factorization",
                "training": training,
                "listening": listening,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
