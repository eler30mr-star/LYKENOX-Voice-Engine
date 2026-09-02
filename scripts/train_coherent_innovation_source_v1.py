"""One-command train-and-listen entrypoint for the active coherent + innovation source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.speech_vocoder_coherent_innovation_source_train_v1 import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_UPDATES,
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_SEED,
    DEFAULT_TRAIN_ITEMS,
    DEFAULT_VAL_ITEMS,
    train_coherent_innovation_source,
)
from scripts.render_coherent_innovation_source_v1 import render_heldout


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

    training = train_coherent_innovation_source(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        segment_frames=args.segment_frames,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    )
    listening = render_heldout(
        args.root,
        heldout_items=args.val_items,
        checkpoint=Path(str(training["best_checkpoint"])),
    )
    print(
        json.dumps(
            {
                "status": "coherent_innovation_source_train_and_listen_complete",
                "root_fix": "separate_coherent_residual_trajectory_from_stochastic_aperiodic_innovation",
                "training": training,
                "listening": listening,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
