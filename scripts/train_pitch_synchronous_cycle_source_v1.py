"""One-command train-and-listen entrypoint for the active pitch-synchronous source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.speech_vocoder_pitch_synchronous_cycle_source_train_v1 import (
    DEFAULT_CYCLES_PER_UPDATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_UPDATES,
    DEFAULT_SEED,
    DEFAULT_TRAIN_ITEMS,
    DEFAULT_VAL_ITEMS,
    train_pitch_synchronous_cycle_source_v1,
)
from scripts.render_pitch_synchronous_cycle_source_v1 import (
    render_heldout_pitch_synchronous_cycle_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--val-items", type=int, default=DEFAULT_VAL_ITEMS)
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--cycles-per-update", type=int, default=DEFAULT_CYCLES_PER_UPDATE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    training = train_pitch_synchronous_cycle_source_v1(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        max_updates=args.max_updates,
        cycles_per_update=args.cycles_per_update,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    )
    listening = render_heldout_pitch_synchronous_cycle_source(
        args.root,
        heldout_items=args.val_items,
        checkpoint=Path(str(training["best_checkpoint"])),
    )
    print(json.dumps({
        "status": "pitch_synchronous_cycle_source_train_and_listen_complete",
        "root_fix": "real_step3f_residual_cycles_in_f0_phase_coordinates",
        "training": training,
        "listening": listening,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
