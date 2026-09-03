"""One-command train + held-out listening entrypoint for the unified phase residual source V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.speech_vocoder_unified_phase_residual_source_train_v1 import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_UPDATES,
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_SEED,
    DEFAULT_TRAIN_ITEMS,
    DEFAULT_VAL_ITEMS,
    train_unified_phase_residual_source_v1,
)
from scripts.render_unified_phase_residual_source_v1 import render_heldout_unified_phase_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--val-items", type=int, default=DEFAULT_VAL_ITEMS)
    parser.add_argument("--segment-frames", type=int, default=DEFAULT_SEGMENT_FRAMES)
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--heldout-items", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    training = train_unified_phase_residual_source_v1(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        segment_frames=args.segment_frames,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    )
    listening = render_heldout_unified_phase_source(
        args.root,
        heldout_items=args.heldout_items,
        checkpoint=Path(str(training["best_checkpoint"])),
    )
    report = {
        "status": "unified_phase_residual_source_train_and_listen_complete",
        "root_fix": "single_joint_phase_aware_periodic_aperiodic_residual_source",
        "training_status": training["status"],
        "updates": training["updates"],
        "best_val_total": training["best_val_total"],
        "best_checkpoint": training["best_checkpoint"],
        "single_model": True,
        "single_recurrent_state": True,
        "second_source_checkpoint_fallback_used": False,
        "source_handoff_or_bridge_used": False,
        "codebook_used": False,
        "teacher_forcing_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "listening": listening,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
