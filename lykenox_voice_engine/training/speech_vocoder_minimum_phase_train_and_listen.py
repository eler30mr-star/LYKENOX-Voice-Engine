"""One-shot owned minimum-phase train -> complete held-out listening pipeline.

The command runs the scoped CPU-only candidate path.  The trainer performs its V2 authority
preflight internally before creating an optimizer.  Only a validation-selected ``best.pt``
may be rendered for listening.  Metrics never accept product voice quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.training.speech_vocoder_minimum_phase_heldout_audio import (
    render_heldout_audio,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_train_v2 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_UPDATES,
    DEFAULT_SEGMENT_MEL_FRAMES,
    DEFAULT_TRAIN_ITEMS,
    DEFAULT_VAL_ITEMS,
    TRAINER_VERSION,
    run_minimum_phase_training_v2,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_train_and_listen_contract import (
    CONTRACT_VERSION as TRAIN_AND_LISTEN_CONTRACT_VERSION,
)


PIPELINE_VERSION = "owned-minimum-phase-train-and-listen-v2"


def run_train_and_listen(
    root: Path,
    *,
    max_updates: int = DEFAULT_MAX_UPDATES,
    max_updates_this_run: int | None = None,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    train_items: int = DEFAULT_TRAIN_ITEMS,
    val_items: int = DEFAULT_VAL_ITEMS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    heldout_items: int = 5,
) -> dict[str, object]:
    root = Path(root).resolve()
    training = run_minimum_phase_training_v2(
        root,
        max_updates=max_updates,
        max_updates_this_run=max_updates_this_run,
        segment_mel_frames=segment_mel_frames,
        train_items=train_items,
        val_items=val_items,
        batch_size=batch_size,
        learning_rate=learning_rate,
        grad_clip=grad_clip,
    )
    if training.get("status") != "pass":
        return {
            "status": str(training.get("status", "training_not_ready_for_listening")),
            "pipeline_version": PIPELINE_VERSION,
            "trainer_version": TRAINER_VERSION,
            "train_and_listen_contract_version": TRAIN_AND_LISTEN_CONTRACT_VERSION,
            "training": training,
            "heldout_audio": None,
            "metrics_accept_voice_quality": False,
        }

    best_checkpoint = training.get("best_checkpoint")
    if not isinstance(best_checkpoint, str) or not best_checkpoint:
        raise RuntimeError("training reported pass without a validation-selected best checkpoint")
    heldout = render_heldout_audio(
        root,
        checkpoint=Path(best_checkpoint),
        split="val",
        max_items=heldout_items,
    )
    return {
        "status": "ready_for_listening",
        "pipeline_version": PIPELINE_VERSION,
        "trainer_version": TRAINER_VERSION,
        "train_and_listen_contract_version": TRAIN_AND_LISTEN_CONTRACT_VERSION,
        "training": training,
        "heldout_audio": heldout,
        "metrics_accept_voice_quality": False,
        "product_decision": "listen_to_complete_prediction_reference_pairs",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    parser.add_argument("--segment-mel-frames", type=int, default=DEFAULT_SEGMENT_MEL_FRAMES)
    parser.add_argument("--train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--val-items", type=int, default=DEFAULT_VAL_ITEMS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--heldout-items", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_train_and_listen(
                args.root,
                max_updates=args.max_updates,
                max_updates_this_run=args.max_updates_this_run,
                segment_mel_frames=args.segment_mel_frames,
                train_items=args.train_items,
                val_items=args.val_items,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                grad_clip=args.grad_clip,
                heldout_items=args.heldout_items,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
