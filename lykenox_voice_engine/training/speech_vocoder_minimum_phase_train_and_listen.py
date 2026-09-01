"""One-shot owned minimum-phase train -> held-out audio pipeline.

This command exists to collapse the workflow to the product-relevant path: train the owned
predictor with the active minimum-phase v2 objective, then render complete validation
utterances for listening.  It does not treat metrics as product acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.training.speech_vocoder_minimum_phase_heldout_audio import (
    render_heldout_audio,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_UPDATES,
    DEFAULT_SEGMENT_MEL_FRAMES,
    DEFAULT_TRAIN_ITEMS,
    DEFAULT_VAL_ITEMS,
    run_minimum_phase_training,
)


PIPELINE_VERSION = "owned-minimum-phase-train-and-listen-v1"


def run_train_and_listen(
    root: Path,
    *,
    max_updates: int = DEFAULT_MAX_UPDATES,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    train_items: int = DEFAULT_TRAIN_ITEMS,
    val_items: int = DEFAULT_VAL_ITEMS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    heldout_items: int = 5,
) -> dict[str, object]:
    root = Path(root).resolve()
    training = run_minimum_phase_training(
        root,
        max_updates=max_updates,
        segment_mel_frames=segment_mel_frames,
        train_items=train_items,
        val_items=val_items,
        batch_size=batch_size,
        learning_rate=learning_rate,
        grad_clip=grad_clip,
    )
    if training.get("status") == "incomplete":
        return {
            "status": "training_incomplete",
            "pipeline_version": PIPELINE_VERSION,
            "training": training,
            "heldout_audio": None,
        }

    heldout = render_heldout_audio(
        root,
        split="val",
        max_items=heldout_items,
    )
    return {
        "status": "ready_for_listening",
        "pipeline_version": PIPELINE_VERSION,
        "training": training,
        "heldout_audio": heldout,
        "metrics_accept_voice_quality": False,
        "product_decision": "listen_to_complete_prediction_reference_pairs",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
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
