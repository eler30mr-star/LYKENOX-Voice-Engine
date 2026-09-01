"""One-shot owned calibration -> calibrated glottal oracle runner.

Sequence:
1. measure glottal pulse statistics from all requested owned train utterances;
2. measure four-band aperiodicity from the same owned train source family;
3. render three complete held-out validation utterances with the calibrated excitation candidate.

This script performs calibration + fixed DSP inference only. It never trains a model, creates an
optimizer, loads/writes a checkpoint, modifies the production renderer, or accepts product quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostic_calibrated_glottal_oracle_v1 import run_calibrated_glottal_oracle
from lykenox_voice_engine.training.speech_band_aperiodicity_calibration import (
    run_band_aperiodicity_calibration,
)
from lykenox_voice_engine.training.speech_glottal_calibration import (
    DEFAULT_MAX_ITEMS,
    run_glottal_calibration,
)


RUNNER_VERSION = "owned-calibrated-glottal-candidate-one-shot-v1"
POLICY_ID = "LYX-POL-001"


def run_calibration_and_oracle(
    root: Path,
    *,
    max_train_items: int = DEFAULT_MAX_ITEMS,
    heldout_items: int = 3,
) -> dict[str, object]:
    root = Path(root).resolve()
    glottal = run_glottal_calibration(
        root,
        split="train",
        max_items=max_train_items,
    )
    aperiodicity = run_band_aperiodicity_calibration(
        root,
        split="train",
        max_items=max_train_items,
    )
    oracle = run_calibrated_glottal_oracle(
        root,
        split="val",
        max_items=heldout_items,
    )
    return {
        "status": "ready_for_human_listening",
        "runner_version": RUNNER_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "glottal_calibration_status": glottal.get("status"),
        "glottal_calibration_cycle_count": glottal.get("cycle_count"),
        "band_aperiodicity_calibration_status": aperiodicity.get("status"),
        "band_aperiodicity_cycle_count": aperiodicity.get("cycle_count"),
        "oracle_status": oracle.get("status"),
        "oracle_item_count": oracle.get("item_count"),
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "production_renderer_modified": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "next_action": "listen_to_calibrated_glottal_heldout_pairs_before_any_production_integration",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-train-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--heldout-items", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run_calibration_and_oracle(
                args.root,
                max_train_items=args.max_train_items,
                heldout_items=args.heldout_items,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
