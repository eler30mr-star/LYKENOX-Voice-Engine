"""Rejected V6 trainer compatibility surface.

The original bounded/resumable implementation is intentionally no longer executable.
Full-utterance oracle listening on 2026-08-30 showed that V6 produced unintelligible,
periodic, nasal/gangoso output that was materially worse than v4.2. Historical V6
checkpoints remain loadable for forensic analysis, but no new or resumed V6 updates are
allowed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV6
from lykenox_voice_engine.training.speech_vocoder_v6_rejection import (
    V6_PERCEPTUALLY_REJECTED,
    V6_REJECTION_DATE,
    V6_REJECTION_GATE,
    V6_REJECTION_REASON,
    V6_TRAINING_ENABLED,
    require_v6_training_enabled,
)


# Kept stable so historical checkpoint metadata and inexpensive optimizer-contract tests
# remain interpretable. This version is rejected and must never be treated as active.
TRAINER_CONTRACT_VERSION = "v6-bounded-resumable-v1"
TRAIN_SEGMENT_SCHEDULE_VERSION = "epoch-crop-shuffle-v1"
DEFAULT_TIME_BUDGET_SECONDS = 80.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 10.0
DEFAULT_VALIDATION_RESERVE_SECONDS = 20.0
DEFAULT_VALIDATION_SEED_OFFSET = 100_003


def _optimizer(
    generator: LykenoxVocoderGeneratorV6,
    lr: float,
    level_mult: float,
) -> torch.optim.AdamW:
    """Historical optimizer grouping retained for checkpoint/test forensics only."""
    level = list(generator.level_parameters())
    level_ids = {id(parameter) for parameter in level}
    shape = [
        parameter
        for parameter in generator.parameters()
        if id(parameter) not in level_ids
    ]
    if not shape or not level:
        raise RuntimeError("v6 optimizer grouping is incomplete")
    return torch.optim.AdamW(
        [
            {"params": shape, "lr": lr, "weight_decay": 1e-5},
            {
                "params": level,
                "lr": lr * level_mult,
                "weight_decay": 0.0,
            },
        ]
    )


def run_bounded_resumable_v6_training(
    root: Path,
    **_legacy_arguments: object,
) -> dict[str, object]:
    """Reject every new or resumed V6 training invocation before any side effect."""
    del root
    require_v6_training_enabled()
    raise AssertionError("unreachable: rejected V6 training unexpectedly enabled")


def rejection_status() -> dict[str, object]:
    return {
        "status": "rejected",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "training_enabled": V6_TRAINING_ENABLED,
        "perceptually_rejected": V6_PERCEPTUALLY_REJECTED,
        "rejection_date": V6_REJECTION_DATE,
        "rejection_gate": V6_REJECTION_GATE,
        "reason": V6_REJECTION_REASON,
        "historical_checkpoints_forensic_only": True,
        "persistent_training_complete": False,
        "next_gate": "design_new_architecture_without_sample_phase_or_unit_rms_shortcuts",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    parser.parse_args()
    # Print the diagnosis before returning a failing exit status via RuntimeError.
    print(json.dumps(rejection_status(), indent=2, ensure_ascii=False))
    require_v6_training_enabled()


if __name__ == "__main__":
    main()
