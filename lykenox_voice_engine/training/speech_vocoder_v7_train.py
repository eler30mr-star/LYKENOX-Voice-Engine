"""Rejected V7 trainer compatibility surface.

The original bounded/resumable implementation is intentionally no longer executable.
Full-utterance oracle listening on 2026-08-30 showed that V7 produced a frame-grid hum
instead of intelligible speech. Historical V7 checkpoints remain loadable for forensic
analysis, but no new or resumed V7 updates are allowed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.models.vocoder import VOCODER_GENERATOR_V7_ARCHITECTURE
from lykenox_voice_engine.training.speech_vocoder_v7_content_loss import (
    VOCODER_V7_CONTENT_LOSS_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_v7_rejection import (
    V7_PERCEPTUALLY_REJECTED,
    V7_REJECTION_DATE,
    V7_REJECTION_GATE,
    V7_REJECTION_REASON,
    V7_TRAINING_ENABLED,
    require_v7_training_enabled,
)


# Retained so historical checkpoint metadata and contract tests remain interpretable.
TRAINER_CONTRACT_VERSION = "v7-first-epoch-bounded-resumable-v1"
TRAIN_SEGMENT_SCHEDULE_VERSION = "epoch-crop-shuffle-v1"
ARTIFACT_DIR_NAME = "vocoder_source_free_v7_first_epoch"
DEFAULT_TIME_BUDGET_SECONDS = 80.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 10.0
DEFAULT_VALIDATION_RESERVE_SECONDS = 20.0
DEFAULT_VALIDATION_SEED_OFFSET = 100_003
V7_FIRST_EPOCH_GENERATOR_KWARGS = {
    "frame_channels": 96,
    "upsample_channels": (80, 56, 40),
    "upsample_factors": (8, 8, 4),
    "residual_kernels": (3, 7),
    "residual_dilations": (1, 3),
}


def _run_config(**kwargs: object) -> dict[str, object]:
    """Historical run-config surface retained for checkpoint forensics only."""
    return {
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "generator_architecture": VOCODER_GENERATOR_V7_ARCHITECTURE,
        "generator_hyperparameters": dict(V7_FIRST_EPOCH_GENERATOR_KWARGS),
        "source_free": True,
        "sample_phase_conditioning": False,
        "sample_rate_pitch_features": False,
        "pitch_conditioning_scope": "frame_latent_only",
        "deterministic_noise_conditioning": False,
        "level_rescue_branch": False,
        "v7_content_loss_version": VOCODER_V7_CONTENT_LOSS_VERSION,
        "train_segment_schedule_version": TRAIN_SEGMENT_SCHEDULE_VERSION,
        "hard_epoch_limit": 1,
        "training_enabled": False,
        "perceptually_rejected": True,
        **kwargs,
    }


def run_bounded_resumable_v7_first_epoch(
    root: Path,
    **_legacy_arguments: object,
) -> dict[str, object]:
    """Reject every new or resumed V7 training invocation before any side effect."""
    del root
    require_v7_training_enabled()
    raise AssertionError("unreachable: rejected V7 training unexpectedly enabled")


def rejection_status() -> dict[str, object]:
    return {
        "status": "rejected",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "architecture": VOCODER_GENERATOR_V7_ARCHITECTURE,
        "training_enabled": V7_TRAINING_ENABLED,
        "perceptually_rejected": V7_PERCEPTUALLY_REJECTED,
        "rejection_date": V7_REJECTION_DATE,
        "rejection_gate": V7_REJECTION_GATE,
        "reason": V7_REJECTION_REASON,
        "historical_checkpoints_forensic_only": True,
        "persistent_training_complete": False,
        "epoch2_training_authorized": False,
        "next_gate": "return_to_v4_2_and_require_frame_grid_artifact_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    parser.parse_args()
    print(json.dumps(rejection_status(), indent=2, ensure_ascii=False))
    require_v7_training_enabled()


if __name__ == "__main__":
    main()
