"""Rejected V6 clarity-guard training entrypoint.

The clarity experiment is preserved by name for reproducibility, but the 2026-08-30
full-utterance oracle listening result rejected it perceptually. Invocations fail before
artifact-directory creation or checkpoint mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.training.speech_vocoder_v6_rejection import (
    V6_PERCEPTUALLY_REJECTED,
    V6_REJECTION_DATE,
    V6_REJECTION_GATE,
    V6_REJECTION_REASON,
    V6_TRAINING_ENABLED,
    require_v6_training_enabled,
)
from lykenox_voice_engine.training.speech_vocoder_v6_train import (
    DEFAULT_TIME_BUDGET_SECONDS,
)


EXPERIMENT_CONTRACT_VERSION = "v6-clarity-guard-v1"
ARTIFACT_DIR_NAME = "vocoder_direct_waveform_v6_clarity_guard_v1"


def run_v6_clarity_guard_training(
    root: Path,
    *,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    max_updates_this_run: int | None = None,
) -> dict[str, object]:
    """Reject before resolving paths, reading data, or touching checkpoints."""
    del root, time_budget_seconds, max_updates_this_run
    require_v6_training_enabled()
    raise AssertionError("unreachable: rejected V6 clarity training unexpectedly enabled")


def rejection_status() -> dict[str, object]:
    return {
        "status": "rejected",
        "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
        "artifact_dir_name": ARTIFACT_DIR_NAME,
        "training_enabled": V6_TRAINING_ENABLED,
        "perceptually_rejected": V6_PERCEPTUALLY_REJECTED,
        "rejection_date": V6_REJECTION_DATE,
        "rejection_gate": V6_REJECTION_GATE,
        "reason": V6_REJECTION_REASON,
        "checkpoint_mutation_allowed": False,
        "next_gate": "design_new_architecture_without_sample_phase_or_unit_rms_shortcuts",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
    )
    parser.add_argument("--max-updates-this-run", type=int, default=None)
    parser.parse_args()
    print(json.dumps(rejection_status(), indent=2, ensure_ascii=False))
    require_v6_training_enabled()


if __name__ == "__main__":
    main()
