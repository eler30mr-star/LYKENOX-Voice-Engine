"""Persistent V6 clarity-guard training entrypoint.

This launcher intentionally writes to a fresh artifact directory so the earlier
vocoder_direct_waveform_v6 experiment remains frozen for diagnosis. The underlying
trainer remains bounded and exactly resumable; the active level/presence loss version
is carried in checkpoint provenance and run_config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_v6_train import (
    DEFAULT_TIME_BUDGET_SECONDS,
    run_bounded_resumable_v6_training,
)


EXPERIMENT_CONTRACT_VERSION = "v6-clarity-guard-v1"
ARTIFACT_DIR_NAME = "vocoder_direct_waveform_v6_clarity_guard_v1"


def run_v6_clarity_guard_training(
    root: Path,
    *,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    max_updates_this_run: int | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / ARTIFACT_DIR_NAME
    )
    result = run_bounded_resumable_v6_training(
        root,
        time_budget_seconds=time_budget_seconds,
        max_updates_this_run=max_updates_this_run,
        artifact_dir_override=artifact_dir,
    )
    return {
        **result,
        "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
        "level_presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
        "prior_v6_experiment_preserved": True,
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
    args = parser.parse_args()
    print(
        json.dumps(
            run_v6_clarity_guard_training(
                args.root,
                time_budget_seconds=args.time_budget_seconds,
                max_updates_this_run=args.max_updates_this_run,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
