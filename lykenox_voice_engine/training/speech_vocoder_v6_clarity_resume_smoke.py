"""Exact-resume gate for the V6 clarity-guard training experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    VOCODER_LEVEL_PRESENCE_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_v6_clarity_train import (
    EXPERIMENT_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_v6_resume_smoke import (
    run_v6_resume_smoke,
)


SMOKE_VERSION = "vocoder-v6-clarity-exact-resume-smoke-v1"
EXPECTED_LEVEL_PRESENCE_VERSION = "vocoder-level-presence-v3"


def run_v6_clarity_resume_smoke(root: Path) -> dict[str, object]:
    base = run_v6_resume_smoke(root)
    loss_version_exact = (
        VOCODER_LEVEL_PRESENCE_VERSION == EXPECTED_LEVEL_PRESENCE_VERSION
    )
    passed = base.get("status") == "pass" and loss_version_exact
    return {
        **base,
        "status": "pass" if passed else "fail",
        "smoke_version": SMOKE_VERSION,
        "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
        "level_presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
        "level_presence_loss_version_exact": loss_version_exact,
        "persistent_v6_training_started": False,
        "next_gate": (
            "start_bounded_resumable_v6_clarity_guard_training"
            if passed
            else "fix_v6_clarity_resume_gate_before_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_v6_clarity_resume_smoke(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
