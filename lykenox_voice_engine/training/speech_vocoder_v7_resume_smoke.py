"""Rejected V7 exact-resume-smoke compatibility surface.

Exact resume was correct, but that property cannot override perceptual rejection. The
historical smoke is retained as a status endpoint only and cannot create temporary V7
training runs anymore.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.training.speech_vocoder_v7_rejection import (
    V7_REJECTION_REASON,
    require_v7_training_enabled,
)
from lykenox_voice_engine.training.speech_vocoder_v7_train import (
    TRAINER_CONTRACT_VERSION,
)


SMOKE_VERSION = "vocoder-v7-first-epoch-exact-resume-smoke-v2"


def rejection_status() -> dict[str, object]:
    return {
        "status": "rejected",
        "smoke_version": SMOKE_VERSION,
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "historical_exact_resume_passed": True,
        "exact_resume_does_not_authorize_training": True,
        "persistent_v7_training_started": True,
        "persistent_training_complete": False,
        "epoch2_training_authorized": False,
        "reason": V7_REJECTION_REASON,
        "next_gate": "return_to_v4_2_and_require_grid_artifact_gate_before_training",
    }


def run_v7_resume_smoke(root: Path) -> dict[str, object]:
    del root
    require_v7_training_enabled()
    raise AssertionError("unreachable: rejected V7 resume smoke enabled")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.parse_args()
    print(json.dumps(rejection_status(), indent=2, ensure_ascii=False))
    require_v7_training_enabled()


if __name__ == "__main__":
    main()
