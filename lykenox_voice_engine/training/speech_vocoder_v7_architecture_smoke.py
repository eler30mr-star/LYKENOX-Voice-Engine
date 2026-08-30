"""Rejected V7 architecture-smoke compatibility surface.

The historical smoke proved shape/gradient trainability but failed to detect the
frame-grid tone that dominated full utterances. It is retained only to explain the
rejection and cannot authorize V7 training again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.models.vocoder import VOCODER_GENERATOR_V7_ARCHITECTURE
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    VOCODER_GRID_ARTIFACT_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_v7_rejection import (
    V7_REJECTION_REASON,
    require_v7_training_enabled,
)


SMOKE_VERSION = "vocoder-v7-source-free-content-smoke-v1"


def rejection_status() -> dict[str, object]:
    return {
        "status": "rejected",
        "smoke_version": SMOKE_VERSION,
        "architecture": VOCODER_GENERATOR_V7_ARCHITECTURE,
        "historical_smoke_invalidated": True,
        "invalidated_reason": (
            "The old smoke checked source-free structure and short-crop loss decrease, "
            "but did not test hop-period autocorrelation or frame-rate harmonic locking."
        ),
        "grid_artifact_detector_version": VOCODER_GRID_ARTIFACT_VERSION,
        "observed_frame_rate_hz": 93.75,
        "observed_second_harmonic_hz": 187.5,
        "reason": V7_REJECTION_REASON,
        "persistent_training_started": True,
        "persistent_training_complete": False,
        "next_gate": "return_to_v4_2_and_require_grid_artifact_gate_before_training",
    }


def run_v7_architecture_smoke(
    root: Path,
    **_legacy_arguments: object,
) -> dict[str, object]:
    del root
    require_v7_training_enabled()
    raise AssertionError("unreachable: rejected V7 architecture smoke enabled")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--segment-mel-frames", type=int, default=24)
    parser.add_argument("--updates", type=int, default=6)
    parser.parse_args()
    print(json.dumps(rejection_status(), indent=2, ensure_ascii=False))
    require_v7_training_enabled()


if __name__ == "__main__":
    main()
