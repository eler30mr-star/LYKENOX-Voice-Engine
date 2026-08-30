"""Retired V6 architecture smoke.

The former crop-level smoke could pass while complete utterances were unintelligible.
It is retained as a command surface only to report the authoritative perceptual rejection;
it performs no data loading, optimization, or checkpoint mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV6,
    VOCODER_GENERATOR_V6_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_vocoder_v6_rejection import (
    V6_REJECTION_DATE,
    V6_REJECTION_GATE,
    V6_REJECTION_REASON,
)


SMOKE_VERSION = "vocoder-v6-retired-after-perceptual-rejection-v3"


def run_v6_architecture_smoke(root: Path) -> dict[str, object]:
    del root
    return {
        "status": "rejected",
        "smoke_version": SMOKE_VERSION,
        "architecture": VOCODER_GENERATOR_V6_ARCHITECTURE,
        "source_free": LykenoxVocoderGeneratorV6.source_free,
        "sample_phase_conditioning": (
            LykenoxVocoderGeneratorV6.sample_phase_conditioning
        ),
        "deterministic_unvoiced_noise_conditioning": (
            LykenoxVocoderGeneratorV6.deterministic_unvoiced_noise_conditioning
        ),
        "local_unit_rms_shape_normalization": (
            LykenoxVocoderGeneratorV6.local_unit_rms_shape_normalization
        ),
        "perceptually_rejected": LykenoxVocoderGeneratorV6.perceptually_rejected,
        "rejection_date": V6_REJECTION_DATE,
        "rejection_gate": V6_REJECTION_GATE,
        "reason": V6_REJECTION_REASON,
        "persistent_training_started": False,
        "historical_checkpoints_mutated": False,
        "next_gate": "design_new_architecture_without_sample_phase_or_unit_rms_shortcuts",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_v6_architecture_smoke(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
