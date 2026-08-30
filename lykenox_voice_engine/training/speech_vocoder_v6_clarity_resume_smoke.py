"""Exact-resume gate for the V6 clarity-guard training experiment."""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_v6_clarity_resume_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    prior_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_direct_waveform_v6"
    )
    prior_paths = {
        "v6_prior_last": prior_dir / "last.pt",
        "v6_prior_best": prior_dir / "best.pt",
    }
    before = {name: _sha256_if_exists(path) for name, path in prior_paths.items()}

    base = run_v6_resume_smoke(root)

    after = {name: _sha256_if_exists(path) for name, path in prior_paths.items()}
    prior_v6_unchanged = before == after
    loss_version_exact = (
        VOCODER_LEVEL_PRESENCE_VERSION == EXPECTED_LEVEL_PRESENCE_VERSION
    )
    passed = (
        base.get("status") == "pass"
        and loss_version_exact
        and prior_v6_unchanged
    )
    return {
        **base,
        "status": "pass" if passed else "fail",
        "smoke_version": SMOKE_VERSION,
        "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
        "level_presence_loss_version": VOCODER_LEVEL_PRESENCE_VERSION,
        "level_presence_loss_version_exact": loss_version_exact,
        "prior_v6_checkpoints_present": {
            name: digest is not None for name, digest in before.items()
        },
        "prior_v6_checkpoints_unchanged": prior_v6_unchanged,
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
