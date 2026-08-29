"""Exact-resume smoke for the bounded LYKENOX v4.4 persistent trainer.

Runs the same four deterministic CPU updates two ways:
1) one invocation with four updates;
2) two invocations with two updates each.

The gate passes only when generator, discriminator, both optimizers, RNG, epoch/offset and
run config are bit-exact. Historical v4.3 is hashed before/after and temporary v4.4 artifacts
are deleted automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from lykenox_voice_engine.training.speech_vocoder_v4_4_train import (
    TRAINER_CONTRACT_VERSION,
    run_bounded_resumable_v4_4_training,
)


SMOKE_VERSION = "vocoder-v4-4-exact-resume-smoke-v1"


def _sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact(a: object, b: object) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return (
            a.dtype == b.dtype
            and tuple(a.shape) == tuple(b.shape)
            and torch.equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_exact(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _exact(x, y) for x, y in zip(a, b, strict=True)
        )
    return a == b


def _load_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def run_v4_4_resume_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    v4_3_best = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_mel_filtered_carrier_v4_3"
        / "best.pt"
    )
    v4_3_before = _sha256_if_exists(v4_3_best)

    common = dict(
        segment_mel_frames=32,
        train_items=5,
        val_items=2,
        max_epochs=2,
        warmup_epochs=0,
        patience=2,
        seed=44440,
        generator_lr=2e-4,
        discriminator_lr=1e-4,
        envelope_weight=0.50,
        balance_weight=0.25,
        contrast_weight=0.15,
        harmonic_exposure_weight=0.25,
        adversarial_weight=0.03,
        feature_matching_weight=0.50,
        checkpoint_every_updates=2,
        time_budget_seconds=100.0,
    )

    with tempfile.TemporaryDirectory(prefix="lykenox_v44_resume_") as temporary:
        temp = Path(temporary)
        direct_dir = temp / "direct"
        split_dir = temp / "split"

        direct = run_bounded_resumable_v4_4_training(
            root,
            artifact_dir_override=direct_dir,
            max_updates_this_run=4,
            **common,
        )
        split_first = run_bounded_resumable_v4_4_training(
            root,
            artifact_dir_override=split_dir,
            max_updates_this_run=2,
            **common,
        )
        split_second = run_bounded_resumable_v4_4_training(
            root,
            artifact_dir_override=split_dir,
            max_updates_this_run=2,
            **common,
        )

        direct_payload = _load_payload(direct_dir / "last.pt")
        split_payload = _load_payload(split_dir / "last.pt")
        checks = {
            "global_step_exact": (
                direct_payload.get("global_step")
                == split_payload.get("global_step")
                == 4
            ),
            "epoch_exact": (
                direct_payload.get("epoch") == split_payload.get("epoch")
            ),
            "next_item_offset_exact": (
                direct_payload.get("next_item_offset")
                == split_payload.get("next_item_offset")
            ),
            "generator_state_exact": _exact(
                direct_payload.get("generator_state"),
                split_payload.get("generator_state"),
            ),
            "discriminator_state_exact": _exact(
                direct_payload.get("discriminator_state"),
                split_payload.get("discriminator_state"),
            ),
            "generator_optimizer_exact": _exact(
                direct_payload.get("generator_optimizer_state"),
                split_payload.get("generator_optimizer_state"),
            ),
            "discriminator_optimizer_exact": _exact(
                direct_payload.get("discriminator_optimizer_state"),
                split_payload.get("discriminator_optimizer_state"),
            ),
            "torch_rng_state_exact": _exact(
                direct_payload.get("torch_rng_state"),
                split_payload.get("torch_rng_state"),
            ),
            "run_config_exact": (
                isinstance(direct_payload.get("training_metadata"), dict)
                and isinstance(split_payload.get("training_metadata"), dict)
                and direct_payload["training_metadata"].get("run_config")
                == split_payload["training_metadata"].get("run_config")
            ),
        }
        direct_stop = str(direct.get("stop_reason"))
        split_first_stop = str(split_first.get("stop_reason"))
        split_second_stop = str(split_second.get("stop_reason"))

    v4_3_after = _sha256_if_exists(v4_3_best)
    v4_3_unchanged = v4_3_before == v4_3_after
    all_exact = all(bool(value) for value in checks.values()) and v4_3_unchanged

    return {
        "status": "pass" if all_exact else "fail",
        "smoke_version": SMOKE_VERSION,
        "device": "cpu",
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "updates_compared": 4,
        "direct_stop_reason": direct_stop,
        "split_first_stop_reason": split_first_stop,
        "split_second_stop_reason": split_second_stop,
        **checks,
        "v4_3_checkpoint_present": v4_3_before is not None,
        "v4_3_checkpoint_unchanged": v4_3_unchanged,
        "temporary_artifacts_removed": True,
        "persistent_v4_4_training_started": False,
        "next_gate": (
            "start_bounded_resumable_v4_4_persistent_training"
            if all_exact
            else "fix_v4_4_resume_contract_before_persistent_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_v4_4_resume_smoke(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
