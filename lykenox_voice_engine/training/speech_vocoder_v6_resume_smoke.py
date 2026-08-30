"""Exact-resume gate for the bounded LYKENOX v6 persistent trainer.

This smoke compares four uninterrupted optimizer updates with two updates + checkpoint/load
+ two updates. All artifacts live in temporary directories. Historical checkpoints are
hashed before and after and must remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from lykenox_voice_engine.training.speech_vocoder_v6_train import (
    TRAINER_CONTRACT_VERSION,
    run_bounded_resumable_v6_training,
)


SMOKE_VERSION = "vocoder-v6-exact-resume-smoke-v1"


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
        return a.keys() == b.keys() and all(
            _exact(a[key], b[key])
            for key in a
        )
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _exact(x, y)
            for x, y in zip(a, b, strict=True)
        )
    return a == b


def _load_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def run_v6_resume_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    historical = {
        "v4_2": (
            root
            / "models"
            / "lykenox_identity"
            / "training"
            / "vocoder_source_filter_v4_2"
            / "best.pt"
        ),
        "v4_3": (
            root
            / "models"
            / "lykenox_identity"
            / "training"
            / "vocoder_mel_filtered_carrier_v4_3"
            / "best.pt"
        ),
        "v4_4": (
            root
            / "models"
            / "lykenox_identity"
            / "training"
            / "vocoder_dynamic_filter_hybrid_v4_4"
            / "best.pt"
        ),
        "v5": (
            root
            / "models"
            / "lykenox_identity"
            / "training"
            / "vocoder_stochastic_glottal_filter_v5"
            / "best.pt"
        ),
    }
    before = {
        name: _sha256_if_exists(path)
        for name, path in historical.items()
    }

    common = dict(
        segment_mel_frames=32,
        train_items=5,
        val_items=2,
        max_epochs=2,
        warmup_epochs=0,
        patience=2,
        seed=66000,
        generator_lr=2e-4,
        level_lr_multiplier=4.0,
        discriminator_lr=1e-4,
        envelope_weight=0.50,
        balance_weight=0.25,
        contrast_weight=0.15,
        level_weight=0.75,
        presence_weight=0.35,
        adversarial_weight=0.03,
        feature_matching_weight=0.50,
        checkpoint_every_updates=2,
        time_budget_seconds=100.0,
    )

    with tempfile.TemporaryDirectory(
        prefix="lykenox_v6_resume_"
    ) as temporary:
        temp = Path(temporary)
        direct_dir = temp / "direct"
        split_dir = temp / "split"

        direct = run_bounded_resumable_v6_training(
            root,
            artifact_dir_override=direct_dir,
            max_updates_this_run=4,
            **common,
        )
        split_first = run_bounded_resumable_v6_training(
            root,
            artifact_dir_override=split_dir,
            max_updates_this_run=2,
            **common,
        )
        split_second = run_bounded_resumable_v6_training(
            root,
            artifact_dir_override=split_dir,
            max_updates_this_run=2,
            **common,
        )

        direct_payload = _load_payload(direct_dir / "last.pt")
        split_payload = _load_payload(split_dir / "last.pt")

        direct_metadata = direct_payload.get("training_metadata")
        split_metadata = split_payload.get("training_metadata")
        direct_optimizer = direct_payload.get("generator_optimizer_state")
        split_optimizer = split_payload.get("generator_optimizer_state")
        optimizer_has_two_groups = (
            isinstance(direct_optimizer, dict)
            and isinstance(split_optimizer, dict)
            and isinstance(direct_optimizer.get("param_groups"), list)
            and isinstance(split_optimizer.get("param_groups"), list)
            and len(direct_optimizer["param_groups"]) == 2
            and len(split_optimizer["param_groups"]) == 2
        )

        checks = {
            "global_step_exact": (
                direct_payload.get("global_step")
                == split_payload.get("global_step")
                == 4
            ),
            "epoch_exact": (
                direct_payload.get("epoch")
                == split_payload.get("epoch")
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
                direct_optimizer,
                split_optimizer,
            ),
            "generator_optimizer_two_groups": optimizer_has_two_groups,
            "discriminator_optimizer_exact": _exact(
                direct_payload.get("discriminator_optimizer_state"),
                split_payload.get("discriminator_optimizer_state"),
            ),
            "torch_rng_state_exact": _exact(
                direct_payload.get("torch_rng_state"),
                split_payload.get("torch_rng_state"),
            ),
            "run_config_exact": (
                isinstance(direct_metadata, dict)
                and isinstance(split_metadata, dict)
                and direct_metadata.get("run_config")
                == split_metadata.get("run_config")
            ),
            "architecture_exact": (
                direct_payload.get("generator_architecture")
                == split_payload.get("generator_architecture")
                == "lykenox_direct_conditional_waveform_v6_level_decoupled"
            ),
            "source_family_exact": (
                direct_payload.get("source_family")
                == split_payload.get("source_family")
                == "direct_conditional_waveform_decoder"
            ),
            "no_explicit_source_exact": (
                direct_payload.get("explicit_source") is False
                and split_payload.get("explicit_source") is False
            ),
            "no_voiced_noise_source_exact": (
                direct_payload.get("voiced_noise_source") is False
                and split_payload.get("voiced_noise_source") is False
            ),
            "no_raw_source_bypass_exact": (
                direct_payload.get("raw_source_bypass") is False
                and split_payload.get("raw_source_bypass") is False
            ),
            "waveform_shape_level_decoupled_exact": (
                direct_payload.get("waveform_shape_level_decoupled") is True
                and split_payload.get("waveform_shape_level_decoupled") is True
            ),
            "level_control_family_exact": (
                direct_payload.get("level_control_family")
                == split_payload.get("level_control_family")
                == "mel_conditioned_frame_rms_envelope"
            ),
        }

        direct_stop = str(direct.get("stop_reason"))
        split_first_stop = str(split_first.get("stop_reason"))
        split_second_stop = str(split_second.get("stop_reason"))

    after = {
        name: _sha256_if_exists(path)
        for name, path in historical.items()
    }
    historical_unchanged = before == after
    all_exact = (
        all(bool(value) for value in checks.values())
        and historical_unchanged
    )

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
        "historical_checkpoints_present": {
            name: digest is not None
            for name, digest in before.items()
        },
        "historical_checkpoints_unchanged": historical_unchanged,
        "temporary_artifacts_removed": True,
        "persistent_v6_training_started": False,
        "next_gate": (
            "start_bounded_resumable_v6_persistent_training"
            if all_exact
            else "fix_v6_resume_contract_before_persistent_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_v6_resume_smoke(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
