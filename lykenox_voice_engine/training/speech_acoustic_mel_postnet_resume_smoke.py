"""Exact-resume gate for the isolated acoustic mel residual postnet trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from lykenox_voice_engine.training.speech_acoustic_mel_postnet_train import (
    TRAINER_CONTRACT_VERSION,
    run_mel_postnet_training,
)


SMOKE_VERSION = "acoustic-mel-postnet-exact-resume-smoke-v1"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_paths(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
        "acoustic_v2_last": training / "acoustic_frame_context_v2" / "last.pt",
        "old_mel_best": training / "acoustic_mel_fidelity_v1" / "best.pt",
        "old_mel_last": training / "acoustic_mel_fidelity_v1" / "last.pt",
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
        "persistent_postnet_last": training / "acoustic_mel_postnet_v1" / "last.pt",
        "persistent_postnet_best": training / "acoustic_mel_postnet_v1" / "best.pt",
    }


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("mel postnet resume-smoke checkpoint is invalid")
    return payload


def run_mel_postnet_resume_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected_paths(root)
    before = {name: _sha256(path) for name, path in protected.items()}
    persistent_absent_before = (
        before["persistent_postnet_last"] is None
        and before["persistent_postnet_best"] is None
    )
    if not persistent_absent_before:
        raise RuntimeError("mel postnet resume smoke must run before persistent postnet training")

    with tempfile.TemporaryDirectory(prefix="lykenox-postnet-resume-") as temporary:
        temporary_root = Path(temporary)
        direct_dir = temporary_root / "direct"
        split_dir = temporary_root / "split"
        common = {
            "batch_size": 2,
            "seed": 1701,
            "learning_rate": 1e-4,
            "checkpoint_every_updates": 1,
            "time_budget_seconds": 120.0,
            "checkpoint_reserve_seconds": 5.0,
            "dataset_item_limit": 8,
        }

        direct_report = run_mel_postnet_training(root, output_dir=direct_dir, **common)
        split_first = run_mel_postnet_training(
            root,
            output_dir=split_dir,
            max_updates_this_run=2,
            **common,
        )
        split_report = run_mel_postnet_training(
            root,
            output_dir=split_dir,
            max_updates_this_run=2,
            **common,
        )

        direct_last = direct_dir / "last.pt"
        split_last = split_dir / "last.pt"
        direct_payload = _load_payload(direct_last)
        split_payload = _load_payload(split_last)

        epoch_exact = int(direct_payload["epoch"]) == int(split_payload["epoch"]) == 2
        next_item_offset_exact = (
            int(direct_payload["next_item_offset"])
            == int(split_payload["next_item_offset"])
            == 0
        )
        global_step_exact = (
            int(direct_payload["global_step"])
            == int(split_payload["global_step"])
            == 4
        )
        postnet_state_exact = _nested_equal(
            direct_payload["postnet_state"], split_payload["postnet_state"]
        )
        optimizer_state_exact = _nested_equal(
            direct_payload["optimizer_state"], split_payload["optimizer_state"]
        )
        torch_rng_state_exact = _nested_equal(
            direct_payload["torch_rng_state"], split_payload["torch_rng_state"]
        )
        run_config_exact = direct_payload["run_config"] == split_payload["run_config"]
        metadata_exact = _nested_equal(
            direct_payload["training_metadata"], split_payload["training_metadata"]
        )
        base_identity_exact = (
            direct_payload["base_checkpoint_sha256"]
            == split_payload["base_checkpoint_sha256"]
            == direct_payload["run_config"]["base_checkpoint_sha256"]
        )

        before_rerun_sha = _sha256(split_last)
        rerun_report = run_mel_postnet_training(root, output_dir=split_dir, **common)
        after_rerun_sha = _sha256(split_last)
        gate_checkpoint_unchanged_on_rerun = before_rerun_sha == after_rerun_sha
        epoch1_gate_reached = all(
            report.get("status") == "gate_reached"
            for report in (direct_report, split_report, rerun_report)
        )
        epoch2_training_blocked = all(
            bool(report.get("epoch2_training_blocked", False))
            for report in (direct_report, split_report, rerun_report)
        )
        interruption_observed = split_first.get("status") == "incomplete"

    after = {name: _sha256(path) for name, path in protected.items()}
    protected_checkpoints_unchanged = before == after
    temporary_artifacts_removed = not temporary_root.exists()
    persistent_training_started = any(
        after[name] is not None
        for name in ("persistent_postnet_last", "persistent_postnet_best")
    )

    checks = {
        "interruption_observed": interruption_observed,
        "epoch_exact": epoch_exact,
        "next_item_offset_exact": next_item_offset_exact,
        "global_step_exact": global_step_exact,
        "postnet_state_exact": postnet_state_exact,
        "optimizer_state_exact": optimizer_state_exact,
        "torch_rng_state_exact": torch_rng_state_exact,
        "run_config_exact": run_config_exact,
        "metadata_exact": metadata_exact,
        "base_identity_exact": base_identity_exact,
        "epoch1_gate_reached": epoch1_gate_reached,
        "epoch2_training_blocked": epoch2_training_blocked,
        "gate_checkpoint_unchanged_on_rerun": gate_checkpoint_unchanged_on_rerun,
        "protected_checkpoints_unchanged": protected_checkpoints_unchanged,
        "temporary_artifacts_removed": temporary_artifacts_removed,
        "persistent_training_not_started": not persistent_training_started,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "smoke_version": SMOKE_VERSION,
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "updates_compared": 4,
        **checks,
        "persistent_training_started": persistent_training_started,
        "training_authorized": False,
        "next_gate": (
            "authorize_mel_postnet_first_epoch_training"
            if status == "pass"
            else "fix_mel_postnet_exact_resume_before_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_mel_postnet_resume_smoke(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
