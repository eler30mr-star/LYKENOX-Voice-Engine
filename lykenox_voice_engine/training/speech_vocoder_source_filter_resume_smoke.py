"""Exact-resume gate for the bounded LYKENOX v4.1 persistent trainer.

The checkpoint contract smoke proved that state can be serialized.  This gate proves the
stronger property required by the target command environment: stopping *mid epoch* and
rerunning the trainer reaches the same model/optimizer/RNG state as an uninterrupted run
after the same number of updates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_source_filter_artifact import (
    load_source_filter_checkpoint,
)
from lykenox_voice_engine.training.speech_vocoder_source_filter_train import (
    TRAINER_CONTRACT_VERSION,
    run_bounded_resumable_source_filter_training,
)


SMOKE_VERSION = "source-filter-exact-resume-smoke-v1"


def _exact_nested(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _exact_nested(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _exact_nested(first, second)
            for first, second in zip(left, right, strict=True)
        )
    return left == right


def run_resume_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    artifact_root = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "source_filter_resume_smoke"
    )
    reference_dir = artifact_root / "reference_uninterrupted"
    split_dir = artifact_root / "split_resume"
    shutil.rmtree(reference_dir, ignore_errors=True)
    shutil.rmtree(split_dir, ignore_errors=True)

    common: dict[str, object] = {
        "segment_mel_frames": 64,
        "train_items": 4,
        "val_items": 2,
        "max_epochs": 4,
        "warmup_epochs": 1,
        "patience": 10,
        "seed": 1337,
        "checkpoint_every_updates": 2,
        "time_budget_seconds": 50.0,
        "checkpoint_reserve_seconds": 5.0,
    }

    reference_result = run_bounded_resumable_source_filter_training(
        root,
        **common,
        max_updates_this_run=10,
        artifact_dir_override=reference_dir,
    )
    split_first = run_bounded_resumable_source_filter_training(
        root,
        **common,
        max_updates_this_run=3,
        artifact_dir_override=split_dir,
    )
    split_second = run_bounded_resumable_source_filter_training(
        root,
        **common,
        max_updates_this_run=7,
        artifact_dir_override=split_dir,
    )

    reference_last = reference_dir / "last.pt"
    split_last = split_dir / "last.pt"
    reference_generator, reference_discriminator, reference_payload = (
        load_source_filter_checkpoint(reference_last)
    )
    split_generator, split_discriminator, split_payload = (
        load_source_filter_checkpoint(split_last)
    )

    generator_state_exact = _exact_nested(
        reference_generator.state_dict(),
        split_generator.state_dict(),
    )
    discriminator_state_exact = _exact_nested(
        reference_discriminator.state_dict(),
        split_discriminator.state_dict(),
    )
    generator_optimizer_exact = _exact_nested(
        reference_payload.get("generator_optimizer_state"),
        split_payload.get("generator_optimizer_state"),
    )
    discriminator_optimizer_exact = _exact_nested(
        reference_payload.get("discriminator_optimizer_state"),
        split_payload.get("discriminator_optimizer_state"),
    )
    torch_rng_exact = _exact_nested(
        reference_payload.get("torch_rng_state"),
        split_payload.get("torch_rng_state"),
    )
    position_exact = all(
        reference_payload.get(key) == split_payload.get(key)
        for key in ("epoch", "global_step", "next_item_offset")
    )

    val_segments, _ = collect_vocoder_segments(
        root,
        "val",
        segment_mel_frames=64,
        max_items=1,
        seed=1337 + 100_003,
    )
    probe = val_segments[0]
    pitch = extract_pitch_frames(
        probe.waveform,
        frame_count=probe.mel_frames,
    )
    with torch.no_grad():
        reference_waveform = reference_generator(
            probe.mel.unsqueeze(0),
            pitch.f0_hz.unsqueeze(0),
            pitch.voiced.unsqueeze(0),
        )
        split_waveform = split_generator(
            probe.mel.unsqueeze(0),
            pitch.f0_hz.unsqueeze(0),
            pitch.voiced.unsqueeze(0),
        )
    waveform_delta = float((reference_waveform - split_waveform).abs().max())
    waveform_exact = waveform_delta == 0.0

    reference_metadata = reference_payload.get("training_metadata")
    split_metadata = split_payload.get("training_metadata")
    partial_epoch_state_exact = (
        isinstance(reference_metadata, dict)
        and isinstance(split_metadata, dict)
        and _exact_nested(
            reference_metadata.get("partial_epoch_state"),
            split_metadata.get("partial_epoch_state"),
        )
    )
    history_exact = (
        isinstance(reference_metadata, dict)
        and isinstance(split_metadata, dict)
        and _exact_nested(
            reference_metadata.get("history"),
            split_metadata.get("history"),
        )
    )

    expected_incomplete = (
        reference_result.get("status") == "incomplete"
        and split_first.get("status") == "incomplete"
        and split_second.get("status") == "incomplete"
    )
    expected_global_step = (
        int(reference_payload.get("global_step", -1)) == 10
        and int(split_payload.get("global_step", -1)) == 10
    )

    status = (
        "pass"
        if all(
            (
                generator_state_exact,
                discriminator_state_exact,
                generator_optimizer_exact,
                discriminator_optimizer_exact,
                torch_rng_exact,
                position_exact,
                waveform_exact,
                partial_epoch_state_exact,
                history_exact,
                expected_incomplete,
                expected_global_step,
            )
        )
        else "needs_review"
    )

    report = {
        "status": status,
        "device": "cpu",
        "smoke_version": SMOKE_VERSION,
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "reference_global_step": reference_payload.get("global_step"),
        "resumed_global_step": split_payload.get("global_step"),
        "resume_generator_state_exact": generator_state_exact,
        "resume_discriminator_state_exact": discriminator_state_exact,
        "resume_generator_optimizer_exact": generator_optimizer_exact,
        "resume_discriminator_optimizer_exact": discriminator_optimizer_exact,
        "resume_torch_rng_exact": torch_rng_exact,
        "resume_position_exact": position_exact,
        "resume_partial_epoch_state_exact": partial_epoch_state_exact,
        "resume_history_exact": history_exact,
        "resume_waveform_exact": waveform_exact,
        "resume_waveform_max_abs_delta": waveform_delta,
        "reference_stop_reason": reference_result.get("stop_reason"),
        "split_first_stop_reason": split_first.get("stop_reason"),
        "split_second_stop_reason": split_second.get("stop_reason"),
        "reference_last_checkpoint": str(reference_last),
        "resumed_last_checkpoint": str(split_last),
        "next_gate": (
            "run_bounded_resumable_v4_1_training"
            if status == "pass"
            else "fix_exact_resume_before_persistent_training"
        ),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_path = artifact_root / "resume_smoke_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_resume_smoke(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
