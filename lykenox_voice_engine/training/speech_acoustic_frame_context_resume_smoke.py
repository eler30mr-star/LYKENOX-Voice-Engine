"""Exact split/resume equivalence gate for persistent LYKENOX acoustic frame-context v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.speech.config import FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    FRAME_CONTEXT_VERSION,
    TRAINER_CONTRACT_VERSION,
    run_acoustic_frame_context_training,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
)


SMOKE_VERSION = "acoustic-frame-context-exact-resume-smoke-v2"


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _max_model_delta(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    maximum = 0.0
    for key in left:
        if key not in right or left[key].shape != right[key].shape:
            return float("inf")
        if left[key].numel():
            maximum = max(maximum, float((left[key] - right[key]).abs().max()))
    return maximum


def run_resume_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    smoke_root = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_frame_context_v2_resume_smoke"
    )
    reference_dir = smoke_root / "reference_uninterrupted"
    split_dir = smoke_root / "split_resume"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    reference_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "batch_size": 2,
        "max_epochs": 2,
        "patience": 2,
        "seed": 1337,
        "learning_rate": 2e-4,
        "duration_weight": 0.10,
        "f0_weight": 0.25,
        "voicing_weight": 0.25,
        "checkpoint_every_updates": 2,
        "time_budget_seconds": 30.0,
        "checkpoint_reserve_seconds": 8.0,
    }

    reference_result = run_acoustic_frame_context_training(
        root,
        output_dir=reference_dir,
        max_updates_this_run=10,
        **common,
    )
    split_first = run_acoustic_frame_context_training(
        root,
        output_dir=split_dir,
        max_updates_this_run=3,
        **common,
    )
    split_second = run_acoustic_frame_context_training(
        root,
        output_dir=split_dir,
        max_updates_this_run=7,
        **common,
    )

    reference_path = reference_dir / "last.pt"
    resumed_path = split_dir / "last.pt"
    reference_payload = torch.load(reference_path, map_location="cpu", weights_only=False)
    resumed_payload = torch.load(resumed_path, map_location="cpu", weights_only=False)

    reference_state = reference_payload["model_state"]
    resumed_state = resumed_payload["model_state"]
    model_exact = _nested_equal(reference_state, resumed_state)
    optimizer_exact = _nested_equal(
        reference_payload["optimizer_state"], resumed_payload["optimizer_state"]
    )
    rng_exact = _nested_equal(
        reference_payload["torch_rng_state"], resumed_payload["torch_rng_state"]
    )
    position_exact = (
        int(reference_payload["epoch"]) == int(resumed_payload["epoch"])
        and int(reference_payload["next_item_offset"])
        == int(resumed_payload["next_item_offset"])
        and int(reference_payload["global_step"]) == int(resumed_payload["global_step"])
    )
    metadata_exact = _nested_equal(
        reference_payload["training_metadata"], resumed_payload["training_metadata"]
    )
    run_config_exact = reference_payload["run_config"] == resumed_payload["run_config"]
    provenance_exact = (
        reference_payload["training_provenance"] == resumed_payload["training_provenance"]
    )
    frame_context_exact = (
        reference_payload["model_config"].get("frame_context_version") == FRAME_CONTEXT_VERSION
        and resumed_payload["model_config"].get("frame_context_version") == FRAME_CONTEXT_VERSION
        and FRAME_CONTEXT_VERSION == FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
    )

    reference_model, _ = load_acoustic_prosody_checkpoint(reference_path)
    resumed_model, _ = load_acoustic_prosody_checkpoint(resumed_path)
    frontend = SpanishTextFrontend()
    config = LykenoxSpeechConfig(
        vocab_size=frontend.vocab_size,
        frame_context_version=FRAME_CONTEXT_VERSION,
    )
    dataset = LykenoxAlignedSpeechDataset(root, "val", config, include_pitch_targets=True)
    probe = collate_aligned_speech([dataset[0], dataset[1]]).to("cpu")
    reference_model.eval()
    resumed_model.eval()
    with torch.no_grad():
        reference_output = reference_model(probe.token_ids, probe.token_mask, probe.durations)
        resumed_output = resumed_model(probe.token_ids, probe.token_mask, probe.durations)
    output_keys = (
        "mel",
        "f0_prediction_hz",
        "f0_log_prediction",
        "voicing_logits",
        "duration_prediction",
        "mel_mask",
        "mel_lengths",
    )
    output_exact = all(
        torch.equal(reference_output[key], resumed_output[key]) for key in output_keys
    )
    max_output_delta = max(
        float((reference_output[key] - resumed_output[key]).abs().max())
        for key in (
            "mel",
            "f0_prediction_hz",
            "f0_log_prediction",
            "voicing_logits",
            "duration_prediction",
        )
    )

    checks = {
        "model_state_exact": model_exact,
        "optimizer_state_exact": optimizer_exact,
        "torch_rng_state_exact": rng_exact,
        "position_exact": position_exact,
        "training_metadata_exact": metadata_exact,
        "run_config_exact": run_config_exact,
        "provenance_exact": provenance_exact,
        "frame_context_identity_exact": frame_context_exact,
        "probe_output_exact": output_exact,
    }
    status = "pass" if all(checks.values()) else "needs_review"
    report = {
        "status": status,
        "device": "cpu",
        "smoke_version": SMOKE_VERSION,
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "frame_context_version": FRAME_CONTEXT_VERSION,
        "reference_first_result_status": reference_result.get("status"),
        "split_first_result_status": split_first.get("status"),
        "split_second_result_status": split_second.get("status"),
        "reference_global_step": int(reference_payload["global_step"]),
        "resumed_global_step": int(resumed_payload["global_step"]),
        **checks,
        "max_model_parameter_delta": _max_model_delta(reference_state, resumed_state),
        "max_probe_output_delta": max_output_delta,
        "reference_checkpoint": str(reference_path),
        "resumed_checkpoint": str(resumed_path),
        "next_gate": (
            "start_bounded_resumable_acoustic_frame_context_v2_training"
            if status == "pass"
            else "fix_acoustic_frame_context_v2_resume_contract"
        ),
    }
    report_path = smoke_root / "resume_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_resume_smoke(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
