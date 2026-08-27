"""Recover and audit an interrupted persistent LYKENOX aligner run.

This command never trains. It inspects the already-written best checkpoint,
reconstructs the missing validation report, audits monotonic alignments on the
held-out split, and only then generates production duration caches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import (
    LykenoxCTCAligner,
    LykenoxCTCAlignerConfig,
    LykenoxSpeechConfig,
)
from lykenox_voice_engine.training.alignment_artifact import load_aligner_checkpoint
from lykenox_voice_engine.training.speech_aligner_train import (
    _audit_forced_alignments,
    _dataset,
    _eligible_indices,
    _mean_ctc_loss,
)
from lykenox_voice_engine.training.speech_duration_cache import generate_duration_cache


def audit_existing_checkpoint(
    root: Path,
    *,
    max_mel_frames: int = 1800,
    seed: int = 1337,
) -> dict[str, object]:
    """Reconstruct the interrupted run's missing validation gate from best.pt."""

    root = Path(root).resolve()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    frontend = SpanishTextFrontend()
    speech_config = LykenoxSpeechConfig()
    expected_config = LykenoxCTCAlignerConfig(
        num_symbols=frontend.vocab_size,
        mel_bins=speech_config.mel_bins,
    )
    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "speech_aligner"
        / frontend.version
    )
    best_path = artifact_dir / "best.pt"
    report_path = artifact_dir / "recovery_validation_report.json"

    if not best_path.exists():
        report = {
            "status": "no_checkpoint",
            "best_checkpoint": str(best_path),
            "next_gate": "train_or_resume_aligner",
        }
        artifact_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    val_dataset = _dataset(root, "val", speech_config)
    val_indices, val_excluded = _eligible_indices(
        val_dataset,
        expected_config,
        max_mel_frames,
    )
    if not val_indices:
        raise RuntimeError("No eligible validation utterances for checkpoint recovery audit")

    criterion = torch.nn.CTCLoss(blank=expected_config.blank_id, zero_infinity=True)

    # Reconstruct the deterministic random-init baseline used by the original run.
    torch.manual_seed(seed)
    baseline_model = LykenoxCTCAligner(expected_config).cpu().train()
    initial_val = _mean_ctc_loss(baseline_model, val_dataset, val_indices, criterion)

    best_model, payload = load_aligner_checkpoint(best_path)
    actual_val = _mean_ctc_loss(best_model, val_dataset, val_indices, criterion)
    forced_audit = _audit_forced_alignments(best_model, val_dataset, val_indices)

    stored_val = payload.get("validation_ctc_loss")
    stored_val_float = float(stored_val) if stored_val is not None else None
    stored_metric_delta = (
        abs(actual_val - stored_val_float) if stored_val_float is not None else None
    )
    validation_loss_decreased = actual_val < initial_val
    status = (
        "pass"
        if validation_loss_decreased and bool(forced_audit["pass"])
        else "needs_review"
    )

    report = {
        "status": status,
        "mode": "recover_existing_best_checkpoint",
        "device": "cpu",
        "frontend_version": frontend.version,
        "parameters": best_model.parameter_count(),
        "best_checkpoint": str(best_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "stored_validation_ctc_loss": (
            round(stored_val_float, 6) if stored_val_float is not None else None
        ),
        "recomputed_validation_ctc_loss": round(actual_val, 6),
        "stored_metric_delta": (
            round(stored_metric_delta, 8) if stored_metric_delta is not None else None
        ),
        "initial_random_validation_ctc_loss": round(initial_val, 6),
        "validation_loss_decreased": validation_loss_decreased,
        "val_items_total": len(val_dataset),
        "val_items_eligible": len(val_indices),
        "val_excluded": val_excluded,
        "validation_alignment_audit": forced_audit,
        "next_gate": (
            "generate_and_audit_duration_cache"
            if status == "pass"
            else "resume_aligner_training"
        ),
        "note": (
            "This recovery audit performs no training. It salvages the best checkpoint "
            "written before the interrupted process and reconstructs the missing gate."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def recover_pipeline(
    root: Path,
    *,
    max_mel_frames: int = 1800,
    seed: int = 1337,
    nonpause_warn_frames: int = 100,
) -> dict[str, object]:
    """Audit the interrupted best checkpoint and, if valid, build duration caches."""

    root = Path(root).resolve()
    recovery = audit_existing_checkpoint(
        root,
        max_mel_frames=max_mel_frames,
        seed=seed,
    )
    if recovery["status"] != "pass":
        return {
            "status": "checkpoint_gate_failed",
            "recovery": recovery,
            "durations": None,
            "next_gate": "resume_aligner_training",
        }

    durations = generate_duration_cache(
        root,
        Path(str(recovery["best_checkpoint"])),
        nonpause_warn_frames=nonpause_warn_frames,
    )
    suspicious = int(durations.get("suspicious_utterance_count", 0))
    if durations["status"] != "pass":
        status = "duration_gate_failed"
        next_gate = "review_failed_alignments"
    elif suspicious > 0:
        status = "duration_review_required"
        next_gate = "review_duration_outliers"
    else:
        status = "pass"
        next_gate = "aligned_acoustic_smoke"

    result = {
        "status": status,
        "recovery": recovery,
        "durations": durations,
        "next_gate": next_gate,
    }
    report_dir = Path(str(recovery["best_checkpoint"])).parent
    (report_dir / "recovery_pipeline_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-mel-frames", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--nonpause-warn-frames", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            recover_pipeline(
                args.root,
                max_mel_frames=args.max_mel_frames,
                seed=args.seed,
                nonpause_warn_frames=args.nonpause_warn_frames,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
