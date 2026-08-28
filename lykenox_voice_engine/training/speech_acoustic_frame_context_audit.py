"""Held-out audit for the persistent LYKENOX acoustic frame-context v2 checkpoint.

This gate reuses the established teacher-duration acoustic/prosody audit so v1 and v2 are
measured on the same validation contract, then adds strict v2 architecture/trainer identity
checks and an optional side-by-side comparison against the rejected v1 held-out report.

Teacher durations are intentional here: predicted-duration semantics are still a separate
inference gate. Passing this audit means the persistent v2 checkpoint preserved exact frame
contracts and non-zero intra-token mel/F0 motion on held-out data. It does not yet authorize
unseen-text product synthesis.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lykenox_voice_engine.models.speech.config import FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_audit import (
    run_acoustic_prosody_audit,
)
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    TRAINER_CONTRACT_VERSION,
)


AUDIT_VERSION = "acoustic-frame-context-heldout-audit-v2"
FRAME_CONTEXT_VERSION = FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _numeric(report: dict[str, Any], key: str) -> float | None:
    value = report.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compare_metric(
    current: dict[str, Any],
    previous: dict[str, Any],
    key: str,
    *,
    higher_is_better: bool,
) -> dict[str, object] | None:
    current_value = _numeric(current, key)
    previous_value = _numeric(previous, key)
    if current_value is None or previous_value is None:
        return None
    delta = current_value - previous_value
    improved = delta > 0.0 if higher_is_better else delta < 0.0
    return {
        "v1": previous_value,
        "v2": current_value,
        "delta_v2_minus_v1": delta,
        "higher_is_better": higher_is_better,
        "improved": improved,
    }


def _v1_comparison(root: Path, current: dict[str, Any]) -> dict[str, object] | None:
    v1_path = (
        Path(root)
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_prosody_v1"
        / "heldout_audit.json"
    )
    previous = _load_json(v1_path)
    if previous is None:
        return None

    directions = {
        "mel_l1": False,
        "f0_log_pearson": True,
        "f0_mae_cents": False,
        "f0_rmse_cents": False,
        "f0_median_abs_cents": False,
        "voicing_f1": True,
        "voicing_balanced_accuracy": True,
        "intra_token_mel_delta_l1_predicted": True,
        "intra_token_f0_delta_cents_predicted": True,
    }
    metrics: dict[str, object] = {}
    for key, higher_is_better in directions.items():
        comparison = _compare_metric(
            current,
            previous,
            key,
            higher_is_better=higher_is_better,
        )
        if comparison is not None:
            metrics[key] = comparison

    return {
        "v1_report": str(v1_path),
        "v1_status": previous.get("status"),
        "metrics": metrics,
        "note": (
            "This comparison is diagnostic only. v2 acceptance is gated by exact frame "
            "contracts, persistent frame expressivity, and exact v2 architecture identity; "
            "not every scalar metric is required to improve monotonically."
        ),
    }


def run_acoustic_frame_context_audit(
    root: Path,
    *,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_frame_context_v2"
        / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Persistent acoustic v2 best checkpoint not found: {checkpoint}")

    model, payload = load_acoustic_prosody_checkpoint(checkpoint)
    run_config = payload.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("Persistent acoustic v2 checkpoint is missing run_config")

    architecture_identity_exact = (
        model.config.frame_context_version == FRAME_CONTEXT_VERSION
        and run_config.get("trainer_contract_version") == TRAINER_CONTRACT_VERSION
        and run_config.get("frame_context_version") == FRAME_CONTEXT_VERSION
        and int(run_config.get("frame_context_layers", -1))
        == int(model.config.frame_context_layers)
        and int(run_config.get("frame_context_kernel_size", -1))
        == int(model.config.frame_context_kernel_size)
    )
    if not architecture_identity_exact:
        raise RuntimeError(
            "Refusing held-out v2 audit: checkpoint frame-context/trainer identity mismatch"
        )

    base = run_acoustic_prosody_audit(root, checkpoint=checkpoint)
    contracts_pass = bool(base.get("all_frame_contracts_exact", False))
    expressivity_pass = bool(base.get("frame_expressivity_pass", False))
    status = (
        "pass"
        if architecture_identity_exact and contracts_pass and expressivity_pass
        else "needs_review"
    )

    comparison = _v1_comparison(root, base)
    report: dict[str, object] = {
        **base,
        "status": status,
        "audit_version": AUDIT_VERSION,
        "base_audit_version": base.get("audit_version"),
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "frame_context_version": model.config.frame_context_version,
        "frame_context_layers": int(model.config.frame_context_layers),
        "frame_context_kernel_size": int(model.config.frame_context_kernel_size),
        "architecture_identity_exact": architecture_identity_exact,
        "v1_comparison": comparison,
        "next_gate": (
            "fix_predicted_duration_semantics_before_end_to_end"
            if status == "pass"
            else "review_acoustic_frame_context_v2_before_duration_fix"
        ),
        "interpretation": (
            "Teacher durations are intentionally used to isolate the persistent v2 "
            "acoustic/prosody checkpoint. A pass confirms exact held-out frame contracts "
            "and that the post-regulation frame-context fix survives persistent training. "
            "Predicted-duration semantics remain unresolved and must be fixed before any "
            "unseen-text end-to-end synthesis."
        ),
    }

    report_path = checkpoint.parent / "heldout_audit_v2.json"
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_acoustic_frame_context_audit(args.root, checkpoint=args.checkpoint),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
