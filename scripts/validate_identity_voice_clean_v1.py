"""Validate externally prepared CLEAN_V1 audio without perceptually accepting it by metrics.

The script verifies source immutability, output presence, basic audio geometry and signal integrity,
records external-tool provenance, and creates a human listening review queue. Technical checks can
reject a file, but cannot mark perceptual quality as accepted. Policy: LYX-POL-001 v1.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from lykenox_voice_engine.training.identity_voice_clean_v1 import (
    CLEAN_V1_STATE_SCHEMA,
    CLEAN_V1_VERSION,
    POLICY_ID,
    clean_v1_review_path,
    clean_v1_root,
    clean_v1_state_path,
    clean_v1_technical_report_path,
    clean_v1_work_manifest_path,
    load_clean_v1_state,
    sha256_file,
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _rms(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1.0e-12))


def _load_previous_review(path: Path) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    result: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[str(row["utterance_id"])] = (
                str(row.get("auditory_decision", "PENDING")).strip().upper() or "PENDING",
                str(row.get("auditory_notes", "")),
            )
    return result


def validate_clean_v1(
    root: Path,
    *,
    tool_name: str,
    tool_version: str,
    tool_terms_note: str,
    rejected_ids: set[str] | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    rejected_ids = set(rejected_ids or ())
    if not tool_name.strip():
        raise ValueError("--tool-name is required for CLEAN_V1 provenance")
    if not tool_version.strip():
        raise ValueError("--tool-version is required for CLEAN_V1 provenance")
    if not tool_terms_note.strip():
        raise ValueError("--tool-terms-note is required; record why cleaned outputs may be used by LYKENOX")

    state = load_clean_v1_state(root)
    if state is None:
        raise RuntimeError("CLEAN_V1 has not been prepared")
    if state.get("status") == "active":
        raise RuntimeError("CLEAN_V1 is already active; rebuild deliberately before replacing it")

    work_path = clean_v1_work_manifest_path(root)
    if not work_path.exists():
        raise FileNotFoundError(f"CLEAN_V1 work manifest missing: {work_path}")
    with work_path.open("r", encoding="utf-8", newline="") as handle:
        work_rows = list(csv.DictReader(handle))
    if not work_rows:
        raise RuntimeError("CLEAN_V1 work manifest is empty")

    known_ids = {str(row["utterance_id"]) for row in work_rows}
    unknown_rejections = rejected_ids - known_ids
    if unknown_rejections:
        raise ValueError("unknown --reject utterance ids: " + ", ".join(sorted(unknown_rejections)))

    previous_review = _load_previous_review(clean_v1_review_path(root))
    items: list[dict[str, object]] = []
    review_rows: list[dict[str, str]] = []
    technical_failures = 0
    clean_candidates = 0

    for row in work_rows:
        utterance_id = str(row["utterance_id"])
        split = str(row["split"])
        source_path = Path(str(row["source_wav_path"]))
        clean_path = Path(str(row["clean_wav_path"]))
        expected_source_hash = str(row["source_sha256"])

        if not source_path.exists():
            raise FileNotFoundError(f"source WAV disappeared: {source_path}")
        actual_source_hash = sha256_file(source_path)
        if actual_source_hash != expected_source_hash:
            raise RuntimeError(
                f"source immutability violation for {utterance_id}: expected {expected_source_hash}, got {actual_source_hash}"
            )

        if utterance_id in rejected_ids:
            items.append(
                {
                    "utterance_id": utterance_id,
                    "split": split,
                    "technical_status": "REJECTED_BY_CURATOR",
                    "source_sha256": actual_source_hash,
                    "clean_wav_path": str(clean_path),
                }
            )
            review_rows.append(
                {
                    "utterance_id": utterance_id,
                    "split": split,
                    "clean_wav_path": str(clean_path),
                    "technical_status": "REJECTED_BY_CURATOR",
                    "auditory_decision": "REJECT",
                    "auditory_notes": "Rejected before CLEAN_V1 activation",
                }
            )
            continue

        clean_candidates += 1
        failures: list[str] = []
        warnings: list[str] = []
        if not clean_path.exists():
            failures.append("clean_wav_missing")
            clean_info = None
            clean_audio = None
        else:
            clean_info = sf.info(str(clean_path))
            clean_audio, clean_rate = sf.read(str(clean_path), dtype="float32", always_2d=True)
            if int(clean_rate) != int(clean_info.samplerate):
                failures.append("clean_samplerate_read_mismatch")
            if clean_audio.size == 0:
                failures.append("clean_audio_empty")
            elif not bool(np.isfinite(clean_audio).all()):
                failures.append("clean_audio_nonfinite")

        source_info = sf.info(str(source_path))
        source_duration = float(source_info.frames) / float(source_info.samplerate)
        source_peak = None
        clean_duration = None
        clean_peak = None
        clean_rms = None

        if clean_info is not None and clean_audio is not None and clean_audio.size:
            clean_duration = float(clean_info.frames) / float(clean_info.samplerate)
            clean_peak = float(np.max(np.abs(clean_audio)))
            clean_rms = _rms(clean_audio)
            if int(clean_info.samplerate) != int(source_info.samplerate):
                failures.append("sample_rate_changed")
            if int(clean_info.channels) != int(source_info.channels):
                failures.append("channel_count_changed")
            if clean_peak > 1.0001:
                failures.append("clean_audio_peak_above_full_scale")
            if clean_rms < 1.0e-5:
                failures.append("clean_audio_effectively_silent")
            duration_ratio = clean_duration / max(source_duration, 1.0e-9)
            if duration_ratio < 0.50 or duration_ratio > 1.10:
                failures.append("duration_change_outside_safety_bound")
            elif abs(duration_ratio - 1.0) > 0.05:
                warnings.append("duration_changed_more_than_5_percent_review_alignment")
        else:
            duration_ratio = None

        status = "PASS" if not failures else "FAIL"
        if failures:
            technical_failures += 1
        else:
            previous_decision, previous_notes = previous_review.get(utterance_id, ("PENDING", ""))
            if previous_decision not in {"ACCEPT", "REJECT", "PENDING"}:
                previous_decision = "PENDING"
            review_rows.append(
                {
                    "utterance_id": utterance_id,
                    "split": split,
                    "clean_wav_path": str(clean_path),
                    "technical_status": status,
                    "auditory_decision": previous_decision,
                    "auditory_notes": previous_notes,
                }
            )

        items.append(
            {
                "utterance_id": utterance_id,
                "split": split,
                "technical_status": status,
                "failures": failures,
                "warnings": warnings,
                "source_wav_path": str(source_path),
                "clean_wav_path": str(clean_path),
                "source_sha256": actual_source_hash,
                "clean_sha256": sha256_file(clean_path) if clean_path.exists() else None,
                "source_sample_rate": int(source_info.samplerate),
                "clean_sample_rate": int(clean_info.samplerate) if clean_info is not None else None,
                "source_channels": int(source_info.channels),
                "clean_channels": int(clean_info.channels) if clean_info is not None else None,
                "source_duration_seconds": source_duration,
                "clean_duration_seconds": clean_duration,
                "duration_ratio": duration_ratio,
                "clean_peak": clean_peak,
                "clean_rms_dbfs": _dbfs(clean_rms) if clean_rms is not None else None,
            }
        )

    if clean_candidates < 1:
        raise RuntimeError("all CLEAN_V1 items were rejected; at least one clean candidate is required")

    review_path = clean_v1_review_path(root)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utterance_id",
                "split",
                "clean_wav_path",
                "technical_status",
                "auditory_decision",
                "auditory_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    technical_passed = technical_failures == 0
    report: dict[str, object] = {
        "status": "ready_for_auditory_review" if technical_passed else "technical_validation_failed",
        "schema": "lykenox-clean-v1-technical-validation-v1",
        "policy_id": POLICY_ID,
        "policy_version": "1.1",
        "dataset_version": CLEAN_V1_VERSION,
        "metrics_can_accept_perceptual_quality": False,
        "source_audio_mutated": False,
        "external_tool": {
            "name": tool_name.strip(),
            "version": tool_version.strip(),
            "terms_or_license_note": tool_terms_note.strip(),
            "integrated_into_lykenox": False,
            "required_for_lykenox_inference": False,
            "weights_imported_or_distilled_into_lykenox": False,
        },
        "items_total": len(work_rows),
        "clean_candidates": clean_candidates,
        "curator_rejected": len(rejected_ids),
        "technical_failures": technical_failures,
        "technical_validation_passed": technical_passed,
        "human_auditory_review_complete": False,
        "listening_review_csv": str(review_path),
        "items": items,
        "next_action": (
            "complete listening_review.csv with ACCEPT/REJECT then run activate_identity_voice_clean_v1.py"
            if technical_passed
            else "fix or reject failed cleaned items and rerun technical validation"
        ),
    }
    _atomic_json(clean_v1_technical_report_path(root), report)

    new_state = dict(state)
    new_state.update(
        {
            "schema": CLEAN_V1_STATE_SCHEMA,
            "policy_id": POLICY_ID,
            "dataset_version": CLEAN_V1_VERSION,
            "status": "ready_for_auditory_review" if technical_passed else "technical_validation_failed",
            "technical_validation_passed": technical_passed,
            "human_auditory_review_complete": False,
            "training_authorized": False,
            "external_tool": report["external_tool"],
            "technical_validation_report": str(clean_v1_technical_report_path(root)),
            "listening_review_csv": str(review_path),
            "next_action": report["next_action"],
        }
    )
    _atomic_json(clean_v1_state_path(root), new_state)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tool-name", required=True)
    parser.add_argument("--tool-version", required=True)
    parser.add_argument("--tool-terms-note", required=True)
    parser.add_argument("--reject", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            validate_clean_v1(
                args.root,
                tool_name=args.tool_name,
                tool_version=args.tool_version,
                tool_terms_note=args.tool_terms_note,
                rejected_ids=set(args.reject),
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
