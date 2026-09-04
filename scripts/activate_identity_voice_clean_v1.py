"""Activate CLEAN_V1 only after technical validation and explicit human listening decisions.

Activation changes the canonical speech-manifest resolver to CLEAN_V1. It does NOT authorize new
vocoder training yet: mel/pitch/residual/cepstral targets and caches must be regenerated from the
clean WAV files and GOLD oracle controls must be rerun first. Policy: LYX-POL-001.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from lykenox_voice_engine.training.identity_voice_clean_v1 import (
    CLEAN_V1_STATE_SCHEMA,
    CLEAN_V1_VERSION,
    POLICY_ID,
    clean_v1_manifest_path,
    clean_v1_review_path,
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def activate_clean_v1(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    state = load_clean_v1_state(root)
    if state is None:
        raise RuntimeError("CLEAN_V1 has not been prepared")
    if state.get("status") == "active":
        raise RuntimeError("CLEAN_V1 is already active")

    technical_path = clean_v1_technical_report_path(root)
    review_path = clean_v1_review_path(root)
    work_path = clean_v1_work_manifest_path(root)
    for path in (technical_path, review_path, work_path):
        if not path.exists():
            raise FileNotFoundError(f"required CLEAN_V1 gate artifact missing: {path}")

    technical = json.loads(technical_path.read_text(encoding="utf-8"))
    if technical.get("policy_id") != POLICY_ID or technical.get("dataset_version") != CLEAN_V1_VERSION:
        raise RuntimeError("technical CLEAN_V1 report contract mismatch")
    if technical.get("technical_validation_passed") is not True:
        raise RuntimeError("CLEAN_V1 technical validation has not passed")

    work_rows = _read_csv(work_path)
    review_rows = _read_csv(review_path)
    work_by_id = {row["utterance_id"]: row for row in work_rows}
    review_by_id = {row["utterance_id"]: row for row in review_rows}
    if set(work_by_id) != set(review_by_id):
        missing = sorted(set(work_by_id) - set(review_by_id))
        extra = sorted(set(review_by_id) - set(work_by_id))
        raise RuntimeError(f"CLEAN_V1 review/work id mismatch: missing={missing[:10]} extra={extra[:10]}")

    technical_items = {str(item["utterance_id"]): item for item in technical.get("items", [])}
    if set(technical_items) != set(work_by_id):
        raise RuntimeError("CLEAN_V1 technical report does not cover exactly the work manifest")

    accepted: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    rejected: list[str] = []
    clean_hashes: dict[str, str] = {}

    for utterance_id, work in work_by_id.items():
        review = review_by_id[utterance_id]
        decision = str(review.get("auditory_decision", "")).strip().upper()
        if decision not in {"ACCEPT", "REJECT"}:
            raise RuntimeError(
                f"human auditory review incomplete for {utterance_id}: decision must be ACCEPT or REJECT"
            )
        tech = technical_items[utterance_id]
        tech_status = str(tech.get("technical_status", ""))
        source_path = Path(work["source_wav_path"])
        if not source_path.exists() or sha256_file(source_path) != work["source_sha256"]:
            raise RuntimeError(f"source immutability check failed at activation for {utterance_id}")

        if decision == "REJECT":
            rejected.append(utterance_id)
            continue
        if tech_status != "PASS":
            raise RuntimeError(f"cannot ACCEPT technically invalid CLEAN_V1 item {utterance_id}: {tech_status}")

        clean_path = Path(work["clean_wav_path"])
        if not clean_path.exists():
            raise FileNotFoundError(f"accepted CLEAN_V1 WAV missing: {clean_path}")
        current_clean_hash = sha256_file(clean_path)
        expected_clean_hash = tech.get("clean_sha256")
        if current_clean_hash != expected_clean_hash:
            raise RuntimeError(
                f"clean WAV changed after technical validation for {utterance_id}: rerun validation/listening"
            )
        clean_hashes[utterance_id] = current_clean_hash
        split = work["split"]
        if split not in accepted:
            raise RuntimeError(f"unsupported split in CLEAN_V1: {split}")
        accepted[split].append(
            {
                "utterance_id": utterance_id,
                "wav_path": f"../wav/{clean_path.name}",
                "text": work["text"],
            }
        )

    if not accepted["train"]:
        raise RuntimeError("CLEAN_V1 activation requires at least one accepted train utterance")
    if not accepted["val"]:
        raise RuntimeError("CLEAN_V1 activation requires at least one accepted val utterance")

    manifest_hashes: dict[str, str] = {}
    for split in ("train", "val"):
        path = clean_v1_manifest_path(root, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["utterance_id", "wav_path", "text"])
            writer.writeheader()
            writer.writerows(accepted[split])
        manifest_hashes[split] = sha256_file(path)

    activation_report: dict[str, object] = {
        "status": "active",
        "schema": CLEAN_V1_STATE_SCHEMA,
        "policy_id": POLICY_ID,
        "policy_version": "1.1",
        "dataset_version": CLEAN_V1_VERSION,
        "technical_validation_passed": True,
        "human_auditory_review_complete": True,
        "human_auditory_quality_is_authority": True,
        "accepted_train": len(accepted["train"]),
        "accepted_val": len(accepted["val"]),
        "rejected_total": len(rejected),
        "rejected_utterance_ids": rejected,
        "manifest_sha256": manifest_hashes,
        "accepted_clean_wav_sha256": clean_hashes,
        "external_tool": technical.get("external_tool"),
        "source_audio_mutated": False,
        "external_tool_integrated_into_lykenox": False,
        "all_acoustic_targets_and_caches_regenerated": False,
        "gold_oracles_rerun_after_clean_v1": False,
        "training_authorized": False,
        "next_action": "regenerate CLEAN_V1 mel/F0/periodicity/cepstrum/residual targets/caches then rerun GOLD oracles",
    }
    _atomic_json(clean_v1_state_path(root), activation_report)
    return activation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(activate_clean_v1(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
