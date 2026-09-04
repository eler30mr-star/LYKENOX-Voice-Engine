"""Prepare the LYKENOX CLEAN_V1 workset without modifying source audio.

This script does not clean audio. It inventories the currently owned segmented speech corpus,
freezes source hashes, creates the CLEAN_V1 output directory and writes a portable work manifest for
an external offline cleaning tool. External tooling remains outside the LYKENOX implementation under
LYX-POL-001 v1.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.identity_voice_clean_v1 import (
    CLEAN_V1_STATE_SCHEMA,
    CLEAN_V1_VERSION,
    POLICY_ID,
    clean_v1_root,
    clean_v1_state_path,
    clean_v1_work_manifest_path,
    legacy_segmented_manifest_path,
    read_csv_rows,
    sha256_file,
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _portable(root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def prepare_clean_v1(root: Path, *, overwrite: bool = False) -> dict[str, object]:
    root = Path(root).resolve()
    out_root = clean_v1_root(root)
    wav_dir = out_root / "wav"
    manifests_dir = out_root / "manifests"
    out_root.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    work_path = clean_v1_work_manifest_path(root)
    state_path = clean_v1_state_path(root)
    if not overwrite and (work_path.exists() or state_path.exists()):
        raise FileExistsError(
            "CLEAN_V1 workset already exists. Use --overwrite only when deliberately rebuilding "
            "the pre-cleaning inventory; existing cleaned WAV files are never deleted by this script."
        )

    rows: list[dict[str, str]] = []
    source_manifest_hashes: dict[str, str] = {}
    source_ids: set[str] = set()
    for split in ("train", "val"):
        source_manifest = legacy_segmented_manifest_path(root, split)
        if not source_manifest.exists():
            raise FileNotFoundError(f"source segmented manifest missing: {source_manifest}")
        source_manifest_hashes[split] = sha256_file(source_manifest)
        for row in read_csv_rows(source_manifest):
            utterance_id = row["utterance_id"].strip()
            if not utterance_id:
                raise RuntimeError(f"blank utterance_id in {source_manifest}")
            if utterance_id in source_ids:
                raise RuntimeError(f"duplicate utterance_id across CLEAN_V1 source manifests: {utterance_id}")
            source_ids.add(utterance_id)
            source_wav = Path(row["wav_path"])
            if not source_wav.is_absolute():
                source_wav = (source_manifest.parent / source_wav).resolve()
            if not source_wav.exists():
                raise FileNotFoundError(f"source WAV missing for {utterance_id}: {source_wav}")
            clean_wav = wav_dir / f"{utterance_id}.wav"
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "split": split,
                    "source_wav_path": _portable(root, source_wav),
                    "source_sha256": sha256_file(source_wav),
                    "clean_wav_path": _portable(root, clean_wav),
                    "text": row["text"],
                    "preparation_status": "PENDING_EXTERNAL_CLEANING",
                    "notes": "",
                }
            )

    fieldnames = [
        "utterance_id",
        "split",
        "source_wav_path",
        "source_sha256",
        "clean_wav_path",
        "text",
        "preparation_status",
        "notes",
    ]
    with work_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    state: dict[str, object] = {
        "schema": CLEAN_V1_STATE_SCHEMA,
        "policy_id": POLICY_ID,
        "policy_version": "1.1",
        "dataset_version": CLEAN_V1_VERSION,
        "status": "ready_for_external_cleaning",
        "source_dataset": "prepared/speech_segmented",
        "source_audio_mutated": False,
        "raw_or_prepared_source_must_remain_immutable": True,
        "external_offline_tooling_permitted": True,
        "external_tool_integrated_into_lykenox": False,
        "external_model_or_checkpoint_imported_into_lykenox": False,
        "external_tool_required_for_product_inference": False,
        "items_total": len(rows),
        "items_train": sum(1 for row in rows if row["split"] == "train"),
        "items_val": sum(1 for row in rows if row["split"] == "val"),
        "source_manifest_sha256": source_manifest_hashes,
        "work_manifest": _portable(root, work_path),
        "clean_wav_directory": _portable(root, wav_dir),
        "technical_validation_passed": False,
        "human_auditory_review_complete": False,
        "all_acoustic_targets_and_caches_regenerated": False,
        "gold_oracles_rerun_after_clean_v1": False,
        "training_authorized": False,
        "next_action": "clean_or_reject_each_work_manifest_item_offline_then_run_validate_identity_voice_clean_v1.py",
    }
    _atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_clean_v1(args.root, overwrite=args.overwrite), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
