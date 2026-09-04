"""Prepare a deterministic 10-take RECORDING_V2 capture pilot.

The pilot is a capture-quality gate only. It selects a small, reproducible subset of the already
prepared RECORDING_V2 session, including the historical 0021/0022 audit prompts, without touching
or processing audio. Human listening remains the final authority before recording all 132 takes.
Policy: LYX-POL-001 v1.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.identity_voice_recording_v2 import (
    POLICY_ID,
    RECORDING_V2_VERSION,
    recording_v2_session_manifest,
)

PILOT_VERSION = "lykenox-identity-voice-recording-v2-pilot-v1"
DEFAULT_PILOT_ITEMS = 10
TARGET_TRAIN_ITEMS = 6
TARGET_VAL_ITEMS = 4
REQUIRED_SOURCE_PROMPT_IDS = (
    "speech_0021_6cd35984e877_seg_001",
    "speech_0022_ba721f6129b9_seg_005",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty RECORDING_V2 session manifest: {path}")
    return rows


def _even_length_sample(
    rows: list[dict[str, str]],
    count: int,
    *,
    excluded_ids: set[str],
) -> list[dict[str, str]]:
    candidates = [row for row in rows if row["recording_id"] not in excluded_ids]
    candidates.sort(key=lambda row: (len(row["text"]), row["recording_id"]))
    if count < 1:
        return []
    if len(candidates) < count:
        raise RuntimeError(f"not enough pilot candidates: need {count}, have {len(candidates)}")
    if count == 1:
        return [candidates[len(candidates) // 2]]

    selected: list[dict[str, str]] = []
    used: set[str] = set()
    for position in range(count):
        target = position * (len(candidates) - 1) / float(count - 1)
        index = int(round(target))
        # Resolve rare rounding collisions deterministically by nearest unused index.
        if candidates[index]["recording_id"] in used:
            offsets = list(range(1, len(candidates)))
            found = None
            for offset in offsets:
                for candidate_index in (index - offset, index + offset):
                    if 0 <= candidate_index < len(candidates):
                        candidate = candidates[candidate_index]
                        if candidate["recording_id"] not in used:
                            found = candidate_index
                            break
                if found is not None:
                    break
            if found is None:
                raise RuntimeError("unable to resolve deterministic pilot selection collision")
            index = found
        row = candidates[index]
        selected.append(row)
        used.add(row["recording_id"])
    return selected


def prepare_recording_v2_pilot(root: Path, *, items: int = DEFAULT_PILOT_ITEMS) -> dict[str, object]:
    root = Path(root).resolve()
    if items != DEFAULT_PILOT_ITEMS:
        raise ValueError(f"RECORDING_V2 v1 pilot is fixed at {DEFAULT_PILOT_ITEMS} items")

    session_path = recording_v2_session_manifest(root)
    if not session_path.exists():
        raise FileNotFoundError(
            f"RECORDING_V2 session manifest missing: {session_path}. Run prepare_identity_voice_recording_v2_session.py first."
        )
    rows = _read_rows(session_path)
    if len(rows) != 132:
        raise RuntimeError(f"RECORDING_V2 session count changed: expected 132, got {len(rows)}")

    by_source = {row["source_prompt_id"]: row for row in rows}
    required: list[dict[str, str]] = []
    for source_id in REQUIRED_SOURCE_PROMPT_IDS:
        row = by_source.get(source_id)
        if row is None:
            raise RuntimeError(f"required pilot source prompt missing: {source_id}")
        required.append(row)

    selected_ids = {row["recording_id"] for row in required}
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    required_train = sum(row["split"] == "train" for row in required)
    required_val = sum(row["split"] == "val" for row in required)

    train_needed = TARGET_TRAIN_ITEMS - required_train
    val_needed = TARGET_VAL_ITEMS - required_val
    if train_needed < 0 or val_needed < 0:
        raise RuntimeError("required pilot anchors exceed split targets")

    selected = list(required)
    train_extra = _even_length_sample(train_rows, train_needed, excluded_ids=selected_ids)
    selected.extend(train_extra)
    selected_ids.update(row["recording_id"] for row in train_extra)
    val_extra = _even_length_sample(val_rows, val_needed, excluded_ids=selected_ids)
    selected.extend(val_extra)

    if len(selected) != items:
        raise RuntimeError(f"pilot selection produced {len(selected)} items instead of {items}")

    # Audition order: required anchors first, then deterministic train/val coverage.
    metadata_dir = session_path.parent
    pilot_manifest = metadata_dir / "pilot_manifest.csv"
    with pilot_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pilot_order",
                "recording_id",
                "source_prompt_id",
                "split",
                "text",
                "raw_wav_path",
                "capture_status",
                "technical_status",
                "auditory_status",
                "notes",
            ],
        )
        writer.writeheader()
        for order, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "pilot_order": order,
                    "recording_id": row["recording_id"],
                    "source_prompt_id": row["source_prompt_id"],
                    "split": row["split"],
                    "text": row["text"],
                    "raw_wav_path": row["raw_wav_path"],
                    "capture_status": "PENDING",
                    "technical_status": "PENDING",
                    "auditory_status": "PENDING",
                    "notes": "",
                }
            )

    report = {
        "status": "ready_for_recording_v2_pilot_capture",
        "schema": PILOT_VERSION,
        "policy_id": POLICY_ID,
        "dataset_version": RECORDING_V2_VERSION,
        "items_total": len(selected),
        "items_train": sum(row["split"] == "train" for row in selected),
        "items_val": sum(row["split"] == "val" for row in selected),
        "required_source_prompt_ids": list(REQUIRED_SOURCE_PROMPT_IDS),
        "selection_strategy": "required_0021_0022_plus_even_text_length_coverage_by_split",
        "selection_metric_is_acceptance_evidence": False,
        "audio_processed": False,
        "training_authorized": False,
        "pilot_manifest": str(pilot_manifest),
        "next_action": "record exactly these 10 RAW WAVs with final microphone settings, then run validate_identity_voice_recording_v2_pilot.py",
    }
    (metadata_dir / "pilot_state.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--items", type=int, default=DEFAULT_PILOT_ITEMS)
    args = parser.parse_args()
    print(json.dumps(prepare_recording_v2_pilot(args.root, items=args.items), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
