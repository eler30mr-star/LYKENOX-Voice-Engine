"""Prepare the clean RECORDING_V2 capture session without touching any audio.

The session reuses the existing 132 segmented prompt texts and train/validation split assignment,
but creates new recording IDs and RAW target paths. No training, denoise, feature extraction or audio
mutation occurs here. Policy: LYX-POL-001 v1.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.identity_voice_recording_v2 import (
    POLICY_ID,
    RECORDING_V2_VERSION,
    recording_v2_accepted_dir,
    recording_v2_raw_dir,
    recording_v2_root,
    recording_v2_session_manifest,
)


def _source_manifest(root: Path, split: str) -> Path:
    path = (
        root
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech_segmented"
        / f"{split}.segmented.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"segmented source manifest missing: {path}")
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty source manifest: {path}")
    return rows


def _rec2_id(old_id: str) -> str:
    return f"{old_id}_rec2"


def prepare_recording_v2_session(root: Path, *, overwrite: bool = False) -> dict[str, object]:
    root = Path(root).resolve()
    out_root = recording_v2_root(root)
    raw_dir = recording_v2_raw_dir(root)
    accepted_dir = recording_v2_accepted_dir(root)
    manifest_path = recording_v2_session_manifest(root)

    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"RECORDING_V2 session already exists: {manifest_path}. Use --overwrite only to regenerate metadata."
        )

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for split in ("train", "val"):
        for source_row in _read_rows(_source_manifest(root, split)):
            source_id = source_row["utterance_id"].strip()
            rec_id = _rec2_id(source_id)
            if rec_id in seen_ids:
                raise RuntimeError(f"duplicate RECORDING_V2 id: {rec_id}")
            seen_ids.add(rec_id)
            rows.append(
                {
                    "recording_id": rec_id,
                    "source_prompt_id": source_id,
                    "split": split,
                    "text": source_row["text"],
                    "raw_wav_path": str(Path("..") / "raw" / f"{rec_id}.wav"),
                    "capture_status": "PENDING",
                    "auditory_status": "PENDING",
                    "notes": "",
                }
            )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "recording_id",
                "source_prompt_id",
                "split",
                "text",
                "raw_wav_path",
                "capture_status",
                "auditory_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "ready_for_clean_recapture",
        "policy_id": POLICY_ID,
        "dataset_version": RECORDING_V2_VERSION,
        "items_total": len(rows),
        "items_train": sum(row["split"] == "train" for row in rows),
        "items_val": sum(row["split"] == "val" for row in rows),
        "session_manifest": str(manifest_path),
        "raw_wav_directory": str(raw_dir),
        "accepted_wav_directory": str(accepted_dir),
        "old_audio_reused": False,
        "old_text_and_split_assignment_reused": True,
        "training_authorized": False,
        "next_action": "record each PENDING prompt cleanly; retake any external event overlapping speech",
    }
    (manifest_path.parent / "session_state.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_recording_v2_session(args.root, overwrite=args.overwrite), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
