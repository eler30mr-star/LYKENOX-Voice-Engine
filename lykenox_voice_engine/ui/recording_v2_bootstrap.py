"""Desktop bootstrap for the RECORDING_V2 capture pilot.

The desktop app must be usable without PowerShell. This module prepares only metadata when the
workspace is missing RECORDING_V2 session/pilot CSVs. It never records, edits, cleans, resamples,
normalizes, deletes, or trains on audio. Existing canonical RAW WAVs are left untouched.

Policy: LYX-POL-001 v1.1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lykenox_voice_engine.training.identity_voice_recording_v2 import (
    POLICY_ID,
    RECORDING_V2_VERSION,
    recording_v2_accepted_dir,
    recording_v2_raw_dir,
    recording_v2_root,
    recording_v2_session_manifest,
)

PILOT_FILENAME = "pilot_manifest.csv"
PILOT_STATE_FILENAME = "pilot_state.json"
PILOT_VERSION = "lykenox-identity-voice-recording-v2-pilot-v1"
REQUIRED_SOURCE_PROMPT_IDS = (
    "speech_0021_6cd35984e877_seg_001",
    "speech_0022_ba721f6129b9_seg_005",
)
TARGET_TRAIN_ITEMS = 6
TARGET_VAL_ITEMS = 4


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
        raise FileNotFoundError(f"Falta manifiesto fuente para RECORDING_V2: {path}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Manifiesto vacío: {path}")
    return rows


def _ensure_session(root: Path) -> Path:
    manifest = recording_v2_session_manifest(root)
    if manifest.exists():
        rows = _read_csv(manifest)
        if len(rows) != 132:
            raise RuntimeError(f"session_manifest.csv inválido: se esperaban 132 filas y hay {len(rows)}")
        return manifest

    out_root = recording_v2_root(root)
    raw_dir = recording_v2_raw_dir(root)
    accepted_dir = recording_v2_accepted_dir(root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for split in ("train", "val"):
        for source in _read_csv(_source_manifest(root, split)):
            source_id = source["utterance_id"].strip()
            recording_id = f"{source_id}_rec2"
            if recording_id in seen:
                raise RuntimeError(f"ID RECORDING_V2 duplicado: {recording_id}")
            seen.add(recording_id)
            rows.append(
                {
                    "recording_id": recording_id,
                    "source_prompt_id": source_id,
                    "split": split,
                    "text": source["text"],
                    "raw_wav_path": str(Path("..") / "raw" / f"{recording_id}.wav"),
                    "capture_status": "PENDING",
                    "auditory_status": "PENDING",
                    "notes": "",
                }
            )

    if len(rows) != 132:
        raise RuntimeError(f"Fuente RECORDING_V2 cambió: se esperaban 132 filas y hay {len(rows)}")

    with manifest.open("w", encoding="utf-8", newline="") as handle:
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

    state = {
        "status": "ready_for_clean_recapture",
        "policy_id": POLICY_ID,
        "dataset_version": RECORDING_V2_VERSION,
        "items_total": 132,
        "items_train": sum(row["split"] == "train" for row in rows),
        "items_val": sum(row["split"] == "val" for row in rows),
        "session_manifest": str(manifest),
        "raw_wav_directory": str(raw_dir),
        "accepted_wav_directory": str(accepted_dir),
        "old_audio_reused": False,
        "old_text_and_split_assignment_reused": True,
        "audio_modified": False,
        "training_authorized": False,
        "prepared_by": "desktop_metadata_bootstrap",
    }
    (manifest.parent / "session_state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _even_length_sample(
    rows: list[dict[str, str]], count: int, *, excluded_ids: set[str]
) -> list[dict[str, str]]:
    candidates = [row for row in rows if row["recording_id"] not in excluded_ids]
    candidates.sort(key=lambda row: (len(row["text"]), row["recording_id"]))
    if count <= 0:
        return []
    if len(candidates) < count:
        raise RuntimeError(f"No hay suficientes candidatos para piloto: {len(candidates)} < {count}")
    if count == 1:
        return [candidates[len(candidates) // 2]]

    selected: list[dict[str, str]] = []
    used: set[str] = set()
    for position in range(count):
        index = int(round(position * (len(candidates) - 1) / float(count - 1)))
        if candidates[index]["recording_id"] in used:
            found: int | None = None
            for offset in range(1, len(candidates)):
                for candidate_index in (index - offset, index + offset):
                    if 0 <= candidate_index < len(candidates):
                        if candidates[candidate_index]["recording_id"] not in used:
                            found = candidate_index
                            break
                if found is not None:
                    break
            if found is None:
                raise RuntimeError("No se pudo resolver selección determinista del piloto")
            index = found
        row = candidates[index]
        selected.append(row)
        used.add(row["recording_id"])
    return selected


def ensure_recording_v2_pilot(root: Path) -> dict[str, object]:
    """Ensure the 132-row session and deterministic 10-row pilot metadata exist.

    Existing files are validated and reused. No audio file is created, copied, changed, or removed.
    """

    root = Path(root).resolve()
    session = _ensure_session(root)
    metadata_dir = session.parent
    pilot = metadata_dir / PILOT_FILENAME

    if pilot.exists():
        rows = _read_csv(pilot)
        if len(rows) != 10:
            raise RuntimeError(f"pilot_manifest.csv inválido: se esperaban 10 filas y hay {len(rows)}")
        return {
            "status": "ready_for_recording_v2_pilot_capture",
            "items_total": 10,
            "pilot_manifest": str(pilot),
            "metadata_created": False,
            "audio_modified": False,
        }

    rows = _read_csv(session)
    by_source = {row["source_prompt_id"]: row for row in rows}
    required: list[dict[str, str]] = []
    for source_id in REQUIRED_SOURCE_PROMPT_IDS:
        row = by_source.get(source_id)
        if row is None:
            raise RuntimeError(f"Falta prompt obligatorio del piloto: {source_id}")
        required.append(row)

    selected_ids = {row["recording_id"] for row in required}
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    train_needed = TARGET_TRAIN_ITEMS - sum(row["split"] == "train" for row in required)
    val_needed = TARGET_VAL_ITEMS - sum(row["split"] == "val" for row in required)

    selected = list(required)
    train_extra = _even_length_sample(train_rows, train_needed, excluded_ids=selected_ids)
    selected.extend(train_extra)
    selected_ids.update(row["recording_id"] for row in train_extra)
    selected.extend(_even_length_sample(val_rows, val_needed, excluded_ids=selected_ids))

    if len(selected) != 10:
        raise RuntimeError(f"Selección de piloto inválida: {len(selected)}")

    with pilot.open("w", encoding="utf-8", newline="") as handle:
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

    state = {
        "status": "ready_for_recording_v2_pilot_capture",
        "schema": PILOT_VERSION,
        "policy_id": POLICY_ID,
        "dataset_version": RECORDING_V2_VERSION,
        "items_total": 10,
        "items_train": sum(row["split"] == "train" for row in selected),
        "items_val": sum(row["split"] == "val" for row in selected),
        "required_source_prompt_ids": list(REQUIRED_SOURCE_PROMPT_IDS),
        "selection_strategy": "required_0021_0022_plus_even_text_length_coverage_by_split",
        "selection_metric_is_acceptance_evidence": False,
        "audio_processed": False,
        "audio_modified": False,
        "training_authorized": False,
        "pilot_manifest": str(pilot),
        "prepared_by": "desktop_metadata_bootstrap",
    }
    (metadata_dir / PILOT_STATE_FILENAME).write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "status": state["status"],
        "items_total": 10,
        "pilot_manifest": str(pilot),
        "metadata_created": True,
        "audio_modified": False,
    }


__all__ = ["ensure_recording_v2_pilot"]
