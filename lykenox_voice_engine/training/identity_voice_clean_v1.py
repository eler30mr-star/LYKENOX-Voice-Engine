"""CLEAN_V1 identity-voice dataset contract.

CLEAN_V1 is the only dataset authorized for new persistent speech/vocoder training after the
2026-09-04 data-quality gate. External offline tools may prepare/clean audio under LYX-POL-001 v1.1,
but no external model, checkpoint or service becomes a LYKENOX runtime/training dependency.

The original prepared speech_segmented corpus remains immutable historical input. CLEAN_V1 is
activated only after technical validation and explicit human auditory review. Activation switches
dataset reads to the clean WAVs; persistent source training remains blocked until every stale
acoustic derivative has been regenerated and the post-clean GOLD oracle gate has passed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


POLICY_ID = "LYX-POL-001"
CLEAN_V1_VERSION = "lykenox-identity-voice-clean-v1"
CLEAN_V1_STATE_SCHEMA = "lykenox-clean-v1-state-v1"
CLEAN_V1_DIR = Path("datasets/lykenox/identity_voice/clean_v1")
LEGACY_SEGMENTED_DIR = Path("datasets/lykenox/identity_voice/prepared/speech_segmented")


def clean_v1_root(root: Path) -> Path:
    return Path(root).resolve() / CLEAN_V1_DIR


def clean_v1_state_path(root: Path) -> Path:
    return clean_v1_root(root) / "state.json"


def clean_v1_work_manifest_path(root: Path) -> Path:
    return clean_v1_root(root) / "work_manifest.csv"


def clean_v1_review_path(root: Path) -> Path:
    return clean_v1_root(root) / "listening_review.csv"


def clean_v1_technical_report_path(root: Path) -> Path:
    return clean_v1_root(root) / "technical_validation.json"


def clean_v1_manifest_path(root: Path, split: str) -> Path:
    if split not in {"train", "val"}:
        raise ValueError(f"unsupported identity speech split: {split}")
    return clean_v1_root(root) / "manifests" / f"{split}.clean_v1.csv"


def legacy_segmented_manifest_path(root: Path, split: str) -> Path:
    if split not in {"train", "val"}:
        raise ValueError(f"unsupported identity speech split: {split}")
    return Path(root).resolve() / LEGACY_SEGMENTED_DIR / f"{split}.segmented.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clean_v1_state(root: Path) -> dict[str, object] | None:
    path = clean_v1_state_path(root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CLEAN_V1_STATE_SCHEMA:
        raise RuntimeError(f"CLEAN_V1 state schema mismatch: {path}")
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError(f"CLEAN_V1 policy id mismatch: {path}")
    if payload.get("dataset_version") != CLEAN_V1_VERSION:
        raise RuntimeError(f"CLEAN_V1 dataset version mismatch: {path}")
    return payload


def clean_v1_is_active(root: Path) -> bool:
    payload = load_clean_v1_state(root)
    if payload is None or payload.get("status") != "active":
        return False
    if payload.get("human_auditory_review_complete") is not True:
        return False
    if payload.get("technical_validation_passed") is not True:
        return False
    for split in ("train", "val"):
        if not clean_v1_manifest_path(root, split).exists():
            return False
    return True


def require_clean_v1_active(root: Path, *, purpose: str) -> dict[str, object]:
    payload = load_clean_v1_state(root)
    if payload is None:
        raise RuntimeError(
            f"{purpose} is blocked by LYX-POL-001 data gate: CLEAN_V1 has not been prepared. "
            "Run scripts/prepare_identity_voice_clean_v1.py first."
        )
    if not clean_v1_is_active(root):
        raise RuntimeError(
            f"{purpose} is blocked by LYX-POL-001 data gate: CLEAN_V1 status="
            f"{payload.get('status')!r}; technical validation and human auditory approval are required."
        )
    return payload


def require_clean_v1_training_ready(root: Path, *, purpose: str) -> dict[str, object]:
    """Require the stronger gate used by new persistent source/acoustic training."""
    payload = require_clean_v1_active(root, purpose=purpose)
    missing: list[str] = []
    if payload.get("all_acoustic_targets_and_caches_regenerated") is not True:
        missing.append("all_acoustic_targets_and_caches_regenerated")
    if payload.get("gold_oracles_rerun_after_clean_v1") is not True:
        missing.append("gold_oracles_rerun_after_clean_v1")
    if payload.get("training_authorized") is not True:
        missing.append("training_authorized")
    if missing:
        raise RuntimeError(
            f"{purpose} remains blocked after CLEAN_V1 activation; pending gates: " + ", ".join(missing)
        )
    return payload


def resolve_identity_speech_manifest(
    root: Path,
    split: str,
    *,
    allow_legacy_forensics: bool = True,
) -> Path:
    """Resolve the active speech manifest.

    Once CLEAN_V1 is active, every consumer is switched to CLEAN_V1. Before activation, historical
    diagnostics may still read the legacy prepared corpus when explicitly allowed. Persistent new
    training must enforce the appropriate CLEAN_V1 gate before constructing datasets.
    """
    if clean_v1_is_active(root):
        return clean_v1_manifest_path(root, split)
    if not allow_legacy_forensics:
        require_clean_v1_active(root, purpose=f"identity speech {split} dataset access")
    legacy = legacy_segmented_manifest_path(root, split)
    if legacy.exists():
        return legacy
    fallback = (
        Path(root).resolve()
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech"
        / f"{split}.csv"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No LYKENOX speech manifest found for split={split}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"utterance_id", "wav_path", "text"}
    if not rows:
        raise RuntimeError(f"empty speech manifest: {path}")
    if not required.issubset(rows[0]):
        raise RuntimeError(f"speech manifest missing required fields {sorted(required)}: {path}")
    return rows


__all__ = [
    "POLICY_ID",
    "CLEAN_V1_VERSION",
    "CLEAN_V1_STATE_SCHEMA",
    "CLEAN_V1_DIR",
    "clean_v1_root",
    "clean_v1_state_path",
    "clean_v1_work_manifest_path",
    "clean_v1_review_path",
    "clean_v1_technical_report_path",
    "clean_v1_manifest_path",
    "legacy_segmented_manifest_path",
    "sha256_file",
    "load_clean_v1_state",
    "clean_v1_is_active",
    "require_clean_v1_active",
    "require_clean_v1_training_ready",
    "resolve_identity_speech_manifest",
    "read_csv_rows",
]
