"""Technical validation for the 10-take RECORDING_V2 pilot.

The validator never modifies audio. It checks capture geometry and basic signal integrity only;
metrics may reject a take but cannot accept perceptual quality. Human listening remains mandatory.
Policy: LYX-POL-001 v1.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.identity_voice_recording_v2 import (
    POLICY_ID,
    RECORDING_V2_VERSION,
    recording_v2_raw_dir,
    recording_v2_session_manifest,
)
from scripts.prepare_identity_voice_recording_v2_pilot import PILOT_VERSION

VALID_SUBTYPES = {"PCM_24", "FLOAT"}
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty pilot manifest: {path}")
    return rows


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1.0e-12))


def _resolve_raw(root: Path, raw_dir: Path, row: dict[str, str]) -> Path:
    # Canonical capture location is derived from recording_id, not trusted from editable CSV text.
    expected = raw_dir / f"{row['recording_id']}.wav"
    return expected.resolve()


def validate_recording_v2_pilot(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    metadata_dir = recording_v2_session_manifest(root).parent
    pilot_manifest = metadata_dir / "pilot_manifest.csv"
    pilot_state = metadata_dir / "pilot_state.json"
    if not pilot_manifest.exists() or not pilot_state.exists():
        raise FileNotFoundError(
            "RECORDING_V2 pilot metadata missing. Run prepare_identity_voice_recording_v2_pilot.py first."
        )

    state = json.loads(pilot_state.read_text(encoding="utf-8"))
    if state.get("schema") != PILOT_VERSION:
        raise RuntimeError("RECORDING_V2 pilot state schema mismatch")
    rows = _read_rows(pilot_manifest)
    if len(rows) != 10:
        raise RuntimeError(f"RECORDING_V2 pilot must contain exactly 10 rows, got {len(rows)}")

    raw_dir = recording_v2_raw_dir(root)
    items: list[dict[str, object]] = []
    technical_failures = 0
    missing = 0

    for row in rows:
        recording_id = row["recording_id"]
        wav_path = _resolve_raw(root, raw_dir, row)
        failures: list[str] = []
        warnings: list[str] = []

        if not wav_path.exists():
            missing += 1
            failures.append("wav_missing")
            items.append(
                {
                    "pilot_order": int(row["pilot_order"]),
                    "recording_id": recording_id,
                    "split": row["split"],
                    "wav_path": str(wav_path),
                    "technical_status": "FAIL",
                    "failures": failures,
                    "warnings": warnings,
                }
            )
            technical_failures += 1
            continue

        info = sf.info(str(wav_path))
        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if int(sample_rate) != TARGET_SAMPLE_RATE or int(info.samplerate) != TARGET_SAMPLE_RATE:
            failures.append("sample_rate_must_be_48000")
        if int(info.channels) != TARGET_CHANNELS or int(audio.shape[1]) != TARGET_CHANNELS:
            failures.append("channel_count_must_be_mono")
        if str(info.subtype) not in VALID_SUBTYPES:
            failures.append("subtype_must_be_pcm24_or_float32")
        if audio.size == 0:
            failures.append("audio_empty")
        elif not bool(np.isfinite(audio).all()):
            failures.append("audio_nonfinite")

        duration_seconds = float(info.frames) / float(info.samplerate)
        mono = audio[:, 0].astype(np.float64, copy=False) if audio.size else np.zeros(0)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        dc = float(abs(np.mean(mono))) if mono.size else 0.0
        clipping_fraction = float(np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0
        peak_dbfs = _dbfs(peak)
        rms_dbfs = _dbfs(rms)
        dc_dbfs = _dbfs(dc)

        if duration_seconds < 0.50:
            failures.append("duration_too_short")
        elif duration_seconds > 60.0:
            warnings.append("duration_over_60_seconds_review_take")
        if peak > 1.0001:
            failures.append("peak_above_full_scale")
        if clipping_fraction > 0.001:
            failures.append("excessive_clipping")
        elif clipping_fraction > 0.0:
            warnings.append("possible_clipping_samples_detected")
        if rms < 1.0e-5:
            failures.append("effectively_silent")

        # Level targets are warnings only: no gain normalization is performed or authorized here.
        if peak_dbfs > -3.0:
            warnings.append("peak_hotter_than_minus_3_dbfs")
        if peak_dbfs < -24.0:
            warnings.append("peak_quieter_than_minus_24_dbfs")
        if rms_dbfs < -40.0:
            warnings.append("rms_quiet_review_input_gain")
        if dc_dbfs > -35.0:
            warnings.append("dc_offset_review")

        status = "PASS" if not failures else "FAIL"
        if failures:
            technical_failures += 1
        items.append(
            {
                "pilot_order": int(row["pilot_order"]),
                "recording_id": recording_id,
                "source_prompt_id": row["source_prompt_id"],
                "split": row["split"],
                "text": row["text"],
                "wav_path": str(wav_path),
                "technical_status": status,
                "failures": failures,
                "warnings": warnings,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "subtype": str(info.subtype),
                "duration_seconds": round(duration_seconds, 4),
                "peak_dbfs": round(peak_dbfs, 3),
                "rms_dbfs": round(rms_dbfs, 3),
                "dc_dbfs": round(dc_dbfs, 3),
                "clipping_fraction": clipping_fraction,
            }
        )

    technical_passed = technical_failures == 0
    report = {
        "status": "ready_for_human_listening" if technical_passed else "pilot_capture_needs_retake_or_fix",
        "schema": "lykenox-recording-v2-pilot-technical-validation-v1",
        "policy_id": POLICY_ID,
        "dataset_version": RECORDING_V2_VERSION,
        "pilot_schema": PILOT_VERSION,
        "items_total": len(rows),
        "technical_failures": technical_failures,
        "missing_wavs": missing,
        "technical_validation_passed": technical_passed,
        "audio_processed": False,
        "gain_normalization_used": False,
        "denoise_used": False,
        "metrics_can_accept_perceptual_quality": False,
        "human_auditory_acceptance_required": True,
        "items": items,
        "next_action": (
            "listen to all 10 raw pilot takes and approve microphone/environment consistency"
            if technical_passed
            else "retake or correct only failed pilot captures, then rerun this validator"
        ),
    }
    report_path = metadata_dir / "pilot_technical_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    review_path = metadata_dir / "pilot_listening_review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pilot_order",
                "recording_id",
                "split",
                "wav_path",
                "technical_status",
                "auditory_status",
                "auditory_notes",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "pilot_order": item["pilot_order"],
                    "recording_id": item["recording_id"],
                    "split": item["split"],
                    "wav_path": item["wav_path"],
                    "technical_status": item["technical_status"],
                    "auditory_status": "PENDING",
                    "auditory_notes": "",
                }
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate_recording_v2_pilot(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
