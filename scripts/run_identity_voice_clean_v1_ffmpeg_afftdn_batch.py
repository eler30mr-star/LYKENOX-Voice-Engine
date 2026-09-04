"""Apply one human-approved FFmpeg afftdn profile to all CLEAN_V1 work items.

This is external offline dataset preparation under LYX-POL-001 v1.1. It never mutates source audio,
never runs unless a profile is explicitly selected, and writes only canonical CLEAN_V1 candidate WAVs.
The output still requires the existing technical validation + human auditory ACCEPT/REJECT gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.identity_voice_clean_v1 import (
    CLEAN_V1_VERSION,
    POLICY_ID,
    clean_v1_root,
    clean_v1_state_path,
    clean_v1_work_manifest_path,
    load_clean_v1_state,
    sha256_file,
)
from scripts.calibrate_identity_voice_clean_v1_ffmpeg_afftdn import PROFILES


BATCH_VERSION = "clean-v1-ffmpeg-afftdn-batch-v1"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty CLEAN_V1 work manifest: {path}")
    return rows


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _ffmpeg_version(ffmpeg: str) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-version"], check=True, capture_output=True, text=True
    )
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0].strip() if lines else "unknown"


def _require_afftdn(ffmpeg: str) -> None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"], check=True, capture_output=True, text=True
    )
    text = f"{result.stdout}\n{result.stderr}"
    if "afftdn" not in text:
        raise RuntimeError("the selected FFmpeg build does not expose the afftdn audio filter")


def _render(ffmpeg: str, source: Path, output: Path, filter_graph: str) -> dict[str, object]:
    source_info = sf.info(str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".tmp.wav")
    tmp.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-af",
        filter_graph,
        "-ar",
        str(int(source_info.samplerate)),
        "-ac",
        str(int(source_info.channels)),
        "-c:a",
        "pcm_s24le",
        str(tmp),
    ]
    subprocess.run(command, check=True)
    out_info = sf.info(str(tmp))
    source_duration = source_info.frames / float(source_info.samplerate)
    out_duration = out_info.frames / float(out_info.samplerate)
    ratio = out_duration / max(source_duration, 1.0e-9)
    if int(out_info.samplerate) != int(source_info.samplerate):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sample rate changed for {source.name}")
    if int(out_info.channels) != int(source_info.channels):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"channel count changed for {source.name}")
    if abs(ratio - 1.0) > 0.002:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"duration changed beyond tolerance for {source.name}: {ratio:.6f}")
    os.replace(tmp, output)
    return {
        "sample_rate": int(out_info.samplerate),
        "channels": int(out_info.channels),
        "duration_ratio": ratio,
        "sha256": sha256_file(output),
    }


def run_clean_v1_ffmpeg_afftdn_batch(
    root: Path,
    *,
    profile: str,
    ffmpeg: str = "ffmpeg",
    resume: bool = True,
) -> dict[str, object]:
    root = Path(root).resolve()
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    state = load_clean_v1_state(root)
    if state is None or state.get("status") != "ready_for_external_cleaning":
        raise RuntimeError("CLEAN_V1 must be in ready_for_external_cleaning state before batch cleaning")
    if state.get("training_authorized") is True:
        raise RuntimeError("unexpected CLEAN_V1 state: training is already authorized")

    work_path = clean_v1_work_manifest_path(root)
    rows = _read_rows(work_path)
    if int(state.get("items_total", -1)) != len(rows):
        raise RuntimeError("CLEAN_V1 state/work-manifest item count mismatch")

    _require_afftdn(ffmpeg)
    tool_version = _ffmpeg_version(ffmpeg)
    filter_graph = PROFILES[profile]
    clean_root = clean_v1_root(root)
    clean_wav_dir = clean_root / "wav"
    clean_wav_dir.mkdir(parents=True, exist_ok=True)

    report_items: list[dict[str, object]] = []
    processed = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        utterance_id = row["utterance_id"]
        source = _resolve(root, row["source_wav_path"])
        output = _resolve(root, row["clean_wav_path"])
        expected_output = clean_wav_dir / f"{utterance_id}.wav"
        if output.resolve() != expected_output.resolve():
            raise RuntimeError(
                f"CLEAN_V1 work manifest output path changed for {utterance_id}: {output} != {expected_output}"
            )
        if not source.exists():
            raise FileNotFoundError(f"source WAV missing: {source}")
        if sha256_file(source) != row["source_sha256"]:
            raise RuntimeError(f"source immutability violation: {utterance_id}")

        if resume and output.exists():
            out_info = sf.info(str(output))
            source_info = sf.info(str(source))
            ratio = (out_info.frames / float(out_info.samplerate)) / max(
                source_info.frames / float(source_info.samplerate), 1.0e-9
            )
            if (
                int(out_info.samplerate) == int(source_info.samplerate)
                and int(out_info.channels) == int(source_info.channels)
                and abs(ratio - 1.0) <= 0.002
            ):
                skipped += 1
                result = {
                    "sample_rate": int(out_info.samplerate),
                    "channels": int(out_info.channels),
                    "duration_ratio": ratio,
                    "sha256": sha256_file(output),
                }
            else:
                result = _render(ffmpeg, source, output, filter_graph)
                processed += 1
        else:
            result = _render(ffmpeg, source, output, filter_graph)
            processed += 1

        report_items.append(
            {
                "utterance_id": utterance_id,
                "split": row["split"],
                "source_sha256": row["source_sha256"],
                "clean_wav_path": str(output),
                **result,
            }
        )
        print(f"[CLEAN_V1] {index}/{len(rows)} {utterance_id}", flush=True)

    report: dict[str, object] = {
        "status": "batch_cleaning_complete_awaiting_validation",
        "schema": BATCH_VERSION,
        "policy_id": POLICY_ID,
        "policy_version": "1.1",
        "dataset_version": CLEAN_V1_VERSION,
        "tool": "FFmpeg afftdn",
        "tool_version": tool_version,
        "profile": profile,
        "filter_graph": filter_graph,
        "external_offline_tool": True,
        "external_model_or_checkpoint_used": False,
        "external_service_used": False,
        "source_audio_mutated": False,
        "gain_normalization_used": False,
        "eq_used": False,
        "dereverb_used": False,
        "duration_modification_requested": False,
        "items_total": len(rows),
        "processed": processed,
        "skipped_existing_valid_geometry": skipped,
        "items": report_items,
        "next_action": "run validate_identity_voice_clean_v1.py then complete human auditory review",
    }
    report_path = clean_root / "external_cleaning_report.json"
    _atomic_json(report_path, report)

    new_state = dict(state)
    new_state.update(
        {
            "status": "external_cleaning_complete_awaiting_validation",
            "external_cleaning_report": str(report_path),
            "external_cleaning_tool": {
                "name": "FFmpeg afftdn",
                "version": tool_version,
                "profile": profile,
                "filter_graph": filter_graph,
                "external_model_or_checkpoint_used": False,
                "external_service_used": False,
            },
            "technical_validation_passed": False,
            "human_auditory_review_complete": False,
            "all_acoustic_targets_and_caches_regenerated": False,
            "training_authorized": False,
            "next_action": "run validate_identity_voice_clean_v1.py",
        }
    )
    _atomic_json(clean_v1_state_path(root), new_state)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_clean_v1_ffmpeg_afftdn_batch(
                args.root,
                profile=args.profile,
                ffmpeg=args.ffmpeg,
                resume=not args.no_resume,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
