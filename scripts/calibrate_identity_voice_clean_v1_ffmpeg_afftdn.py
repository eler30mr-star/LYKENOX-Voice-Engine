"""Calibrate conservative external offline denoising for CLEAN_V1.

This is a dataset-preparation diagnostic, not a LYKENOX runtime component. It never mutates source
WAV files and never writes canonical CLEAN_V1 WAVs. It stages a small listening trial using FFmpeg's
`afftdn` filter with two conservative strengths. Human listening is the acceptance authority.

Policy: LYX-POL-001 v1.1. No external model, checkpoint, service, gain normalization, EQ, dereverb,
or duration modification is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.identity_voice_clean_v1 import (
    CLEAN_V1_VERSION,
    POLICY_ID,
    clean_v1_root,
    clean_v1_work_manifest_path,
    load_clean_v1_state,
    sha256_file,
)


CALIBRATION_VERSION = "clean-v1-ffmpeg-afftdn-calibration-v1"
REQUIRED_AUDIT_IDS = (
    "speech_0021_6cd35984e877_seg_001",
    "speech_0022_ba721f6129b9_seg_005",
)
PROFILES: dict[str, str] = {
    "conservative": "afftdn=nr=6:nf=-50:tn=1:gs=5",
    "moderate": "afftdn=nr=10:nf=-50:tn=1:gs=8",
}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_work_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty CLEAN_V1 work manifest: {path}")
    return rows


def _resolve_manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _frame_rms_noise_floor_dbfs(path: Path) -> float:
    """Return a ranking proxy only; it never accepts perceptual quality.

    The 10th percentile of 30 ms frame RMS is used to find recordings whose quieter regions have
    relatively high energy. It is intentionally only a trial-selection heuristic.
    """
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0 or sample_rate <= 0:
        return -120.0
    mono = audio.mean(axis=1, dtype=np.float64)
    frame = max(32, int(round(0.030 * float(sample_rate))))
    hop = max(16, frame // 2)
    if mono.size < frame:
        rms = math.sqrt(float(np.mean(np.square(mono)))) if mono.size else 0.0
        return 20.0 * math.log10(max(rms, 1.0e-12))
    values: list[float] = []
    for start in range(0, mono.size - frame + 1, hop):
        chunk = mono[start : start + frame]
        values.append(math.sqrt(float(np.mean(np.square(chunk)))))
    percentile = float(np.percentile(np.asarray(values, dtype=np.float64), 10.0))
    return 20.0 * math.log10(max(percentile, 1.0e-12))


def _ffmpeg_version(ffmpeg: str) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    first = (result.stdout or result.stderr).splitlines()
    return first[0].strip() if first else "unknown"


def _require_afftdn(ffmpeg: str) -> None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        check=True,
        capture_output=True,
        text=True,
    )
    text = f"{result.stdout}\n{result.stderr}"
    if " afftdn " not in text and "afftdn" not in text:
        raise RuntimeError("the selected FFmpeg build does not expose the afftdn audio filter")


def _render(ffmpeg: str, source: Path, output: Path, filter_graph: str) -> None:
    info = sf.info(str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".tmp.wav")
    if tmp.exists():
        tmp.unlink()
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
        str(int(info.samplerate)),
        "-ac",
        str(int(info.channels)),
        "-c:a",
        "pcm_f32le",
        str(tmp),
    ]
    subprocess.run(command, check=True)
    rendered = sf.info(str(tmp))
    if int(rendered.samplerate) != int(info.samplerate):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg changed sample rate for {source.name}")
    if int(rendered.channels) != int(info.channels):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg changed channel count for {source.name}")
    duration_ratio = (rendered.frames / rendered.samplerate) / max(info.frames / info.samplerate, 1.0e-9)
    if abs(duration_ratio - 1.0) > 0.002:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"FFmpeg changed duration beyond calibration tolerance for {source.name}: {duration_ratio:.6f}"
        )
    os.replace(tmp, output)


def select_trial_rows(root: Path, rows: list[dict[str, str]], *, items: int) -> list[dict[str, str]]:
    if items < len(REQUIRED_AUDIT_IDS):
        raise ValueError(f"items must be >= {len(REQUIRED_AUDIT_IDS)}")
    by_id = {row["utterance_id"]: row for row in rows}
    missing = [utterance_id for utterance_id in REQUIRED_AUDIT_IDS if utterance_id not in by_id]
    if missing:
        raise RuntimeError("required CLEAN_V1 audit utterances missing: " + ", ".join(missing))

    ranked: list[tuple[float, str, dict[str, str]]] = []
    for row in rows:
        source = _resolve_manifest_path(root, row["source_wav_path"])
        if not source.exists():
            raise FileNotFoundError(f"CLEAN_V1 source WAV missing: {source}")
        score = _frame_rms_noise_floor_dbfs(source)
        ranked.append((score, row["utterance_id"], row))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: list[dict[str, str]] = [by_id[utterance_id] for utterance_id in REQUIRED_AUDIT_IDS]
    selected_ids = {row["utterance_id"] for row in selected}
    for _, _, row in ranked:
        if row["utterance_id"] in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row["utterance_id"])
        if len(selected) >= items:
            break
    return selected


def calibrate_clean_v1_ffmpeg_afftdn(
    root: Path,
    *,
    ffmpeg: str = "ffmpeg",
    items: int = 6,
) -> dict[str, object]:
    root = Path(root).resolve()
    state = load_clean_v1_state(root)
    if state is None or state.get("status") != "ready_for_external_cleaning":
        raise RuntimeError(
            "CLEAN_V1 must be in ready_for_external_cleaning state before denoise calibration"
        )
    work_path = clean_v1_work_manifest_path(root)
    rows = _read_work_manifest(work_path)
    if int(state.get("items_total", -1)) != len(rows):
        raise RuntimeError("CLEAN_V1 state/work-manifest item count mismatch")

    _require_afftdn(ffmpeg)
    version = _ffmpeg_version(ffmpeg)
    selected = select_trial_rows(root, rows, items=items)

    trial_root = clean_v1_root(root) / "trials" / "ffmpeg_afftdn_v1"
    source_dir = trial_root / "source"
    audition_dir = trial_root / "audition"
    source_dir.mkdir(parents=True, exist_ok=True)
    audition_dir.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, object]] = []
    for row in selected:
        utterance_id = row["utterance_id"]
        source = _resolve_manifest_path(root, row["source_wav_path"])
        if sha256_file(source) != row["source_sha256"]:
            raise RuntimeError(f"source immutability violation during calibration: {utterance_id}")
        noise_floor_proxy = _frame_rms_noise_floor_dbfs(source)

        source_copy = source_dir / f"{utterance_id}__SOURCE.wav"
        shutil.copy2(source, source_copy)
        outputs: dict[str, str] = {}
        for profile_name, filter_graph in PROFILES.items():
            output = audition_dir / f"{utterance_id}__{profile_name.upper()}.wav"
            _render(ffmpeg, source, output, filter_graph)
            outputs[profile_name] = str(output)

        report_rows.append(
            {
                "utterance_id": utterance_id,
                "split": row["split"],
                "selection_noise_floor_proxy_dbfs": round(noise_floor_proxy, 3),
                "source": str(source_copy),
                "outputs": outputs,
            }
        )

    report: dict[str, object] = {
        "status": "awaiting_human_listening",
        "schema": CALIBRATION_VERSION,
        "policy_id": POLICY_ID,
        "policy_version": "1.1",
        "dataset_version": CLEAN_V1_VERSION,
        "tool": "FFmpeg afftdn",
        "tool_version": version,
        "external_offline_tool": True,
        "external_model_or_checkpoint_used": False,
        "external_service_used": False,
        "source_audio_mutated": False,
        "canonical_clean_v1_wav_written": False,
        "candidate_wav_encoding": "pcm_f32le",
        "gain_normalization_used": False,
        "eq_used": False,
        "dereverb_used": False,
        "duration_modification_requested": False,
        "metrics_can_accept_perceptual_quality": False,
        "human_auditory_quality_is_authority": True,
        "profiles": PROFILES,
        "selection": {
            "items": len(selected),
            "required_ids": list(REQUIRED_AUDIT_IDS),
            "remaining_items_selected_by_high_quiet_frame_rms_proxy": True,
            "selection_metric_is_acceptance_evidence": False,
        },
        "trial_root": str(trial_root),
        "items": report_rows,
        "next_action": (
            "listen to SOURCE vs CONSERVATIVE vs MODERATE; choose one profile only if denoise improves "
            "cleanliness without altering identity, consonants, pitch, body, or natural phase/texture"
        ),
    }
    _atomic_json(trial_root / "calibration_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--items", type=int, default=6)
    args = parser.parse_args()
    print(
        json.dumps(
            calibrate_clean_v1_ffmpeg_afftdn(args.root, ffmpeg=args.ffmpeg, items=args.items),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
