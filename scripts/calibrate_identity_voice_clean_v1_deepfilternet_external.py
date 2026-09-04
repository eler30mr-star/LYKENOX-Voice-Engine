"""Calibrate external/offline DeepFilterNet noise suppression for CLEAN_V1.

This script does not import or embed DeepFilterNet. It invokes a user-provided external executable
installed outside the LYKENOX project, stages only non-canonical trial WAVs, restores the original
sample rate/channel count/exact frame count with FFmpeg, and leaves human listening as the sole
perceptual acceptance authority.

Policy: LYX-POL-001 v1.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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
    clean_v1_work_manifest_path,
    load_clean_v1_state,
    sha256_file,
)


CALIBRATION_VERSION = "clean-v1-external-deepfilternet-calibration-v1"
EXPECTED_PACKAGE = "deepfilternet-rs"
EXPECTED_PACKAGE_VERSION = "0.1.1"
REQUIRED_AUDIT_IDS = (
    "speech_0021_6cd35984e877_seg_001",
    "speech_0022_ba721f6129b9_seg_005",
)
PROFILES: dict[str, float] = {
    "balanced": 24.0,
    "full": 100.0,
}


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


def _selected_ids_from_previous_trial(root: Path, *, items: int) -> list[str]:
    if items < len(REQUIRED_AUDIT_IDS):
        raise ValueError(f"items must be >= {len(REQUIRED_AUDIT_IDS)}")
    report_path = clean_v1_root(root) / "trials" / "ffmpeg_afftdn_v1" / "calibration_report.json"
    selected: list[str] = list(REQUIRED_AUDIT_IDS)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for item in report.get("items", []):
            utterance_id = str(item.get("utterance_id", "")).strip()
            if utterance_id and utterance_id not in selected:
                selected.append(utterance_id)
            if len(selected) >= items:
                break
    return selected[:items]


def _select_rows(root: Path, rows: list[dict[str, str]], *, items: int) -> list[dict[str, str]]:
    by_id = {row["utterance_id"]: row for row in rows}
    selected_ids = _selected_ids_from_previous_trial(root, items=items)
    for utterance_id in REQUIRED_AUDIT_IDS:
        if utterance_id not in by_id:
            raise RuntimeError(f"required CLEAN_V1 audit utterance missing: {utterance_id}")
    if len(selected_ids) < items:
        for utterance_id in sorted(by_id):
            if utterance_id not in selected_ids:
                selected_ids.append(utterance_id)
            if len(selected_ids) >= items:
                break
    missing = [utterance_id for utterance_id in selected_ids if utterance_id not in by_id]
    if missing:
        raise RuntimeError("previous trial references unknown CLEAN_V1 ids: " + ", ".join(missing))
    return [by_id[utterance_id] for utterance_id in selected_ids]


def _external_tool_version(executable: Path) -> str:
    python_exe = executable.parent / ("python.exe" if os.name == "nt" else "python")
    if not python_exe.exists():
        return "unknown"
    result = subprocess.run(
        [str(python_exe), "-m", "pip", "show", EXPECTED_PACKAGE],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    for line in result.stdout.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _require_tool(executable: Path, ffmpeg: str) -> tuple[str, str]:
    if not executable.exists():
        raise FileNotFoundError(f"external DeepFilterNet executable not found: {executable}")
    help_result = subprocess.run(
        [str(executable), "--help"], capture_output=True, text=True
    )
    if help_result.returncode != 0:
        raise RuntimeError(
            f"external DeepFilterNet executable failed --help: {help_result.stderr.strip()}"
        )
    ffmpeg_result = subprocess.run(
        [ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True
    )
    if ffmpeg_result.returncode != 0:
        raise RuntimeError("FFmpeg is required to restore CLEAN_V1 trial geometry")
    ffmpeg_lines = (ffmpeg_result.stdout or ffmpeg_result.stderr).splitlines()
    return _external_tool_version(executable), (ffmpeg_lines[0].strip() if ffmpeg_lines else "unknown")


def _run_external(
    executable: Path,
    source: Path,
    raw_output: Path,
    *,
    atten_lim_db: float,
) -> None:
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.unlink(missing_ok=True)
    command = [
        str(executable),
        str(source),
        str(raw_output),
        "--atten-lim",
        str(float(atten_lim_db)),
        "--log-level",
        "warn",
        "--compensate-delay",
        "--post-filter-beta",
        "0.0",
    ]
    subprocess.run(command, check=True)
    if not raw_output.exists():
        raise RuntimeError(f"DeepFilterNet did not create expected output: {raw_output}")


def _restore_geometry(
    ffmpeg: str,
    raw_output: Path,
    source: Path,
    final_output: Path,
) -> dict[str, object]:
    source_info = sf.info(str(source))
    if int(source_info.channels) != 1:
        raise RuntimeError(
            f"CLEAN_V1 DeepFilterNet calibration currently requires mono source audio: {source.name}"
        )
    source_frames = int(source_info.frames)
    source_rate = int(source_info.samplerate)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_output.with_name(final_output.stem + ".tmp.wav")
    tmp.unlink(missing_ok=True)
    filter_graph = (
        f"aresample={source_rate},"
        f"apad=whole_len={source_frames},"
        f"atrim=end_sample={source_frames}"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_output),
        "-vn",
        "-af",
        filter_graph,
        "-ar",
        str(source_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        str(tmp),
    ]
    subprocess.run(command, check=True)
    out_info = sf.info(str(tmp))
    if int(out_info.samplerate) != source_rate:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"restored DeepFilterNet output sample-rate mismatch: {source.name}")
    if int(out_info.channels) != 1:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"restored DeepFilterNet output channel mismatch: {source.name}")
    if int(out_info.frames) != source_frames:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"restored DeepFilterNet output frame-count mismatch: {source.name}: "
            f"{out_info.frames} != {source_frames}"
        )
    os.replace(tmp, final_output)
    return {
        "sample_rate": source_rate,
        "channels": 1,
        "frames": source_frames,
        "sha256": sha256_file(final_output),
    }


def calibrate_external_deepfilternet(
    root: Path,
    *,
    deepfilternet_exe: Path,
    ffmpeg: str = "ffmpeg",
    items: int = 6,
) -> dict[str, object]:
    root = Path(root).resolve()
    state = load_clean_v1_state(root)
    if state is None or state.get("status") != "ready_for_external_cleaning":
        raise RuntimeError(
            "CLEAN_V1 must be in ready_for_external_cleaning state before external DeepFilterNet calibration"
        )
    rows = _read_rows(clean_v1_work_manifest_path(root))
    if int(state.get("items_total", -1)) != len(rows):
        raise RuntimeError("CLEAN_V1 state/work-manifest item count mismatch")

    executable = Path(deepfilternet_exe).expanduser().resolve()
    tool_version, ffmpeg_version = _require_tool(executable, ffmpeg)
    selected = _select_rows(root, rows, items=items)

    trial_root = clean_v1_root(root) / "trials" / "external_deepfilternet_v1"
    audition_dir = trial_root / "audition"
    scratch_dir = trial_root / "scratch_48k"
    audition_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    report_items: list[dict[str, object]] = []
    for row in selected:
        utterance_id = row["utterance_id"]
        source = _resolve(root, row["source_wav_path"])
        if not source.exists():
            raise FileNotFoundError(f"CLEAN_V1 source WAV missing: {source}")
        if sha256_file(source) != row["source_sha256"]:
            raise RuntimeError(f"source immutability violation during DeepFilterNet calibration: {utterance_id}")

        source_copy = audition_dir / f"{utterance_id}__SOURCE.wav"
        shutil.copy2(source, source_copy)
        outputs: dict[str, object] = {}
        for profile, attenuation in PROFILES.items():
            raw_output = scratch_dir / f"{utterance_id}__{profile}.wav"
            final_output = audition_dir / f"{utterance_id}__DF_{profile.upper()}.wav"
            _run_external(executable, source, raw_output, atten_lim_db=attenuation)
            geometry = _restore_geometry(ffmpeg, raw_output, source, final_output)
            outputs[profile] = {
                "attenuation_limit_db": attenuation,
                "wav_path": str(final_output),
                **geometry,
            }

        report_items.append(
            {
                "utterance_id": utterance_id,
                "split": row["split"],
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
        "tool": EXPECTED_PACKAGE,
        "tool_version_detected": tool_version,
        "expected_tool_version": EXPECTED_PACKAGE_VERSION,
        "tool_executable_sha256": sha256_file(executable),
        "ffmpeg_version": ffmpeg_version,
        "external_offline_tool": True,
        "external_pretrained_model_used_for_offline_preparation": True,
        "external_model_or_checkpoint_integrated_into_lykenox": False,
        "external_service_used": False,
        "lykenox_runtime_dependency_created": False,
        "source_audio_mutated": False,
        "canonical_clean_v1_wav_written": False,
        "candidate_wav_encoding": "pcm_f32le",
        "internal_model_rate_hz": 48000,
        "output_geometry_restored_to_source": True,
        "gain_normalization_requested": False,
        "eq_requested": False,
        "dereverb_requested": False,
        "post_filter_beta": 0.0,
        "metrics_can_accept_perceptual_quality": False,
        "human_auditory_quality_is_authority": True,
        "profiles": PROFILES,
        "selection_items": len(selected),
        "items": report_items,
        "next_action": (
            "listen to SOURCE vs DF_BALANCED vs DF_FULL; accept only if environmental noise is materially "
            "reduced while identity, body, pitch, consonants, attacks and natural texture remain intact"
        ),
    }
    _atomic_json(trial_root / "calibration_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--deepfilternet-exe", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--items", type=int, default=6)
    args = parser.parse_args()
    print(
        json.dumps(
            calibrate_external_deepfilternet(
                args.root,
                deepfilternet_exe=args.deepfilternet_exe,
                ffmpeg=args.ffmpeg,
                items=args.items,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
