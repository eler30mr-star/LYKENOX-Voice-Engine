"""Persistent frame-aligned F0/voicing target cache for LYKENOX Speech.

The accepted v4.1 vocoder is conditioned by mel + F0 + voicing. During vocoder isolation
those pitch controls came from the paired waveform. Product inference cannot do that, so
the acoustic model must learn to predict the same contract. This module creates the owned
training targets for that next stage.

The cache is deliberately versioned and resumable:
- one target file per utterance, keyed by waveform SHA256 + mel/pitch configuration;
- exact centered-frame alignment with the existing mel cache is required;
- train/val manifest hashes and speech-mel configuration are recorded;
- each invocation has a conservative CPU wall-clock budget and can be rerun safely;
- a completed index contains hashes for every cached target artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch
import torchaudio

from lykenox_voice_engine.audio.io import load_audio
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_aligner_train import _dataset, _manifest_path
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset
from lykenox_voice_engine.training.speech_pitch import (
    PITCH_TARGET_VERSION,
    PitchFrames,
    extract_pitch_frames,
)


PITCH_CACHE_VERSION = "speech-pitch-cache-v1"
DEFAULT_TIME_BUDGET_SECONDS = 70.0
DEFAULT_CHECKPOINT_RESERVE_SECONDS = 8.0
PITCH_CONFIG: dict[str, object] = {
    "frame_length": 1024,
    "min_f0_hz": 60.0,
    "max_f0_hz": 350.0,
    "voiced_periodicity_threshold": 0.30,
    "voiced_rms_fraction": 0.08,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cache_root(root: Path) -> Path:
    return (
        Path(root)
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "features"
        / "speech"
        / "pitch-v1"
    )


def _mono_waveform(path: Path, config: LykenoxSpeechConfig) -> torch.Tensor:
    waveform, sample_rate = load_audio(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            config.sample_rate,
        )
    peak = waveform.abs().max().clamp_min(1e-8)
    if peak > 1.0:
        waveform = waveform / peak
    return waveform.squeeze(0).to(torch.float32).contiguous()


def _cache_identity(
    *,
    split: str,
    utterance_id: str,
    wav_path: Path,
    wav_sha256: str,
    mel_frames: int,
    manifest_sha256: str,
    speech_config: LykenoxSpeechConfig,
) -> dict[str, object]:
    return {
        "cache_version": PITCH_CACHE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "split": split,
        "utterance_id": utterance_id,
        "wav_path": str(Path(wav_path).resolve()),
        "wav_sha256": wav_sha256,
        "mel_frames": int(mel_frames),
        "manifest_sha256": manifest_sha256,
        "mel_cache_version": LykenoxSpeechDataset.CACHE_VERSION,
        "speech_config_sha256": _sha256_json(speech_config.to_dict()),
        "pitch_config": dict(PITCH_CONFIG),
    }


def _target_path(
    cache_root: Path,
    identity: dict[str, object],
) -> Path:
    digest = _sha256_json(identity)[:20]
    utterance_id = str(identity["utterance_id"])
    split = str(identity["split"])
    return cache_root / split / f"{utterance_id}-{digest}.pt"


def _extract_target(
    waveform: torch.Tensor,
    *,
    mel_frames: int,
    speech_config: LykenoxSpeechConfig,
) -> PitchFrames:
    return extract_pitch_frames(
        waveform,
        frame_count=mel_frames,
        sample_rate=speech_config.sample_rate,
        hop_length=speech_config.hop_length,
        frame_length=int(PITCH_CONFIG["frame_length"]),
        min_f0_hz=float(PITCH_CONFIG["min_f0_hz"]),
        max_f0_hz=float(PITCH_CONFIG["max_f0_hz"]),
        voiced_periodicity_threshold=float(
            PITCH_CONFIG["voiced_periodicity_threshold"]
        ),
        voiced_rms_fraction=float(PITCH_CONFIG["voiced_rms_fraction"]),
    )


def _validate_tensors(
    pitch: PitchFrames,
    *,
    mel_frames: int,
) -> None:
    expected = (int(mel_frames),)
    if tuple(pitch.f0_hz.shape) != expected:
        raise RuntimeError("F0 target length does not match mel frames")
    if tuple(pitch.voiced.shape) != expected:
        raise RuntimeError("voicing target length does not match mel frames")
    if tuple(pitch.periodicity.shape) != expected:
        raise RuntimeError("periodicity target length does not match mel frames")
    if not torch.isfinite(pitch.f0_hz).all():
        raise RuntimeError("F0 target contains non-finite values")
    if not torch.isfinite(pitch.voiced).all():
        raise RuntimeError("voicing target contains non-finite values")
    if not torch.isfinite(pitch.periodicity).all():
        raise RuntimeError("periodicity target contains non-finite values")
    if not bool(((pitch.voiced == 0.0) | (pitch.voiced == 1.0)).all()):
        raise RuntimeError("voicing targets must be binary")
    if not bool((pitch.f0_hz[pitch.voiced < 0.5] == 0.0).all()):
        raise RuntimeError("unvoiced F0 targets must be exactly zero")
    voiced_f0 = pitch.f0_hz[pitch.voiced > 0.5]
    if voiced_f0.numel():
        if float(voiced_f0.min()) < float(PITCH_CONFIG["min_f0_hz"]):
            raise RuntimeError("voiced F0 target is below configured range")
        if float(voiced_f0.max()) > float(PITCH_CONFIG["max_f0_hz"]):
            raise RuntimeError("voiced F0 target is above configured range")


def _payload_to_pitch(payload: dict[str, Any]) -> PitchFrames:
    f0 = payload.get("f0_hz")
    voiced = payload.get("voiced")
    periodicity = payload.get("periodicity")
    if not isinstance(f0, torch.Tensor):
        raise RuntimeError("pitch cache payload is missing f0_hz")
    if not isinstance(voiced, torch.Tensor):
        raise RuntimeError("pitch cache payload is missing voiced")
    if not isinstance(periodicity, torch.Tensor):
        raise RuntimeError("pitch cache payload is missing periodicity")
    return PitchFrames(
        f0_hz=f0.to(torch.float32).contiguous(),
        voiced=voiced.to(torch.float32).contiguous(),
        periodicity=periodicity.to(torch.float32).contiguous(),
    )


def _load_valid_target(
    path: Path,
    *,
    expected_identity: dict[str, object],
) -> PitchFrames | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return None
        if payload.get("identity") != expected_identity:
            return None
        pitch = _payload_to_pitch(payload)
        _validate_tensors(pitch, mel_frames=int(expected_identity["mel_frames"]))
        return pitch
    except Exception:
        return None


def load_pitch_cache_index(root: Path) -> dict[str, Any]:
    path = _cache_root(Path(root).resolve()) / "cache_index.json"
    if not path.exists():
        raise FileNotFoundError(f"LYKENOX pitch cache index is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("cache_version") != PITCH_CACHE_VERSION:
        raise RuntimeError("LYKENOX pitch cache index is incompatible")
    return payload


def load_indexed_pitch_target(
    root: Path,
    *,
    split: str,
    utterance_id: str,
) -> PitchFrames:
    """Load one target through the completed versioned index.

    This is the stable consumer boundary intended for the acoustic-model dataset. It does
    not re-run pitch extraction and therefore cannot introduce a waveform dependency into
    product inference.
    """

    index = load_pitch_cache_index(root)
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("LYKENOX pitch cache index has no entries")
    match = next(
        (
            row
            for row in entries
            if isinstance(row, dict)
            and row.get("split") == split
            and row.get("utterance_id") == utterance_id
        ),
        None,
    )
    if not isinstance(match, dict):
        raise KeyError(f"No pitch target indexed for {split}/{utterance_id}")
    relative = match.get("cache_path")
    if not isinstance(relative, str):
        raise RuntimeError("Indexed pitch target has no cache_path")
    path = _cache_root(Path(root).resolve()) / relative
    if _sha256_file(path) != match.get("cache_sha256"):
        raise RuntimeError(f"Pitch cache artifact hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid indexed pitch target payload")
    pitch = _payload_to_pitch(payload)
    _validate_tensors(pitch, mel_frames=int(match["mel_frames"]))
    return pitch


def build_pitch_target_cache(
    root: Path,
    *,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    checkpoint_reserve_seconds: float = DEFAULT_CHECKPOINT_RESERVE_SECONDS,
) -> dict[str, object]:
    if time_budget_seconds <= checkpoint_reserve_seconds + 5.0:
        raise ValueError("time budget is too small for safe pitch-cache progress")

    root = Path(root).resolve()
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    speech_config = LykenoxSpeechConfig()
    cache_root = _cache_root(root)
    cache_root.mkdir(parents=True, exist_ok=True)
    progress_path = cache_root / "cache_progress.json"
    report_path = cache_root / "cache_report.json"
    index_path = cache_root / "cache_index.json"

    manifests = {split: _manifest_path(root, split) for split in ("train", "val")}
    manifest_hashes = {split: _sha256_file(path) for split, path in manifests.items()}
    datasets = {split: _dataset(root, split, speech_config) for split in ("train", "val")}
    total_expected = sum(len(dataset) for dataset in datasets.values())

    entries: list[dict[str, object]] = []
    cached_this_run = 0
    reused_this_run = 0
    exact_centered_alignment_count = 0
    voiced_fractions: list[float] = []
    voiced_f0_values: list[float] = []

    for split in ("train", "val"):
        dataset = datasets[split]
        manifest_sha = manifest_hashes[split]
        for index in range(len(dataset)):
            elapsed = time.perf_counter() - started
            if elapsed >= time_budget_seconds - checkpoint_reserve_seconds:
                progress = {
                    "status": "incomplete",
                    "device": "cpu",
                    "cache_version": PITCH_CACHE_VERSION,
                    "pitch_target_version": PITCH_TARGET_VERSION,
                    "cached_or_reused": len(entries),
                    "total_expected": total_expected,
                    "remaining": total_expected - len(entries),
                    "cached_this_run": cached_this_run,
                    "reused_this_run": reused_this_run,
                    "elapsed_seconds": round(elapsed, 3),
                    "next_gate": "rerun_same_pitch_cache_command",
                }
                _atomic_json(progress_path, progress)
                return {**progress, "progress_report": str(progress_path)}

            item = dataset[index]
            utterance_id = str(item["utterance_id"])
            mel = item["mel"]
            wav_path = Path(str(item["wav_path"])).resolve()
            mel_frames = int(mel.shape[0])
            wav_sha = _sha256_file(wav_path)
            identity = _cache_identity(
                split=split,
                utterance_id=utterance_id,
                wav_path=wav_path,
                wav_sha256=wav_sha,
                mel_frames=mel_frames,
                manifest_sha256=manifest_sha,
                speech_config=speech_config,
            )
            target_path = _target_path(cache_root, identity)
            pitch = _load_valid_target(target_path, expected_identity=identity)

            if pitch is None:
                waveform = _mono_waveform(wav_path, speech_config)
                centered_frame_count = int(waveform.numel()) // speech_config.hop_length + 1
                if centered_frame_count != mel_frames:
                    raise RuntimeError(
                        "Pitch/mel centered-frame mismatch for "
                        f"{utterance_id}: waveform predicts {centered_frame_count}, "
                        f"mel has {mel_frames}"
                    )
                exact_centered_alignment_count += 1
                pitch = _extract_target(
                    waveform,
                    mel_frames=mel_frames,
                    speech_config=speech_config,
                )
                _validate_tensors(pitch, mel_frames=mel_frames)
                payload = {
                    "identity": identity,
                    "f0_hz": pitch.f0_hz,
                    "voiced": pitch.voiced,
                    "periodicity": pitch.periodicity,
                }
                _atomic_torch(target_path, payload)
                cached_this_run += 1
            else:
                # The identity itself contains the exact mel frame count and source hashes;
                # a valid reusable target therefore also satisfies the alignment contract.
                exact_centered_alignment_count += 1
                reused_this_run += 1

            voiced_fraction = float(pitch.voiced.mean())
            voiced_fractions.append(voiced_fraction)
            active = pitch.f0_hz[pitch.voiced > 0.5]
            voiced_f0_values.extend(float(value) for value in active.tolist())
            relative = target_path.relative_to(cache_root).as_posix()
            entries.append(
                {
                    "split": split,
                    "utterance_id": utterance_id,
                    "mel_frames": mel_frames,
                    "wav_sha256": wav_sha,
                    "cache_path": relative,
                    "cache_sha256": _sha256_file(target_path),
                    "voiced_fraction": round(voiced_fraction, 6),
                    "voiced_f0_min_hz": (
                        round(float(active.min()), 4) if active.numel() else None
                    ),
                    "voiced_f0_max_hz": (
                        round(float(active.max()), 4) if active.numel() else None
                    ),
                }
            )

    if len(entries) != total_expected:
        raise RuntimeError("Pitch cache completion count mismatch")

    index_payload: dict[str, object] = {
        "cache_version": PITCH_CACHE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "mel_cache_version": LykenoxSpeechDataset.CACHE_VERSION,
        "speech_config": speech_config.to_dict(),
        "speech_config_sha256": _sha256_json(speech_config.to_dict()),
        "pitch_config": dict(PITCH_CONFIG),
        "pitch_config_sha256": _sha256_json(PITCH_CONFIG),
        "manifest_paths": {split: str(path) for split, path in manifests.items()},
        "manifest_sha256": manifest_hashes,
        "train_count": len(datasets["train"]),
        "val_count": len(datasets["val"]),
        "total_count": total_expected,
        "entries": entries,
    }
    _atomic_json(index_path, index_payload)

    # Final read-through is intentional: the gate is not complete until every indexed
    # artifact can be loaded, hashed, and shown to have exact frame semantics.
    reload_exact = 0
    for entry in entries:
        pitch = load_indexed_pitch_target(
            root,
            split=str(entry["split"]),
            utterance_id=str(entry["utterance_id"]),
        )
        if int(pitch.f0_hz.numel()) == int(entry["mel_frames"]):
            reload_exact += 1

    status = (
        "pass"
        if reload_exact == total_expected
        and exact_centered_alignment_count == total_expected
        else "needs_review"
    )
    report: dict[str, object] = {
        "status": status,
        "device": "cpu",
        "cache_version": PITCH_CACHE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "train_count": len(datasets["train"]),
        "val_count": len(datasets["val"]),
        "total_count": total_expected,
        "exact_centered_frame_alignment_count": exact_centered_alignment_count,
        "indexed_reload_exact_count": reload_exact,
        "all_targets_exact_mel_length": reload_exact == total_expected,
        "all_centered_frame_counts_match_mel": (
            exact_centered_alignment_count == total_expected
        ),
        "mean_voiced_fraction": round(statistics.fmean(voiced_fractions), 6),
        "min_voiced_f0_hz": (
            round(min(voiced_f0_values), 4) if voiced_f0_values else None
        ),
        "max_voiced_f0_hz": (
            round(max(voiced_f0_values), 4) if voiced_f0_values else None
        ),
        "cached_this_run": cached_this_run,
        "reused_this_run": reused_this_run,
        "index_path": str(index_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "next_gate": (
            "add_acoustic_f0_voicing_heads"
            if status == "pass"
            else "fix_pitch_target_cache_contract"
        ),
    }
    _atomic_json(report_path, report)
    _atomic_json(progress_path, report)
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
    )
    parser.add_argument(
        "--checkpoint-reserve-seconds",
        type=float,
        default=DEFAULT_CHECKPOINT_RESERVE_SECONDS,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_pitch_target_cache(
                args.root,
                time_budget_seconds=args.time_budget_seconds,
                checkpoint_reserve_seconds=args.checkpoint_reserve_seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
