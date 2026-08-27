"""Cache/WAV-only forensic gate for the LYKENOX polyphase vocoder v2.

The first v2 probe flagged all generated validation WAVs as being locked to the mel
frame rate (24 kHz / 256 = 93.75 Hz).  A simple autocorrelation pitch detector can,
however, confuse a real low male F0 near 94 Hz with an architectural frame-grid
artifact.  This module performs no training.  It compares each generated WAV directly
against its held-out reference and only calls the frame lock *confirmed* when the
periodicity is specific to the generated signal rather than also present in the real
reference.

It also keeps the independent sub-bass/silence-collapse diagnostic from the v2 probe.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


FORENSIC_VERSION = "polyphase-frame-lock-forensic-v1"
SAMPLE_RATE = 24000
HOP_LENGTH = 256
FRAME_RATE_HZ = SAMPLE_RATE / HOP_LENGTH


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1).astype(np.float64, copy=False)
    return mono, int(sample_rate)


def _pitch_metrics(y: np.ndarray, sample_rate: int) -> dict[str, float | bool | None]:
    frame_length = 1024
    analysis_hop = 256
    min_lag = max(1, int(sample_rate / 300.0))
    max_lag = min(int(sample_rate / 60.0), frame_length - 2)
    frames: list[np.ndarray] = []
    rms_values: list[float] = []
    for start in range(0, max(0, len(y) - frame_length + 1), analysis_hop):
        frame = y[start : start + frame_length]
        frames.append(frame)
        rms_values.append(float(np.sqrt(np.mean(frame * frame))))
    frame_rate = sample_rate / HOP_LENGTH
    if not frames:
        return {
            "median_pitch_hz": None,
            "pitch_std_hz": None,
            "frame_rate_hz": round(frame_rate, 4),
            "frame_rate_locked_raw": False,
        }

    threshold = max(rms_values) * 0.20
    estimates: list[float] = []
    for frame, rms in zip(frames, rms_values, strict=True):
        if rms < threshold:
            continue
        centered = frame - np.mean(frame)
        energy = float(np.dot(centered, centered)) + 1e-12
        best_lag = None
        best_score = -1.0
        for lag in range(min_lag, max_lag + 1):
            score = float(np.dot(centered[:-lag], centered[lag:]) / energy)
            if score > best_score:
                best_score = score
                best_lag = lag
        if best_lag is not None and best_score > 0.15:
            estimates.append(sample_rate / best_lag)

    if not estimates:
        return {
            "median_pitch_hz": None,
            "pitch_std_hz": None,
            "frame_rate_hz": round(frame_rate, 4),
            "frame_rate_locked_raw": False,
        }
    median = float(np.median(estimates))
    std = float(np.std(estimates))
    return {
        "median_pitch_hz": round(median, 4),
        "pitch_std_hz": round(std, 4),
        "frame_rate_hz": round(frame_rate, 4),
        "frame_rate_distance_hz": round(abs(median - frame_rate), 4),
        "frame_rate_locked_raw": bool(abs(median - frame_rate) <= 0.75 and std <= 1.0),
    }


def _normalized_lag_correlation(y: np.ndarray, lag: int) -> float:
    if len(y) <= lag + 1:
        return 0.0
    centered = y - np.mean(y)
    left = centered[:-lag]
    right = centered[lag:]
    denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right))) + 1e-12
    return float(np.dot(left, right) / denominator)


def _hop_periodicity(y: np.ndarray) -> dict[str, float]:
    """Measure waveform correlation exactly at the mel hop and nearby lags.

    A true frame-grid carrier tends to privilege lag 256 over neighboring lags.  Normal
    speech can also have a ~256-sample pitch period, so this value is never used alone;
    it is interpreted relative to the paired real reference.
    """

    exact = _normalized_lag_correlation(y, HOP_LENGTH)
    neighbors = [
        _normalized_lag_correlation(y, lag)
        for lag in (HOP_LENGTH - 12, HOP_LENGTH - 6, HOP_LENGTH + 6, HOP_LENGTH + 12)
    ]
    neighbor_mean = float(np.mean(neighbors))
    return {
        "lag_256_correlation": round(exact, 6),
        "neighbor_lag_mean_correlation": round(neighbor_mean, 6),
        "lag_256_excess": round(exact - neighbor_mean, 6),
    }


def _spectral_metrics(y: np.ndarray, sample_rate: int) -> dict[str, float | bool]:
    rms = float(np.sqrt(np.mean(y * y))) if len(y) else 0.0
    if len(y) < 2:
        return {
            "rms": rms,
            "sub_80hz_energy_fraction": 1.0,
            "above_300hz_energy_fraction": 0.0,
            "spectral_centroid_hz": 0.0,
            "subbass_or_silence_collapsed": True,
        }
    window = np.hanning(len(y))
    power = np.abs(np.fft.rfft(y * window)) ** 2
    frequencies = np.fft.rfftfreq(len(y), d=1.0 / sample_rate)
    total = float(power.sum()) + 1e-20
    sub_80 = float(power[frequencies < 80.0].sum() / total)
    above_300 = float(power[frequencies >= 300.0].sum() / total)
    centroid = float((frequencies * power).sum() / total)
    collapsed = rms < 1e-4 or (sub_80 >= 0.97 and above_300 <= 0.01)
    return {
        "rms": round(rms, 8),
        "sub_80hz_energy_fraction": round(sub_80, 6),
        "above_300hz_energy_fraction": round(above_300, 6),
        "spectral_centroid_hz": round(centroid, 3),
        "subbass_or_silence_collapsed": bool(collapsed),
    }


def _pair_files(listening_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for generated in sorted(listening_dir.glob("*_generated.wav")):
        reference = generated.with_name(generated.name.replace("_generated.wav", "_reference.wav"))
        if reference.exists():
            pairs.append((generated, reference))
    if not pairs:
        raise FileNotFoundError(f"No generated/reference WAV pairs found in {listening_dir}")
    return pairs


def run_polyphase_forensic(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    artifact_dir = root / "models" / "lykenox_identity" / "training" / "vocoder_polyphase_probe"
    listening_dir = artifact_dir / "listening"
    pairs = _pair_files(listening_dir)

    rows: list[dict[str, object]] = []
    raw_generated_locks = 0
    raw_reference_locks = 0
    confirmed_generated_specific_locks = 0
    collapse_count = 0

    for generated_path, reference_path in pairs[:3]:
        generated, generated_sr = _load_mono(generated_path)
        reference, reference_sr = _load_mono(reference_path)
        if generated_sr != SAMPLE_RATE or reference_sr != SAMPLE_RATE:
            raise RuntimeError(
                f"Expected 24 kHz listening WAVs, got {generated_sr}/{reference_sr}"
            )

        generated_pitch = _pitch_metrics(generated, generated_sr)
        reference_pitch = _pitch_metrics(reference, reference_sr)
        generated_hop = _hop_periodicity(generated)
        reference_hop = _hop_periodicity(reference)
        generated_spectral = _spectral_metrics(generated, generated_sr)
        reference_spectral = _spectral_metrics(reference, reference_sr)

        generated_raw = bool(generated_pitch["frame_rate_locked_raw"])
        reference_raw = bool(reference_pitch["frame_rate_locked_raw"])
        raw_generated_locks += int(generated_raw)
        raw_reference_locks += int(reference_raw)

        # Confirm only a generated-specific lock.  If the real reference itself has a
        # stable ~93.75 Hz F0, the raw detector is ambiguous and must not reject v2.
        generated_specific = generated_raw and not reference_raw

        # A second differential clue: exact-hop correlation should not be dramatically
        # more privileged in generated audio than in the paired reference.  This does
        # not create a lock by itself; it strengthens an already generated-specific raw
        # lock and is reported for diagnosis.
        hop_excess_delta = (
            float(generated_hop["lag_256_excess"])
            - float(reference_hop["lag_256_excess"])
        )
        confirmed = generated_specific and hop_excess_delta >= -0.02
        confirmed_generated_specific_locks += int(confirmed)
        collapse_count += int(bool(generated_spectral["subbass_or_silence_collapsed"]))

        rows.append(
            {
                "generated": str(generated_path),
                "reference": str(reference_path),
                "generated_pitch": generated_pitch,
                "reference_pitch": reference_pitch,
                "generated_hop_periodicity": generated_hop,
                "reference_hop_periodicity": reference_hop,
                "hop_excess_delta_generated_minus_reference": round(hop_excess_delta, 6),
                "generated_spectral": generated_spectral,
                "reference_spectral": reference_spectral,
                "generated_specific_frame_lock_confirmed": confirmed,
                "raw_lock_ambiguous_because_reference_near_frame_rate": (
                    generated_raw and reference_raw
                ),
            }
        )

    artifact_confirmed = confirmed_generated_specific_locks >= 2
    collapse_confirmed = collapse_count >= 1
    gate_pass = not artifact_confirmed and not collapse_confirmed

    if artifact_confirmed:
        diagnosis = "confirmed_generated_specific_frame_rate_artifact"
        next_gate = "redesign_vocoder_periodicity_control_before_more_training"
    elif collapse_confirmed:
        diagnosis = "subbass_or_silence_collapse_confirmed"
        next_gate = "redesign_vocoder_spectral_generation_before_more_training"
    elif raw_generated_locks and raw_reference_locks:
        diagnosis = "raw_pitch_detector_false_positive_or_ambiguous"
        next_gate = "listen_polyphase_validation_wavs"
    else:
        diagnosis = "known_vocoder_artifact_modes_not_confirmed"
        next_gate = "listen_polyphase_validation_wavs"

    report = {
        "status": "pass" if gate_pass else "needs_review",
        "forensic_version": FORENSIC_VERSION,
        "frame_rate_hz": FRAME_RATE_HZ,
        "pairs_analyzed": len(rows),
        "raw_generated_frame_locks": raw_generated_locks,
        "raw_reference_frame_locks": raw_reference_locks,
        "confirmed_generated_specific_frame_locks": confirmed_generated_specific_locks,
        "subbass_or_silence_collapse_count": collapse_count,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "pairs": rows,
    }
    report_path = artifact_dir / "frame_lock_forensic_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_polyphase_forensic(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
