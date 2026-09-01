"""Isolated CPU-only diagnostic for pitch-v1 octave discontinuities.

This tool is deliberately outside the product/training pipeline. It reads real LYKENOX WAVs
from the prepared speech manifest, runs the existing ``extract_pitch_frames`` implementation,
counts adjacent voiced-frame transitions near 2x or 0.5x F0, and writes one simple SVG F0
plot per utterance plus a JSON report.

It does not modify pitch caches, model weights, checkpoints, manifests, or training state.
No external/pretrained model is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

import torch
import torchaudio

from lykenox_voice_engine.audio.io import load_audio
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_aligner_train import _manifest_path
from lykenox_voice_engine.training.speech_pitch import (
    PITCH_TARGET_VERSION,
    PitchFrames,
    extract_pitch_frames,
)


DIAGNOSTIC_VERSION = "lykenox-pitch-octave-jump-diagnostic-v1"
DEFAULT_SPLIT = "train"
DEFAULT_ITEMS = 5
MIN_ITEMS = 3
MAX_ITEMS = 5
OCTAVE_RATIO_TOLERANCE = 0.05


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_manifest_rows(manifest: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if not row.get("utterance_id") or not row.get("wav_path"):
                continue
            rows.append({key: str(value) for key, value in row.items() if key is not None})
    if not rows:
        raise RuntimeError(f"No usable rows in manifest: {manifest}")
    return rows


def _evenly_spaced_indices(count: int, requested: int) -> list[int]:
    if count < 1:
        raise ValueError("count must be positive")
    if requested < 1:
        raise ValueError("requested must be positive")
    selected = min(count, requested)
    if selected == 1:
        return [0]
    return [round(index * (count - 1) / (selected - 1)) for index in range(selected)]


def _resolve_wav_path(manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (manifest.parent / path).resolve()
    return path


def _load_pipeline_mono(path: Path, *, sample_rate: int) -> torch.Tensor:
    waveform, observed_rate = load_audio(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if observed_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, observed_rate, sample_rate)
    peak = waveform.abs().max().clamp_min(1e-8)
    if peak > 1.0:
        waveform = waveform / peak
    return waveform.squeeze(0).to(torch.float32).contiguous()


def detect_octave_jumps(
    f0_hz: torch.Tensor,
    *,
    tolerance_fraction: float = OCTAVE_RATIO_TOLERANCE,
) -> list[dict[str, float | int | str]]:
    """Return adjacent voiced-frame transitions near 2x or 0.5x F0."""

    if f0_hz.ndim != 1:
        raise ValueError("f0_hz must be one-dimensional")
    if not 0.0 < tolerance_fraction < 0.25:
        raise ValueError("tolerance_fraction must be in (0, 0.25)")
    values = f0_hz.detach().cpu().to(torch.float64)
    jumps: list[dict[str, float | int | str]] = []
    for index in range(1, int(values.numel())):
        previous = float(values[index - 1])
        current = float(values[index])
        if previous <= 0.0 or current <= 0.0:
            continue
        ratio = current / previous
        up_error = abs(ratio / 2.0 - 1.0)
        down_error = abs(ratio / 0.5 - 1.0)
        if up_error <= tolerance_fraction:
            direction = "up_2x"
            error = up_error
        elif down_error <= tolerance_fraction:
            direction = "down_0.5x"
            error = down_error
        else:
            continue
        jumps.append(
            {
                "frame": index,
                "previous_f0_hz": previous,
                "current_f0_hz": current,
                "ratio": ratio,
                "direction": direction,
                "relative_ratio_error": error,
            }
        )
    return jumps


def _voiced_transition_count(f0_hz: torch.Tensor) -> int:
    voiced = f0_hz.detach().cpu() > 0.0
    if voiced.numel() < 2:
        return 0
    return int((voiced[:-1] & voiced[1:]).sum().item())


def _svg_polyline_points(
    values: Iterable[float],
    *,
    width: int,
    height: int,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    y_min: float,
    y_max: float,
) -> list[list[tuple[float, float]]]:
    source = list(values)
    frame_max = max(1, len(source) - 1)
    span = max(1e-6, y_max - y_min)
    segments: list[list[tuple[float, float]]] = []
    active: list[tuple[float, float]] = []
    for index, value in enumerate(source):
        if value <= 0.0 or not math.isfinite(value):
            if len(active) >= 2:
                segments.append(active)
            active = []
            continue
        x = left + (index / frame_max) * plot_width
        y = top + (1.0 - (value - y_min) / span) * plot_height
        active.append((x, y))
    if len(active) >= 2:
        segments.append(active)
    return segments


def write_f0_svg(
    path: Path,
    *,
    utterance_id: str,
    pitch: PitchFrames,
    jumps: list[dict[str, float | int | str]],
    sample_rate: int,
    hop_length: int,
) -> None:
    """Write a dependency-free F0 plot; voiced gaps remain visually disconnected."""

    values = [float(value) for value in pitch.f0_hz.detach().cpu().tolist()]
    voiced_values = [value for value in values if value > 0.0 and math.isfinite(value)]
    y_min = max(0.0, min(voiced_values) * 0.85) if voiced_values else 0.0
    y_max = max(voiced_values) * 1.15 if voiced_values else 1.0
    width, height = 1200, 420
    left, right, top, bottom = 80, 30, 50, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    duration = max(0.0, (len(values) - 1) * hop_length / float(sample_rate))
    segments = _svg_polyline_points(
        values,
        width=width,
        height=height,
        left=left,
        top=top,
        plot_width=plot_width,
        plot_height=plot_height,
        y_min=y_min,
        y_max=y_max,
    )
    frame_max = max(1, len(values) - 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="18">{utterance_id} — {PITCH_TARGET_VERSION}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="10" y="{top + 8}" font-family="sans-serif" font-size="12">{y_max:.1f} Hz</text>',
        f'<text x="10" y="{top + plot_height}" font-family="sans-serif" font-size="12">{y_min:.1f} Hz</text>',
        f'<text x="{left}" y="{height - 20}" font-family="sans-serif" font-size="12">0.0 s</text>',
        f'<text x="{left + plot_width - 55}" y="{height - 20}" font-family="sans-serif" font-size="12">{duration:.2f} s</text>',
    ]
    for segment in segments:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in segment)
        lines.append(f'<polyline points="{points}" fill="none" stroke="#1f5aa6" stroke-width="1.5"/>')
    for jump in jumps:
        frame = int(jump["frame"])
        x = left + (frame / frame_max) * plot_width
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#c62828" stroke-width="1" opacity="0.8"/>'
        )
    lines.append(
        f'<text x="{left + 8}" y="{top + 20}" font-family="sans-serif" font-size="12" fill="#c62828">octave-like jumps: {len(jumps)}</text>'
    )
    lines.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pitch_octave_diagnostic(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    output_dir: Path | None = None,
    tolerance_fraction: float = OCTAVE_RATIO_TOLERANCE,
) -> dict[str, object]:
    if max_items < MIN_ITEMS or max_items > MAX_ITEMS:
        raise ValueError(f"max_items must be between {MIN_ITEMS} and {MAX_ITEMS}")
    root = Path(root).resolve()
    config = LykenoxSpeechConfig()
    manifest = _manifest_path(root, split)
    rows = _read_manifest_rows(manifest)
    indices = _evenly_spaced_indices(len(rows), max_items)
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "reports" / "pitch_octave_diagnostic_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, object]] = []
    total_jumps = 0
    total_voiced_transitions = 0
    for index in indices:
        row = rows[index]
        utterance_id = row["utterance_id"]
        wav_path = _resolve_wav_path(manifest, row["wav_path"])
        waveform = _load_pipeline_mono(wav_path, sample_rate=config.sample_rate)
        frame_count = int(waveform.numel()) // int(config.hop_length) + 1
        pitch = extract_pitch_frames(
            waveform,
            frame_count=frame_count,
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            frame_length=config.n_fft,
        )
        jumps = detect_octave_jumps(pitch.f0_hz, tolerance_fraction=tolerance_fraction)
        voiced_transitions = _voiced_transition_count(pitch.f0_hz)
        total_jumps += len(jumps)
        total_voiced_transitions += voiced_transitions
        plot_path = output_dir / f"{utterance_id}__f0_v1.svg"
        write_f0_svg(
            plot_path,
            utterance_id=utterance_id,
            pitch=pitch,
            jumps=jumps,
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
        )
        items.append(
            {
                "utterance_id": utterance_id,
                "wav_path": str(wav_path),
                "frames": int(pitch.f0_hz.numel()),
                "voiced_frames": int((pitch.f0_hz > 0.0).sum().item()),
                "voiced_adjacent_transitions": voiced_transitions,
                "octave_jump_count": len(jumps),
                "octave_jump_rate_per_voiced_transition": (
                    len(jumps) / voiced_transitions if voiced_transitions else 0.0
                ),
                "up_2x_count": sum(jump["direction"] == "up_2x" for jump in jumps),
                "down_0.5x_count": sum(jump["direction"] == "down_0.5x" for jump in jumps),
                "octave_jumps": jumps,
                "plot": str(plot_path),
            }
        )

    report: dict[str, object] = {
        "status": "diagnostic_complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "device": "cpu",
        "split": split,
        "manifest": str(manifest),
        "selected_manifest_indices": indices,
        "item_count": len(items),
        "octave_ratio_tolerance_fraction": tolerance_fraction,
        "total_octave_jump_count": total_jumps,
        "total_voiced_adjacent_transitions": total_voiced_transitions,
        "aggregate_octave_jump_rate_per_voiced_transition": (
            total_jumps / total_voiced_transitions if total_voiced_transitions else 0.0
        ),
        "items": items,
        "product_pipeline_modified": False,
        "pitch_cache_modified": False,
        "optimizer_created": False,
        "parameter_update_executed": False,
        "checkpoint_written": False,
        "next_action": "review_octave_jump_counts_and_plots_before_any_pitch_v2_change",
    }
    _atomic_json(output_dir / "pitch_octave_diagnostic_v1.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max-items", type=int, default=DEFAULT_ITEMS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--octave-tolerance", type=float, default=OCTAVE_RATIO_TOLERANCE)
    args = parser.parse_args()
    print(
        json.dumps(
            run_pitch_octave_diagnostic(
                args.root,
                split=args.split,
                max_items=args.max_items,
                output_dir=args.output_dir,
                tolerance_fraction=args.octave_tolerance,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
