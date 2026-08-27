"""Fast forensic review of LYKENOX speech duration-cache outliers.

This command performs no model inference and no training. It reads the latest generated
alignment cache, classifies long non-pause durations by token and utterance position,
and produces a compact report before acoustic-model training is allowed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig


PAUSE_TOKENS = {"<pau_short>", "<pau_long>"}


def _latest_duration_root(root: Path) -> Path:
    base = (
        Path(root).resolve()
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "features"
        / "speech"
    )
    reports = list(base.rglob("duration_audit.json"))
    if not reports:
        raise FileNotFoundError(f"No duration_audit.json found under {base}")
    report = max(reports, key=lambda path: path.stat().st_mtime_ns)
    return report.parent


def _record_paths(duration_root: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    for split in ("train", "val"):
        index_path = duration_root / split / "index.jsonl"
        if not index_path.exists():
            continue
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(str(row["path"]))
            if not path.is_absolute():
                path = (duration_root / split / path).resolve()
            records.append((split, path))
    if not records:
        raise RuntimeError(f"No cached duration records found in {duration_root}")
    return records


def _boundary_role(content: list[dict[str, object]], index: int) -> str:
    nonpause = [
        i for i, row in enumerate(content)
        if str(row.get("token")) not in PAUSE_TOKENS
    ]
    if not nonpause:
        return "no_nonpause"
    first = nonpause[0]
    last = nonpause[-1]
    if index == first and index == last:
        return "only_nonpause"
    if index == first:
        return "first_nonpause"
    if index == last:
        return "last_nonpause"
    return "interior"


def classify_boundary_pattern(boundary_count: int, interior_count: int) -> str:
    total = boundary_count + interior_count
    if total == 0:
        return "no_long_nonpause_outliers"
    ratio = boundary_count / total
    if boundary_count >= 3 and ratio >= 0.70:
        return "boundary_silence_absorption_likely"
    if interior_count > 0 and boundary_count > 0:
        return "mixed_boundary_and_interior_outliers"
    if interior_count > 0:
        return "interior_alignment_outliers"
    return "boundary_outliers_only"


def _diagnosis_for_cache(
    raw_diagnosis: str,
    cache_version: str,
    outlier_count: int,
) -> tuple[str, str]:
    if outlier_count == 0:
        return "duration_distribution_clean", "aligned_acoustic_smoke"
    boundary_heavy = raw_diagnosis in {
        "boundary_silence_absorption_likely",
        "boundary_outliers_only",
    }
    if cache_version == "alignment-v2" and boundary_heavy:
        return "residual_boundary_alignment_outliers", "inspect_residual_boundary_outliers"
    if cache_version == "alignment-v1" and boundary_heavy:
        return raw_diagnosis, "fix_boundary_blank_assignment"
    return raw_diagnosis, "inspect_interior_alignment_outliers"


def review_duration_outliers(
    root: Path,
    *,
    duration_root: Path | None = None,
    threshold_frames: int = 100,
) -> dict[str, object]:
    if threshold_frames < 1:
        raise ValueError("threshold_frames must be >= 1")

    root = Path(root).resolve()
    duration_root = Path(duration_root).resolve() if duration_root else _latest_duration_root(root)
    frame_ms = LykenoxSpeechConfig().hop_length / LykenoxSpeechConfig().sample_rate * 1000.0

    token_durations: dict[str, list[int]] = defaultdict(list)
    loaded: list[tuple[str, dict[str, object]]] = []
    cache_versions: Counter[str] = Counter()
    leading_boundaries: list[int] = []
    trailing_boundaries: list[int] = []
    for split, path in _record_paths(duration_root):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid duration record: {path}")
        loaded.append((split, record))
        cache_versions[str(record.get("cache_version", "unknown"))] += 1
        boundary = record.get("boundary_frames", {})
        if isinstance(boundary, dict):
            leading_boundaries.append(int(boundary.get("leading", 0)))
            trailing_boundaries.append(int(boundary.get("trailing", 0)))
        for row in record.get("content", []):
            token = str(row.get("token"))
            duration = int(row.get("duration_frames", 0))
            if token not in PAUSE_TOKENS:
                token_durations[token].append(duration)

    cache_version = cache_versions.most_common(1)[0][0] if cache_versions else "unknown"
    token_stats: dict[str, dict[str, object]] = {}
    for token, values in sorted(token_durations.items()):
        ordered = sorted(values)
        p95_index = round((len(ordered) - 1) * 0.95)
        token_stats[token] = {
            "count": len(values),
            "median_frames": round(statistics.median(values), 2),
            "p95_frames": int(ordered[p95_index]),
            "max_frames": max(values),
        }

    outliers: list[dict[str, object]] = []
    role_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    utterance_ids: set[str] = set()

    for split, record in loaded:
        content = list(record.get("content", []))
        for index, row in enumerate(content):
            token = str(row.get("token"))
            duration = int(row.get("duration_frames", 0))
            if token in PAUSE_TOKENS or duration <= threshold_frames:
                continue
            role = _boundary_role(content, index)
            role_counts[role] += 1
            token_counts[token] += 1
            utterance_id = str(record.get("utterance_id"))
            utterance_ids.add(utterance_id)
            stats = token_stats.get(token, {})
            median = float(stats.get("median_frames", 0.0) or 0.0)
            ratio_to_token_median = duration / median if median > 0 else None
            outliers.append(
                {
                    "split": split,
                    "utterance_id": utterance_id,
                    "text": str(record.get("text", "")),
                    "token": token,
                    "content_index": index,
                    "boundary_role": role,
                    "duration_frames": duration,
                    "duration_ms": round(duration * frame_ms, 2),
                    "token_median_frames": stats.get("median_frames"),
                    "token_p95_frames": stats.get("p95_frames"),
                    "ratio_to_token_median": (
                        round(ratio_to_token_median, 2)
                        if ratio_to_token_median is not None
                        else None
                    ),
                    "alignment_score_per_step": round(
                        float(record.get("alignment_score_per_step", 0.0)), 6
                    ),
                    "boundary_frames": record.get("boundary_frames"),
                }
            )

    outliers.sort(key=lambda row: int(row["duration_frames"]), reverse=True)
    boundary_count = (
        role_counts["first_nonpause"]
        + role_counts["last_nonpause"]
        + role_counts["only_nonpause"]
    )
    interior_count = role_counts["interior"]
    raw_diagnosis = classify_boundary_pattern(boundary_count, interior_count)
    diagnosis, next_gate = _diagnosis_for_cache(raw_diagnosis, cache_version, len(outliers))

    report = {
        "status": "pass" if not outliers else "review_required",
        "duration_root": str(duration_root),
        "cache_version": cache_version,
        "threshold_frames": threshold_frames,
        "threshold_ms": round(threshold_frames * frame_ms, 2),
        "records_loaded": len(loaded),
        "outlier_token_count": len(outliers),
        "outlier_utterance_count": len(utterance_ids),
        "boundary_outlier_token_count": boundary_count,
        "interior_outlier_token_count": interior_count,
        "boundary_fraction": (
            round(boundary_count / len(outliers), 4) if outliers else 0.0
        ),
        "raw_pattern": raw_diagnosis,
        "diagnosis": diagnosis,
        "role_counts": dict(role_counts),
        "most_common_outlier_tokens": token_counts.most_common(12),
        "top_outliers": outliers[:40],
        "token_stats": token_stats,
        "boundary_frame_stats": {
            "leading_median": round(statistics.median(leading_boundaries), 2) if leading_boundaries else None,
            "leading_max": max(leading_boundaries) if leading_boundaries else None,
            "trailing_median": round(statistics.median(trailing_boundaries), 2) if trailing_boundaries else None,
            "trailing_max": max(trailing_boundaries) if trailing_boundaries else None,
        },
        "algorithmic_policy": (
            "alignment-v2 preserves leading CTC blank frames on BOS and trailing blank "
            "frames on EOS instead of assigning them to spoken phonemes."
            if cache_version == "alignment-v2"
            else "alignment-v1 folds boundary blank runs into neighboring spoken tokens."
        ),
        "next_gate": next_gate,
    }
    report_path = duration_root / "duration_outlier_review.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--duration-root", type=Path)
    parser.add_argument("--threshold-frames", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            review_duration_outliers(
                args.root,
                duration_root=args.duration_root,
                threshold_frames=args.threshold_frames,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
