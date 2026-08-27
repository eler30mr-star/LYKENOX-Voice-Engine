"""Generate and audit real token-duration caches from a trained LYKENOX aligner."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lykenox_voice_engine.core.ctc_alignment import ctc_targets, forced_alignment_durations
from lykenox_voice_engine.core.ctc_timing import expand_alignment_timing_durations
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.alignment_artifact import (
    checkpoint_sha256,
    load_aligner_checkpoint,
)
from lykenox_voice_engine.training.speech_aligner_train import _dataset


DURATION_CACHE_VERSION = "alignment-v3"
BOUNDARY_BLANK_POLICY = "leading_to_bos_trailing_to_eos"
INTERIOR_BLANK_POLICY = "word_boundary_to_wb_pause_to_pause_intra_word_split_neighbors"


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return int(ordered[index])


def _duration_stats(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "median": round(statistics.median(values), 2) if values else None,
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _record_is_reusable(
    record: object,
    *,
    frontend_version: str,
    checkpoint_sha256_value: str,
    utterance_id: str,
    text: str,
    token_ids: list[int],
) -> bool:
    if not isinstance(record, dict):
        return False
    return (
        record.get("cache_version") == DURATION_CACHE_VERSION
        and record.get("frontend_version") == frontend_version
        and record.get("checkpoint_sha256") == checkpoint_sha256_value
        and record.get("interior_blank_policy") == INTERIOR_BLANK_POLICY
        and str(record.get("utterance_id")) == utterance_id
        and str(record.get("text", "")) == text
        and [int(value) for value in record.get("token_ids", [])] == token_ids
        and isinstance(record.get("durations"), list)
        and isinstance(record.get("content"), list)
        and isinstance(record.get("boundary_frames"), dict)
        and isinstance(record.get("timing_frames"), dict)
    )


def _consume_record(
    record: dict[str, object],
    *,
    split: str,
    pause_names: set[str],
    nonpause_warn_frames: int,
    all_nonpause_durations: list[int],
    all_content_durations: list[int],
    leading_boundary_durations: list[int],
    trailing_boundary_durations: list[int],
    word_boundary_blank_frames: list[int],
    pause_blank_frames: list[int],
    neighbor_split_blank_frames: list[int],
    suspicious_utterances: list[dict[str, object]],
) -> tuple[float, int]:
    boundary = dict(record.get("boundary_frames", {}))
    leading_boundary_durations.append(int(boundary.get("leading", 0)))
    trailing_boundary_durations.append(int(boundary.get("trailing", 0)))

    timing = dict(record.get("timing_frames", {}))
    word_boundary_blank_frames.append(int(timing.get("word_boundary_blank", 0)))
    pause_blank_frames.append(int(timing.get("pause_blank", 0)))
    neighbor_split_blank_frames.append(int(timing.get("neighbor_split_blank", 0)))

    utterance_nonpause_max = 0
    for content_row in record.get("content", []):
        row = dict(content_row)
        token_name = str(row.get("token", "<unknown>"))
        duration = int(row.get("duration_frames", 0))
        all_content_durations.append(duration)
        if token_name not in pause_names:
            all_nonpause_durations.append(duration)
            utterance_nonpause_max = max(utterance_nonpause_max, duration)

    if utterance_nonpause_max > nonpause_warn_frames:
        suspicious_utterances.append(
            {
                "split": split,
                "utterance_id": str(record.get("utterance_id")),
                "max_nonpause_duration_frames": utterance_nonpause_max,
            }
        )
    return float(record.get("alignment_score_per_step", 0.0)), utterance_nonpause_max


def _index_row(
    record: dict[str, object],
    target_path: Path,
    max_nonpause_duration_frames: int,
) -> dict[str, object]:
    boundary = dict(record.get("boundary_frames", {}))
    timing = dict(record.get("timing_frames", {}))
    return {
        "utterance_id": str(record.get("utterance_id")),
        "path": str(target_path),
        "mel_frames": int(record.get("mel_frames", 0)),
        "content_tokens": len(record.get("content", [])),
        "leading_boundary_frames": int(boundary.get("leading", 0)),
        "trailing_boundary_frames": int(boundary.get("trailing", 0)),
        "word_boundary_blank_frames": int(timing.get("word_boundary_blank", 0)),
        "pause_blank_frames": int(timing.get("pause_blank", 0)),
        "neighbor_split_blank_frames": int(timing.get("neighbor_split_blank", 0)),
        "alignment_score_per_step": round(
            float(record.get("alignment_score_per_step", 0.0)), 6
        ),
        "max_nonpause_duration_frames": max_nonpause_duration_frames,
    }


def generate_duration_cache(
    root: Path,
    checkpoint_path: Path,
    *,
    nonpause_warn_frames: int = 100,
    time_budget_seconds: float | None = None,
    progress_every: int = 10,
    resume: bool = True,
) -> dict[str, object]:
    """Generate word-boundary-safe durations, safely resumable after interruption."""

    if time_budget_seconds is not None and time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be positive when provided")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")

    started = time.perf_counter()
    root = Path(root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    model, payload = load_aligner_checkpoint(checkpoint_path)
    frontend = SpanishTextFrontend()
    speech_config = LykenoxSpeechConfig()
    digest = checkpoint_sha256(checkpoint_path)
    vocab = frontend.vocabulary()
    id_to_token = {token_id: token for token, token_id in vocab.items()}
    pause_names = {"<pau_short>", "<pau_long>"}

    duration_root = (
        root
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "features"
        / "speech"
        / DURATION_CACHE_VERSION
        / frontend.version
        / digest[:16]
    )
    duration_root.mkdir(parents=True, exist_ok=True)

    overall_failures: list[dict[str, object]] = []
    pending_items: list[dict[str, object]] = []
    split_reports: dict[str, object] = {}
    all_nonpause_durations: list[int] = []
    all_content_durations: list[int] = []
    leading_boundary_durations: list[int] = []
    trailing_boundary_durations: list[int] = []
    word_boundary_blank_totals: list[int] = []
    pause_blank_totals: list[int] = []
    neighbor_split_blank_totals: list[int] = []
    suspicious_utterances: list[dict[str, object]] = []
    total_new_generated = 0
    total_reused = 0

    model.eval()
    with torch.no_grad():
        for split in ("train", "val"):
            dataset = _dataset(root, split, speech_config)
            split_dir = duration_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            index_path = split_dir / "index.jsonl"
            split_failures: list[dict[str, object]] = []
            split_scores: list[float] = []
            generated = reused = new_generated = pending = 0
            index_rows: list[dict[str, object]] = []

            for item_index, source_row in enumerate(dataset.rows):
                utterance_id = str(source_row.utterance_id)
                text = str(source_row.text)
                token_values = [int(value) for value in frontend.encode(text)]
                target_path = split_dir / f"{utterance_id}.pt"
                record: dict[str, object] | None = None

                if resume and target_path.exists():
                    try:
                        existing = torch.load(target_path, map_location="cpu", weights_only=False)
                        if _record_is_reusable(
                            existing,
                            frontend_version=frontend.version,
                            checkpoint_sha256_value=digest,
                            utterance_id=utterance_id,
                            text=text,
                            token_ids=token_values,
                        ):
                            record = existing
                    except Exception:
                        record = None

                if record is None:
                    if time_budget_seconds is not None and time.perf_counter() - started >= time_budget_seconds:
                        pending += 1
                        pending_items.append({"split": split, "utterance_id": utterance_id})
                        continue

                    item = dataset[item_index]
                    mel = item["mel"]
                    token_ids = item["token_ids"]
                    try:
                        targets, positions = ctc_targets(token_ids)
                        logits = model(mel.unsqueeze(0)).squeeze(0)
                        log_probs = F.log_softmax(logits, dim=-1)
                        alignment = forced_alignment_durations(
                            log_probs,
                            targets,
                            model.config.blank_id,
                            mel_frames=int(mel.shape[0]),
                            frame_stride=model.config.frame_stride,
                        )
                        timing = expand_alignment_timing_durations(
                            token_ids,
                            positions,
                            alignment,
                            frame_stride=model.config.frame_stride,
                        )
                        full = timing.durations
                        if int(full.sum().item()) != int(mel.shape[0]):
                            raise RuntimeError("Duration sum does not match mel frame count")
                        if not bool((timing.direct_target_frames > 0).all().item()):
                            raise RuntimeError("At least one acoustic target has zero direct occupancy")

                        duration_values = [int(value) for value in full.tolist()]
                        content_rows: list[dict[str, object]] = []
                        for position in positions:
                            token_id = int(token_values[position])
                            content_rows.append(
                                {
                                    "position": int(position),
                                    "token": id_to_token.get(token_id, "<unknown>"),
                                    "token_id": token_id,
                                    "duration_frames": int(duration_values[position]),
                                }
                            )

                        record = {
                            "cache_version": DURATION_CACHE_VERSION,
                            "frontend_version": frontend.version,
                            "checkpoint_sha256": digest,
                            "utterance_id": utterance_id,
                            "text": text,
                            "mel_frames": int(mel.shape[0]),
                            "token_ids": token_values,
                            "durations": duration_values,
                            "boundary_blank_policy": BOUNDARY_BLANK_POLICY,
                            "interior_blank_policy": INTERIOR_BLANK_POLICY,
                            "boundary_frames": {
                                "leading": timing.leading_boundary_frames,
                                "trailing": timing.trailing_boundary_frames,
                            },
                            "timing_frames": {
                                "word_boundary_blank": timing.word_boundary_blank_frames,
                                "pause_blank": timing.pause_blank_frames,
                                "neighbor_split_blank": timing.neighbor_split_blank_frames,
                            },
                            "content": content_rows,
                            "alignment_score_per_step": float(alignment.score_per_step),
                        }
                        torch.save(record, target_path)
                        new_generated += 1
                        total_new_generated += 1
                        if total_new_generated % progress_every == 0:
                            print(
                                f"[LYKENOX durations v3] new={total_new_generated} reused={total_reused} split={split}",
                                file=sys.stderr,
                                flush=True,
                            )
                    except Exception as error:
                        failure = {
                            "split": split,
                            "utterance_id": utterance_id,
                            "error": f"{type(error).__name__}: {error}",
                        }
                        split_failures.append(failure)
                        overall_failures.append(failure)
                        continue
                else:
                    reused += 1
                    total_reused += 1

                score, utterance_nonpause_max = _consume_record(
                    record,
                    split=split,
                    pause_names=pause_names,
                    nonpause_warn_frames=nonpause_warn_frames,
                    all_nonpause_durations=all_nonpause_durations,
                    all_content_durations=all_content_durations,
                    leading_boundary_durations=leading_boundary_durations,
                    trailing_boundary_durations=trailing_boundary_durations,
                    word_boundary_blank_frames=word_boundary_blank_totals,
                    pause_blank_frames=pause_blank_totals,
                    neighbor_split_blank_frames=neighbor_split_blank_totals,
                    suspicious_utterances=suspicious_utterances,
                )
                split_scores.append(score)
                generated += 1
                index_rows.append(_index_row(record, target_path, utterance_nonpause_max))

            with index_path.open("w", encoding="utf-8") as index_file:
                for index_row in index_rows:
                    index_file.write(json.dumps(index_row, ensure_ascii=False) + "\n")

            split_reports[split] = {
                "items": len(dataset),
                "generated": generated,
                "reused": reused,
                "new_generated": new_generated,
                "pending": pending,
                "failures": split_failures[:20],
                "mean_alignment_score_per_step": (
                    round(statistics.fmean(split_scores), 6) if split_scores else None
                ),
                "index": str(index_path),
                "pass": generated == len(dataset) and pending == 0 and not split_failures,
            }

    frame_ms = speech_config.hop_length / speech_config.sample_rate * 1000.0
    pending_count = len(pending_items)
    if overall_failures:
        status = "needs_review"
        next_gate = "review_failed_alignments"
    elif pending_count:
        status = "incomplete"
        next_gate = "rerun_same_command_to_resume"
    else:
        status = "pass"
        next_gate = "audit_duration_distribution_then_aligned_acoustic_smoke"

    report = {
        "status": status,
        "frontend_version": frontend.version,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "checkpoint_epoch": payload.get("epoch"),
        "duration_cache_root": str(duration_root),
        "duration_cache_version": DURATION_CACHE_VERSION,
        "boundary_blank_policy": BOUNDARY_BLANK_POLICY,
        "interior_blank_policy": INTERIOR_BLANK_POLICY,
        "resume_enabled": resume,
        "time_budget_seconds": time_budget_seconds,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "reused_records": total_reused,
        "new_records_generated": total_new_generated,
        "pending_item_count": pending_count,
        "pending_items": pending_items[:30],
        "frame_ms": round(frame_ms, 6),
        "splits": split_reports,
        "content_duration_frames": _duration_stats(all_content_durations),
        "nonpause_duration_frames": {
            **_duration_stats(all_nonpause_durations),
            "warning_threshold": nonpause_warn_frames,
        },
        "leading_boundary_frames": _duration_stats(leading_boundary_durations),
        "trailing_boundary_frames": _duration_stats(trailing_boundary_durations),
        "word_boundary_blank_frames": _duration_stats(word_boundary_blank_totals),
        "pause_blank_frames": _duration_stats(pause_blank_totals),
        "neighbor_split_blank_frames": _duration_stats(neighbor_split_blank_totals),
        "suspicious_utterance_count": len(suspicious_utterances),
        "suspicious_utterances": suspicious_utterances[:30],
        "failures": overall_failures[:30],
        "next_gate": next_gate,
    }
    report_path = duration_root / "duration_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (duration_root / "duration_progress.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nonpause-warn-frames", type=int, default=100)
    parser.add_argument("--time-budget-seconds", type=float)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            generate_duration_cache(
                args.root,
                args.checkpoint,
                nonpause_warn_frames=args.nonpause_warn_frames,
                time_budget_seconds=args.time_budget_seconds,
                progress_every=args.progress_every,
                resume=not args.no_resume,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
