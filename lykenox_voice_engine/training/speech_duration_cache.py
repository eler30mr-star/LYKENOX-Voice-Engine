"""Generate and audit real token-duration caches from a trained LYKENOX aligner."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from lykenox_voice_engine.core.ctc_alignment import (
    ctc_targets,
    expand_content_durations,
    forced_alignment_durations,
)
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.alignment_artifact import (
    checkpoint_sha256,
    load_aligner_checkpoint,
)
from lykenox_voice_engine.training.speech_aligner_train import _dataset


DURATION_CACHE_VERSION = "alignment-v1"


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return int(ordered[index])


def generate_duration_cache(
    root: Path,
    checkpoint_path: Path,
    *,
    nonpause_warn_frames: int = 100,
) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
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
    split_reports: dict[str, object] = {}
    all_nonpause_durations: list[int] = []
    all_content_durations: list[int] = []
    suspicious_utterances: list[dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for split in ("train", "val"):
            dataset = _dataset(root, split, speech_config)
            split_dir = duration_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            index_path = split_dir / "index.jsonl"
            split_failures: list[dict[str, object]] = []
            split_scores: list[float] = []
            generated = 0

            with index_path.open("w", encoding="utf-8") as index_file:
                for item_index in range(len(dataset)):
                    item = dataset[item_index]
                    utterance_id = str(item["utterance_id"])
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
                        full = expand_content_durations(
                            token_ids,
                            alignment.target_durations,
                            positions,
                        )
                        if int(full.sum().item()) != int(mel.shape[0]):
                            raise RuntimeError("Duration sum does not match mel frame count")
                        if not bool((alignment.target_durations > 0).all().item()):
                            raise RuntimeError("At least one aligned content token has zero duration")

                        token_values = token_ids.tolist()
                        duration_values = full.tolist()
                        content_rows: list[dict[str, object]] = []
                        utterance_nonpause_max = 0
                        for position in positions:
                            token_id = int(token_values[position])
                            token_name = id_to_token.get(token_id, "<unknown>")
                            duration = int(duration_values[position])
                            all_content_durations.append(duration)
                            if token_name not in pause_names:
                                all_nonpause_durations.append(duration)
                                utterance_nonpause_max = max(utterance_nonpause_max, duration)
                            content_rows.append(
                                {
                                    "position": int(position),
                                    "token": token_name,
                                    "token_id": token_id,
                                    "duration_frames": duration,
                                }
                            )

                        if utterance_nonpause_max > nonpause_warn_frames:
                            suspicious_utterances.append(
                                {
                                    "split": split,
                                    "utterance_id": utterance_id,
                                    "max_nonpause_duration_frames": utterance_nonpause_max,
                                }
                            )

                        record = {
                            "cache_version": DURATION_CACHE_VERSION,
                            "frontend_version": frontend.version,
                            "checkpoint_sha256": digest,
                            "utterance_id": utterance_id,
                            "text": str(item["text"]),
                            "mel_frames": int(mel.shape[0]),
                            "token_ids": [int(value) for value in token_values],
                            "durations": [int(value) for value in duration_values],
                            "content": content_rows,
                            "alignment_score_per_step": float(alignment.score_per_step),
                        }
                        target_path = split_dir / f"{utterance_id}.pt"
                        torch.save(record, target_path)
                        index_file.write(
                            json.dumps(
                                {
                                    "utterance_id": utterance_id,
                                    "path": str(target_path),
                                    "mel_frames": int(mel.shape[0]),
                                    "content_tokens": len(content_rows),
                                    "alignment_score_per_step": round(
                                        alignment.score_per_step, 6
                                    ),
                                    "max_nonpause_duration_frames": utterance_nonpause_max,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        split_scores.append(float(alignment.score_per_step))
                        generated += 1
                    except Exception as error:
                        failure = {
                            "split": split,
                            "utterance_id": utterance_id,
                            "error": f"{type(error).__name__}: {error}",
                        }
                        split_failures.append(failure)
                        overall_failures.append(failure)

            split_reports[split] = {
                "items": len(dataset),
                "generated": generated,
                "failures": split_failures[:20],
                "mean_alignment_score_per_step": (
                    round(statistics.fmean(split_scores), 6) if split_scores else None
                ),
                "index": str(index_path),
                "pass": generated == len(dataset) and not split_failures,
            }

    frame_ms = speech_config.hop_length / speech_config.sample_rate * 1000.0
    report = {
        "status": "pass" if not overall_failures else "needs_review",
        "frontend_version": frontend.version,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "checkpoint_epoch": payload.get("epoch"),
        "duration_cache_root": str(duration_root),
        "duration_cache_version": DURATION_CACHE_VERSION,
        "frame_ms": round(frame_ms, 6),
        "splits": split_reports,
        "content_duration_frames": {
            "count": len(all_content_durations),
            "median": (
                round(statistics.median(all_content_durations), 2)
                if all_content_durations
                else None
            ),
            "p95": _percentile(all_content_durations, 0.95),
            "max": max(all_content_durations) if all_content_durations else None,
        },
        "nonpause_duration_frames": {
            "count": len(all_nonpause_durations),
            "median": (
                round(statistics.median(all_nonpause_durations), 2)
                if all_nonpause_durations
                else None
            ),
            "p95": _percentile(all_nonpause_durations, 0.95),
            "max": max(all_nonpause_durations) if all_nonpause_durations else None,
            "warning_threshold": nonpause_warn_frames,
        },
        "suspicious_utterance_count": len(suspicious_utterances),
        "suspicious_utterances": suspicious_utterances[:30],
        "failures": overall_failures[:30],
        "next_gate": (
            "audit_duration_distribution_then_aligned_acoustic_smoke"
            if not overall_failures
            else "review_failed_alignments"
        ),
    }
    report_path = duration_root / "duration_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nonpause-warn-frames", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_duration_cache(
                args.root,
                args.checkpoint,
                nonpause_warn_frames=args.nonpause_warn_frames,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
