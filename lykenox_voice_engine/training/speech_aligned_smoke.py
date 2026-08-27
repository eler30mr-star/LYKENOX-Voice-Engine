"""Aligned real-data acoustic smoke gate for LYKENOX Speech.

This is the first acoustic-model smoke that consumes the validated ``alignment-v3``
durations. It performs no uniform-duration fallback and never clips an utterance or its
timing targets. The gate verifies exact token/cache identity, exact mel-frame coverage,
full-frame acoustic loss, duration-predictor learning, finite gradients, and CPU timing.

Passing this command proves that the current acoustic prototype can optimize from real
LYKENOX text/mel pairs using the cleaned alignment contract. It does not prove final
intelligibility, identity quality, vocoder quality, or readiness for a long training run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_aligner_train import _dataset
from lykenox_voice_engine.training.speech_duration_cache import DURATION_CACHE_VERSION


EXPECTED_DURATION_CACHE_VERSION = "alignment-v3"


def _latest_clean_duration_root(root: Path) -> Path:
    base = (
        Path(root).resolve()
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "features"
        / "speech"
        / EXPECTED_DURATION_CACHE_VERSION
    )
    reports = sorted(
        base.rglob("duration_audit.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for report_path in reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            report.get("status") == "pass"
            and report.get("duration_cache_version") == EXPECTED_DURATION_CACHE_VERSION
            and int(report.get("suspicious_utterance_count", 0)) == 0
        ):
            return report_path.parent
    raise FileNotFoundError(
        f"No clean {EXPECTED_DURATION_CACHE_VERSION} duration cache found under {base}"
    )


def _duration_record_paths(duration_root: Path, split: str) -> dict[str, Path]:
    index_path = duration_root / split / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Duration index not found: {index_path}")
    records: dict[str, Path] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        utterance_id = str(row["utterance_id"])
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = (index_path.parent / path).resolve()
        records[utterance_id] = path
    return records


def validate_aligned_record(
    record: object,
    *,
    utterance_id: str,
    text: str,
    token_ids: torch.Tensor,
    mel_frames: int,
) -> torch.Tensor:
    """Validate one cached timing record and return exact teacher durations."""

    if not isinstance(record, dict):
        raise RuntimeError(f"Invalid aligned duration record for {utterance_id}")
    if record.get("cache_version") != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError(
            f"Wrong duration cache version for {utterance_id}: {record.get('cache_version')}"
        )
    if str(record.get("utterance_id")) != utterance_id:
        raise RuntimeError(f"Duration record utterance mismatch for {utterance_id}")
    if str(record.get("text", "")) != text:
        raise RuntimeError(f"Duration record text mismatch for {utterance_id}")

    expected_tokens = [int(value) for value in token_ids.detach().cpu().tolist()]
    cached_tokens = [int(value) for value in record.get("token_ids", [])]
    if cached_tokens != expected_tokens:
        raise RuntimeError(f"Duration record token mismatch for {utterance_id}")

    values = [int(value) for value in record.get("durations", [])]
    if len(values) != len(expected_tokens):
        raise RuntimeError(f"Duration/token length mismatch for {utterance_id}")
    if any(value < 0 for value in values):
        raise RuntimeError(f"Negative teacher duration for {utterance_id}")
    if sum(values) != int(mel_frames):
        raise RuntimeError(
            f"Duration sum mismatch for {utterance_id}: {sum(values)} != {mel_frames}"
        )
    return torch.tensor(values, dtype=torch.long)


def _loss_components(
    model: LykenoxSpeechAcousticModel,
    token_ids: torch.Tensor,
    mel_target: torch.Tensor,
    durations: torch.Tensor,
    *,
    duration_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_mask = torch.ones_like(token_ids, dtype=torch.bool).unsqueeze(0)
    output = model(
        token_ids.unsqueeze(0),
        token_mask,
        durations.unsqueeze(0),
    )
    mel_pred = output["mel"].squeeze(0)
    if mel_pred.shape != mel_target.shape:
        raise RuntimeError(
            "Aligned acoustic model did not preserve exact mel length: "
            f"pred={tuple(mel_pred.shape)} target={tuple(mel_target.shape)}"
        )

    acoustic_loss = F.l1_loss(mel_pred, mel_target)
    duration_prediction = output["duration_prediction"].squeeze(0)
    duration_loss = F.smooth_l1_loss(
        torch.log1p(duration_prediction),
        torch.log1p(durations.to(torch.float32)),
    )
    total_loss = acoustic_loss + duration_weight * duration_loss
    return total_loss, acoustic_loss, duration_loss


def _probe_losses(
    model: LykenoxSpeechAcousticModel,
    sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    duration_weight: float,
) -> tuple[float, float, float]:
    token_ids, mel_target, durations = sample
    model.eval()
    with torch.no_grad():
        total, acoustic, duration = _loss_components(
            model,
            token_ids,
            mel_target,
            durations,
            duration_weight=duration_weight,
        )
    model.train()
    return (
        float(total.detach().cpu()),
        float(acoustic.detach().cpu()),
        float(duration.detach().cpu()),
    )


def run_aligned_acoustic_smoke(
    root: Path,
    *,
    steps: int = 40,
    max_items: int = 8,
    max_mel_frames: int = 1200,
    duration_weight: float = 0.10,
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if max_items < 1:
        raise ValueError("max_items must be >= 1")
    if max_mel_frames < 1:
        raise ValueError("max_mel_frames must be >= 1")
    if duration_weight < 0:
        raise ValueError("duration_weight must be non-negative")
    if DURATION_CACHE_VERSION != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError(
            "The active LYKENOX duration-cache contract is not alignment-v3"
        )

    root = Path(root).resolve()
    duration_root = _latest_clean_duration_root(root)
    record_paths = _duration_record_paths(duration_root, "train")

    frontend = SpanishTextFrontend()
    config = LykenoxSpeechConfig(vocab_size=frontend.vocab_size)
    dataset = _dataset(root, "train", config)
    if len(dataset) == 0:
        raise RuntimeError("Speech training dataset is empty")

    candidates: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    skipped_too_long = 0
    for index in range(len(dataset)):
        item = dataset[index]
        utterance_id = str(item["utterance_id"])
        mel = item["mel"].to(torch.float32)
        if int(mel.shape[0]) > max_mel_frames:
            skipped_too_long += 1
            continue
        record_path = record_paths.get(utterance_id)
        if record_path is None or not record_path.exists():
            raise RuntimeError(f"Missing alignment-v3 record for {utterance_id}")
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        token_ids = item["token_ids"].to(torch.long)
        durations = validate_aligned_record(
            record,
            utterance_id=utterance_id,
            text=str(item["text"]),
            token_ids=token_ids,
            mel_frames=int(mel.shape[0]),
        )
        candidates.append((utterance_id, token_ids, mel, durations))
        if len(candidates) >= max_items:
            break

    if not candidates:
        raise RuntimeError(
            f"No aligned training items fit max_mel_frames={max_mel_frames}"
        )

    torch.manual_seed(1337)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = LykenoxSpeechAcousticModel(config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    probe = (candidates[0][1], candidates[0][2], candidates[0][3])
    probe_total_before, probe_acoustic_before, probe_duration_before = _probe_losses(
        model,
        probe,
        duration_weight=duration_weight,
    )

    losses: list[float] = []
    acoustic_losses: list[float] = []
    duration_losses: list[float] = []
    timings: list[float] = []
    used_ids: list[str] = []
    max_teacher_duration = 0

    for step in range(steps):
        utterance_id, token_ids, mel_target, durations = candidates[step % len(candidates)]
        max_teacher_duration = max(max_teacher_duration, int(durations.max().item()))

        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, acoustic_loss, duration_loss = _loss_components(
            model,
            token_ids,
            mel_target,
            durations,
            duration_weight=duration_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite aligned loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        optimizer.step()

        timings.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))
        acoustic_losses.append(float(acoustic_loss.detach().cpu()))
        duration_losses.append(float(duration_loss.detach().cpu()))
        used_ids.append(utterance_id)

    probe_total_after, probe_acoustic_after, probe_duration_after = _probe_losses(
        model,
        probe,
        duration_weight=duration_weight,
    )

    total_drop = probe_total_after < probe_total_before
    acoustic_drop = probe_acoustic_after < probe_acoustic_before
    duration_drop = probe_duration_after < probe_duration_before
    status = "pass" if total_drop and acoustic_drop and duration_drop else "needs_review"

    return {
        "status": status,
        "device": "cpu",
        "alignment": EXPECTED_DURATION_CACHE_VERSION,
        "duration_root": str(duration_root),
        "manifest": str(dataset.csv_path),
        "frontend_version": frontend.version,
        "vocab_size": frontend.vocab_size,
        "items_available": len(dataset),
        "items_used": len(set(used_ids)),
        "items_skipped_too_long_before_selection": skipped_too_long,
        "steps": steps,
        "parameters": model.parameter_count(),
        "duration_weight": duration_weight,
        "max_teacher_duration_frames_seen": max_teacher_duration,
        "exact_mel_length_enforced": True,
        "duration_sum_exact_enforced": True,
        "first_training_loss": round(losses[0], 6),
        "last_training_loss": round(losses[-1], 6),
        "first_acoustic_loss": round(acoustic_losses[0], 6),
        "last_acoustic_loss": round(acoustic_losses[-1], 6),
        "first_duration_loss": round(duration_losses[0], 6),
        "last_duration_loss": round(duration_losses[-1], 6),
        "probe_utterance_id": candidates[0][0],
        "probe_mel_frames": int(candidates[0][2].shape[0]),
        "probe_total_loss_before": round(probe_total_before, 6),
        "probe_total_loss_after": round(probe_total_after, 6),
        "probe_total_loss_decreased": total_drop,
        "probe_acoustic_loss_before": round(probe_acoustic_before, 6),
        "probe_acoustic_loss_after": round(probe_acoustic_after, 6),
        "probe_acoustic_loss_decreased": acoustic_drop,
        "probe_duration_loss_before": round(probe_duration_before, 6),
        "probe_duration_loss_after": round(probe_duration_after, 6),
        "probe_duration_loss_decreased": duration_drop,
        "mean_seconds_per_step": round(sum(timings) / len(timings), 4),
        "min_seconds_per_step": round(min(timings), 4),
        "max_seconds_per_step": round(max(timings), 4),
        "next_gate": (
            "fix_acoustic_training_contract_before_long_run"
            if status == "pass"
            else "review_aligned_acoustic_smoke"
        ),
        "warning": (
            "This is a real aligned-data CPU optimization gate, not final TTS training. "
            "Do not start a long run yet: batching/masking, export-safe length regulation, "
            "checkpoint metadata, and the LYKENOX vocoder remain separate gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--max-mel-frames", type=int, default=1200)
    parser.add_argument("--duration-weight", type=float, default=0.10)
    args = parser.parse_args()
    print(
        json.dumps(
            run_aligned_acoustic_smoke(
                args.root,
                steps=args.steps,
                max_items=args.max_items,
                max_mel_frames=args.max_mel_frames,
                duration_weight=args.duration_weight,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
