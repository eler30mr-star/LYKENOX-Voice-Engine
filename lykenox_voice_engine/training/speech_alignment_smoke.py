"""Real-data CTC alignment smoke test for LYKENOX Speech.

This trains only the small LYKENOX-owned aligner long enough to verify that
real mel/text pairs can optimize with CTC and produce exact monotonic durations.
It does not save a production checkpoint and does not start full voice training.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lykenox_voice_engine.core.ctc_alignment import (
    ctc_targets,
    expand_content_durations,
    forced_alignment_durations,
    minimum_ctc_steps,
)
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import (
    LykenoxCTCAligner,
    LykenoxCTCAlignerConfig,
    LykenoxSpeechConfig,
)
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset


def _manifest_path(root: Path) -> Path:
    segmented = (
        root
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech_segmented"
        / "train.segmented.csv"
    )
    if segmented.exists():
        return segmented
    fallback = (
        root
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech"
        / "train.csv"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No LYKENOX speech training manifest found: {fallback}")


def _probe_loss(
    model: LykenoxCTCAligner,
    item: dict[str, object],
    criterion: torch.nn.CTCLoss,
) -> float:
    model.eval()
    with torch.no_grad():
        mel = item["mel"]
        token_ids = item["token_ids"]
        targets, _ = ctc_targets(token_ids)
        logits = model(mel.unsqueeze(0))
        log_probs = F.log_softmax(logits, dim=-1)
        input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
        target_lengths = torch.tensor([targets.numel()], dtype=torch.long)
        loss = criterion(
            log_probs.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
        )
    model.train()
    return float(loss.detach().cpu())


def run_alignment_smoke(
    root: Path,
    steps: int = 120,
    max_items: int = 12,
    max_mel_frames: int = 1400,
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    csv_path = _manifest_path(root)
    cache_dir = (
        root
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "features"
        / "speech"
        / "mel-v1"
        / "train"
    )
    speech_config = LykenoxSpeechConfig()
    dataset = LykenoxSpeechDataset(csv_path, cache_dir, speech_config)
    if len(dataset) == 0:
        raise RuntimeError("Speech dataset is empty")

    frontend = SpanishTextFrontend()
    aligner_config = LykenoxCTCAlignerConfig(
        num_symbols=frontend.vocab_size,
        mel_bins=speech_config.mel_bins,
    )
    model = LykenoxCTCAligner(aligner_config).cpu().train()
    criterion = torch.nn.CTCLoss(blank=aligner_config.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    candidates: list[dict[str, object]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        mel = item["mel"]
        token_ids = item["token_ids"]
        targets, _ = ctc_targets(token_ids)
        output_steps = int(
            (mel.shape[0] + aligner_config.frame_stride - 1)
            // aligner_config.frame_stride
        )
        if (
            mel.shape[0] <= max_mel_frames
            and output_steps >= minimum_ctc_steps(targets)
        ):
            candidates.append(item)
            if len(candidates) >= max_items:
                break
    if not candidates:
        raise RuntimeError("No speech items satisfy the CTC smoke-test length limits")

    probe_item = candidates[0]
    probe_before = _probe_loss(model, probe_item, criterion)

    losses: list[float] = []
    timings: list[float] = []
    used_ids: list[str] = []

    for step in range(steps):
        item = candidates[step % len(candidates)]
        mel = item["mel"]
        token_ids = item["token_ids"]
        targets, _ = ctc_targets(token_ids)

        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(mel.unsqueeze(0))
        log_probs = F.log_softmax(logits, dim=-1)
        input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
        target_lengths = torch.tensor([targets.numel()], dtype=torch.long)
        loss = criterion(
            log_probs.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite CTC loss at step {step}: {float(loss.detach())}"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        optimizer.step()

        timings.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))
        used_ids.append(str(item["utterance_id"]))

    probe_after = _probe_loss(model, probe_item, criterion)

    model.eval()
    with torch.no_grad():
        probe_mel = probe_item["mel"]
        probe_token_ids = probe_item["token_ids"]
        probe_targets, positions = ctc_targets(probe_token_ids)
        logits = model(probe_mel.unsqueeze(0)).squeeze(0)
        log_probs = F.log_softmax(logits, dim=-1)
        alignment = forced_alignment_durations(
            log_probs,
            probe_targets,
            aligner_config.blank_id,
            mel_frames=int(probe_mel.shape[0]),
            frame_stride=aligner_config.frame_stride,
        )
        full_durations = expand_content_durations(
            probe_token_ids,
            alignment.target_durations,
            positions,
            leading_boundary_frames=alignment.leading_boundary_frames,
            trailing_boundary_frames=alignment.trailing_boundary_frames,
        )

    duration_sum_ok = int(full_durations.sum().item()) == int(probe_mel.shape[0])
    content_nonzero = bool((alignment.target_durations > 0).all().item())
    probe_drop = probe_after < probe_before
    status = "pass" if probe_drop and duration_sum_ok and content_nonzero else "needs_review"

    return {
        "status": status,
        "device": "cpu",
        "manifest": str(csv_path),
        "items_available": len(dataset),
        "items_used": len(set(used_ids)),
        "steps": steps,
        "parameters": model.parameter_count(),
        "blank_id": aligner_config.blank_id,
        "frame_stride": aligner_config.frame_stride,
        "first_training_loss": round(losses[0], 6),
        "last_training_loss": round(losses[-1], 6),
        "probe_ctc_loss_before": round(probe_before, 6),
        "probe_ctc_loss_after": round(probe_after, 6),
        "probe_loss_decreased": probe_drop,
        "mean_seconds_per_step": round(sum(timings) / len(timings), 4),
        "min_seconds_per_step": round(min(timings), 4),
        "max_seconds_per_step": round(max(timings), 4),
        "probe_utterance_id": str(probe_item["utterance_id"]),
        "probe_mel_frames": int(probe_mel.shape[0]),
        "probe_content_tokens": int(probe_targets.numel()),
        "duration_sum_matches_mel": duration_sum_ok,
        "all_content_tokens_nonzero": content_nonzero,
        "min_content_duration_frames": int(alignment.target_durations.min().item()),
        "max_content_duration_frames": int(alignment.target_durations.max().item()),
        "leading_boundary_frames": alignment.leading_boundary_frames,
        "trailing_boundary_frames": alignment.trailing_boundary_frames,
        "boundary_blank_policy": "leading_to_bos_trailing_to_eos",
        "alignment_score_per_step": round(alignment.score_per_step, 6),
        "alignment": "lykenox_ctc_viterbi_smoke",
        "warning": (
            "This validates the owned alignment path only. Do not use the smoke "
            "model as a production aligner or start long acoustic training yet."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--max-mel-frames", type=int, default=1400)
    args = parser.parse_args()
    print(
        json.dumps(
            run_alignment_smoke(
                args.root.resolve(),
                steps=args.steps,
                max_items=args.max_items,
                max_mel_frames=args.max_mel_frames,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
