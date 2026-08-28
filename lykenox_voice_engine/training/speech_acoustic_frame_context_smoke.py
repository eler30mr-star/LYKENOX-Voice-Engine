"""Bounded CPU gate for the post-regulation LYKENOX frame-context fix.

The first persistent acoustic model learned all supervised losses but failed the held-out
expressivity audit because every frame inside one token received exactly the same repeated
encoder vector. Mel and F0 were therefore mathematically piecewise-constant within each
phoneme.

This smoke selects the new ``token-progress-conv-v1`` acoustic configuration, overfits one
small real batch, and requires both joint loss improvement and non-zero intra-token mel/F0
motion. It is an architecture gate only; it does not reuse or continue the rejected v1
persistent acoustic checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.models.speech.config import FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
from lykenox_voice_engine.training.speech_aligned_data import (
    AlignedSpeechBatch,
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
)
from lykenox_voice_engine.training.speech_losses import SpeechLosses, speech_training_losses


SMOKE_VERSION = "acoustic-frame-context-smoke-v1"


def _compute_losses(
    model: LykenoxSpeechAcousticModel,
    batch: AlignedSpeechBatch,
    *,
    duration_weight: float,
    f0_weight: float,
    voicing_weight: float,
) -> SpeechLosses:
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("Frame-context smoke requires cached F0/voicing targets")
    output = model(batch.token_ids, batch.token_mask, batch.durations)
    if output["mel"].shape != batch.mel.shape:
        raise RuntimeError("Frame-context model did not preserve exact mel shape")
    if output["f0_prediction_hz"].shape != batch.f0_hz.shape:
        raise RuntimeError("Frame-context F0 output is off the target frame grid")
    if output["voicing_logits"].shape != batch.voiced.shape:
        raise RuntimeError("Frame-context voicing output is off the target frame grid")
    if not torch.equal(output["mel_mask"], batch.mel_mask):
        raise RuntimeError("Frame-context regulated mask differs from target mel mask")
    if not torch.equal(output["mel_lengths"], batch.mel_lengths):
        raise RuntimeError("Frame-context regulated lengths differ from teacher lengths")
    return speech_training_losses(
        mel_prediction=output["mel"],
        mel_target=batch.mel,
        mel_mask=batch.mel_mask,
        duration_prediction=output["duration_prediction"],
        duration_target=batch.durations,
        token_mask=batch.token_mask,
        duration_weight=duration_weight,
        f0_prediction_hz=output["f0_prediction_hz"],
        f0_target_hz=batch.f0_hz,
        voicing_logits=output["voicing_logits"],
        voicing_target=batch.voiced,
        f0_weight=f0_weight,
        voicing_weight=voicing_weight,
    )


def _snapshot(losses: SpeechLosses) -> dict[str, float]:
    if losses.f0 is None or losses.voicing is None:
        raise RuntimeError("Frame-context smoke requires prosody losses")
    return {
        "total": float(losses.total.detach().cpu()),
        "acoustic": float(losses.acoustic.detach().cpu()),
        "duration": float(losses.duration.detach().cpu()),
        "f0": float(losses.f0.detach().cpu()),
        "voicing": float(losses.voicing.detach().cpu()),
    }


def _token_internal_pair_mask(durations: torch.Tensor, frame_count: int) -> torch.Tensor:
    if durations.ndim != 1 or int(durations.sum()) != int(frame_count):
        raise ValueError("duration sum must equal frame_count")
    if frame_count <= 1:
        return torch.zeros((0,), dtype=torch.bool)
    ends = torch.cumsum(durations.to(torch.long), dim=0)
    next_frames = torch.arange(1, frame_count, dtype=torch.long)
    positive_ends = ends[(durations > 0) & (ends < frame_count)]
    if positive_ends.numel() == 0:
        return torch.ones((frame_count - 1,), dtype=torch.bool)
    crosses = (next_frames.unsqueeze(1) == positive_ends.unsqueeze(0)).any(dim=1)
    return ~crosses


def _motion_metrics(
    output: dict[str, torch.Tensor],
    batch: AlignedSpeechBatch,
) -> dict[str, float]:
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("Motion metrics require pitch targets")

    pred_mel_sum = 0.0
    target_mel_sum = 0.0
    mel_pairs = 0
    pred_f0_sum = 0.0
    target_f0_sum = 0.0
    f0_pairs = 0

    for index in range(batch.token_ids.shape[0]):
        frame_count = int(batch.mel_lengths[index])
        internal = _token_internal_pair_mask(
            batch.durations[index, : int(batch.token_lengths[index])].cpu(),
            frame_count,
        )
        if internal.numel() == 0:
            continue

        predicted_mel = output["mel"][index, :frame_count].detach().cpu()
        target_mel = batch.mel[index, :frame_count].detach().cpu()
        pred_mel_delta = torch.abs(predicted_mel[1:] - predicted_mel[:-1]).mean(dim=-1)
        target_mel_delta = torch.abs(target_mel[1:] - target_mel[:-1]).mean(dim=-1)
        pred_mel_sum += float(pred_mel_delta[internal].sum())
        target_mel_sum += float(target_mel_delta[internal].sum())
        mel_pairs += int(internal.sum())

        target_voiced = batch.voiced[index, :frame_count].detach().cpu() > 0.5
        voiced_pair = internal & target_voiced[1:] & target_voiced[:-1]
        if bool(voiced_pair.any()):
            predicted_f0 = output["f0_prediction_hz"][index, :frame_count].detach().cpu()
            target_f0 = batch.f0_hz[index, :frame_count].detach().cpu()
            scale = 1200.0 / math.log(2.0)
            pred_cents = torch.abs(
                scale
                * (
                    torch.log(predicted_f0[1:].clamp_min(1e-6))
                    - torch.log(predicted_f0[:-1].clamp_min(1e-6))
                )
            )
            target_cents = torch.abs(
                scale
                * (
                    torch.log(target_f0[1:].clamp_min(1e-6))
                    - torch.log(target_f0[:-1].clamp_min(1e-6))
                )
            )
            pred_f0_sum += float(pred_cents[voiced_pair].sum())
            target_f0_sum += float(target_cents[voiced_pair].sum())
            f0_pairs += int(voiced_pair.sum())

    return {
        "intra_token_mel_delta_l1_predicted": pred_mel_sum / max(1, mel_pairs),
        "intra_token_mel_delta_l1_target": target_mel_sum / max(1, mel_pairs),
        "intra_token_f0_delta_cents_predicted": pred_f0_sum / max(1, f0_pairs),
        "intra_token_f0_delta_cents_target": target_f0_sum / max(1, f0_pairs),
        "intra_token_mel_pair_count": float(mel_pairs),
        "intra_token_voiced_f0_pair_count": float(f0_pairs),
    }


def run_frame_context_smoke(
    root: Path,
    *,
    steps: int = 60,
    batch_size: int = 2,
    max_mel_frames: int = 900,
    duration_weight: float = 0.10,
    f0_weight: float = 0.25,
    voicing_weight: float = 0.25,
) -> dict[str, object]:
    if steps < 1 or batch_size < 1 or max_mel_frames < 1:
        raise ValueError("steps, batch_size and max_mel_frames must be positive")

    root = Path(root).resolve()
    torch.manual_seed(1337)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    frontend = SpanishTextFrontend()
    config = LykenoxSpeechConfig(
        vocab_size=frontend.vocab_size,
        frame_context_version=FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1,
    )
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "train",
        config,
        include_pitch_targets=True,
    )
    candidates: list[dict[str, object]] = []
    skipped_too_long = 0
    for index in range(len(dataset)):
        item = dataset[index]
        if int(item["mel"].shape[0]) > max_mel_frames:
            skipped_too_long += 1
            continue
        candidates.append(item)
        if len(candidates) >= batch_size:
            break
    if len(candidates) < batch_size:
        raise RuntimeError("Not enough bounded real items for frame-context smoke")

    batch = collate_aligned_speech(candidates).to("cpu")
    model = LykenoxSpeechAcousticModel(config).cpu().train()
    if model.frame_context is None:
        raise RuntimeError("Requested frame-context architecture was not constructed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    model.eval()
    with torch.no_grad():
        before_losses = _snapshot(
            _compute_losses(
                model,
                batch,
                duration_weight=duration_weight,
                f0_weight=f0_weight,
                voicing_weight=voicing_weight,
            )
        )
        before_output = model(batch.token_ids, batch.token_mask, batch.durations)
        before_motion = _motion_metrics(before_output, batch)
    model.train()

    timings: list[float] = []
    frame_context_gradient_seen = False
    for step in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = _compute_losses(
            model,
            batch,
            duration_weight=duration_weight,
            f0_weight=f0_weight,
            voicing_weight=voicing_weight,
        )
        if not torch.isfinite(losses.total):
            raise RuntimeError(f"Non-finite frame-context loss at step {step}")
        losses.total.backward()
        context_grad_sq = 0.0
        for parameter in model.frame_context.parameters():
            if parameter.grad is not None:
                context_grad_sq += float(parameter.grad.detach().square().sum())
        frame_context_gradient_seen = frame_context_gradient_seen or context_grad_sq > 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        timings.append(time.perf_counter() - started)

    model.eval()
    with torch.no_grad():
        after_losses = _snapshot(
            _compute_losses(
                model,
                batch,
                duration_weight=duration_weight,
                f0_weight=f0_weight,
                voicing_weight=voicing_weight,
            )
        )
        after_output = model(batch.token_ids, batch.token_mask, batch.durations)
        after_motion = _motion_metrics(after_output, batch)

    decreases = {
        name: after_losses[name] < before_losses[name]
        for name in ("total", "acoustic", "duration", "f0", "voicing")
    }
    mel_motion_pass = after_motion["intra_token_mel_delta_l1_predicted"] > 1e-5
    f0_motion_pass = after_motion["intra_token_f0_delta_cents_predicted"] > 0.1
    exact_contract = bool(
        torch.equal(after_output["mel_lengths"], batch.mel_lengths)
        and torch.equal(after_output["mel_mask"], batch.mel_mask)
    )
    status = (
        "pass"
        if all(decreases.values())
        and mel_motion_pass
        and f0_motion_pass
        and frame_context_gradient_seen
        and exact_contract
        else "needs_review"
    )

    return {
        "status": status,
        "device": "cpu",
        "smoke_version": SMOKE_VERSION,
        "frame_context_version": config.frame_context_version,
        "frame_context_layers": config.frame_context_layers,
        "frame_context_kernel_size": config.frame_context_kernel_size,
        "parameters": model.parameter_count(),
        "steps": steps,
        "batch_size": batch_size,
        "utterance_ids": batch.utterance_ids,
        "mel_lengths": [int(value) for value in batch.mel_lengths.tolist()],
        "items_skipped_too_long_before_selection": skipped_too_long,
        "exact_duration_to_frame_contract": exact_contract,
        "frame_context_gradient_seen": frame_context_gradient_seen,
        "probe_before": {key: round(value, 6) for key, value in before_losses.items()},
        "probe_after": {key: round(value, 6) for key, value in after_losses.items()},
        "probe_decreased": decreases,
        "motion_before": {
            key: round(value, 8) for key, value in before_motion.items()
        },
        "motion_after": {
            key: round(value, 8) for key, value in after_motion.items()
        },
        "intra_token_mel_motion_pass": mel_motion_pass,
        "intra_token_f0_motion_pass": f0_motion_pass,
        "mean_seconds_per_step": round(sum(timings) / len(timings), 4),
        "max_seconds_per_step": round(max(timings), 4),
        "next_gate": (
            "build_v2_persistent_acoustic_trainer_with_frame_context"
            if status == "pass"
            else "review_post_regulation_frame_context"
        ),
        "warning": (
            "A pass proves the structural zero-motion defect is removed and the new frame "
            "context can optimize on bounded real data. It does not make the rejected v1 "
            "checkpoint usable; persistent v2 acoustic training remains a separate gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-mel-frames", type=int, default=900)
    args = parser.parse_args()
    print(
        json.dumps(
            run_frame_context_smoke(
                args.root,
                steps=args.steps,
                batch_size=args.batch_size,
                max_mel_frames=args.max_mel_frames,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
