"""Bounded CPU smoke for LYKENOX acoustic F0/voicing prediction heads.

This gate consumes only persistent LYKENOX artifacts:
- alignment-v3 teacher durations;
- mel-v1 acoustic targets;
- speech-pitch-cache-v1 frame-aligned F0/voicing targets.

It deliberately overfits one small real padded batch for a few CPU steps. Passing proves
that mel, duration, F0 and voicing heads share an exact frame contract and all four
supervised objectives can optimize together. It is not persistent acoustic training and
is not an end-to-end naturalness/intelligibility test.
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
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
)
from lykenox_voice_engine.training.speech_losses import SpeechLosses, speech_training_losses
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CACHE_VERSION


SMOKE_VERSION = "acoustic-prosody-heads-smoke-v1"


def _compute_losses(
    model: LykenoxSpeechAcousticModel,
    batch,
    *,
    duration_weight: float,
    f0_weight: float,
    voicing_weight: float,
) -> SpeechLosses:
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("Prosody smoke requires cached F0/voicing targets")
    output = model(batch.token_ids, batch.token_mask, batch.durations)
    if output["mel"].shape != batch.mel.shape:
        raise RuntimeError(
            f"Mel prediction shape mismatch: {tuple(output['mel'].shape)} != {tuple(batch.mel.shape)}"
        )
    if output["f0_prediction_hz"].shape != batch.f0_hz.shape:
        raise RuntimeError("F0 prediction is not aligned to the padded mel frame grid")
    if output["voicing_logits"].shape != batch.voiced.shape:
        raise RuntimeError("Voicing prediction is not aligned to the padded mel frame grid")
    if not torch.equal(output["mel_mask"], batch.mel_mask):
        raise RuntimeError("Model regulated mel mask differs from target batch mel mask")
    if not torch.equal(output["mel_lengths"], batch.mel_lengths):
        raise RuntimeError("Model regulated mel lengths differ from teacher mel lengths")

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
        raise RuntimeError("Prosody losses were not computed")
    return {
        "total": float(losses.total.detach().cpu()),
        "acoustic": float(losses.acoustic.detach().cpu()),
        "duration": float(losses.duration.detach().cpu()),
        "f0": float(losses.f0.detach().cpu()),
        "voicing": float(losses.voicing.detach().cpu()),
    }


def run_acoustic_prosody_smoke(
    root: Path,
    *,
    steps: int = 40,
    batch_size: int = 2,
    max_mel_frames: int = 900,
    duration_weight: float = 0.10,
    f0_weight: float = 0.25,
    voicing_weight: float = 0.25,
) -> dict[str, object]:
    if steps < 1 or batch_size < 1 or max_mel_frames < 1:
        raise ValueError("steps, batch_size and max_mel_frames must be positive")
    if min(duration_weight, f0_weight, voicing_weight) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if f0_weight == 0.0 or voicing_weight == 0.0:
        raise ValueError("prosody smoke requires non-zero F0 and voicing weights")

    root = Path(root).resolve()
    torch.manual_seed(1337)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    frontend = SpanishTextFrontend()
    config = LykenoxSpeechConfig(vocab_size=frontend.vocab_size)
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
        raise RuntimeError(
            f"Only {len(candidates)} items fit max_mel_frames={max_mel_frames}; need {batch_size}"
        )

    batch = collate_aligned_speech(candidates).to("cpu")
    assert batch.f0_hz is not None and batch.voiced is not None
    if not torch.equal(batch.durations.sum(dim=1), batch.mel_lengths):
        raise RuntimeError("Teacher duration sum does not equal mel length")
    if batch.f0_hz.shape != batch.mel_mask.shape or batch.voiced.shape != batch.mel_mask.shape:
        raise RuntimeError("Cached pitch targets do not match mel mask shape")
    if not bool((batch.f0_hz[~batch.mel_mask] == 0.0).all()):
        raise RuntimeError("Padded F0 targets must be zero")
    if not bool((batch.voiced[~batch.mel_mask] == 0.0).all()):
        raise RuntimeError("Padded voicing targets must be zero")

    model = LykenoxSpeechAcousticModel(config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    model.eval()
    with torch.no_grad():
        before = _snapshot(
            _compute_losses(
                model,
                batch,
                duration_weight=duration_weight,
                f0_weight=f0_weight,
                voicing_weight=voicing_weight,
            )
        )
    model.train()

    timings: list[float] = []
    first_training: dict[str, float] | None = None
    last_training: dict[str, float] | None = None
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
            raise RuntimeError(f"Non-finite total loss at step {step}")
        losses.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        timings.append(time.perf_counter() - started)
        current = _snapshot(losses)
        if first_training is None:
            first_training = current
        last_training = current

    model.eval()
    with torch.no_grad():
        after = _snapshot(
            _compute_losses(
                model,
                batch,
                duration_weight=duration_weight,
                f0_weight=f0_weight,
                voicing_weight=voicing_weight,
            )
        )

    decreases = {
        name: after[name] < before[name]
        for name in ("total", "acoustic", "duration", "f0", "voicing")
    }
    status = "pass" if all(decreases.values()) else "needs_review"
    voiced_real = batch.voiced[batch.mel_mask]
    active_f0 = batch.f0_hz[(batch.voiced > 0.5) & batch.mel_mask]

    return {
        "status": status,
        "device": "cpu",
        "smoke_version": SMOKE_VERSION,
        "pitch_cache_version": PITCH_CACHE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "frontend_version": frontend.version,
        "vocab_size": frontend.vocab_size,
        "parameters": model.parameter_count(),
        "steps": steps,
        "batch_size": batch_size,
        "utterance_ids": batch.utterance_ids,
        "mel_lengths": [int(value) for value in batch.mel_lengths.tolist()],
        "items_skipped_too_long_before_selection": skipped_too_long,
        "exact_duration_to_mel_length": bool(torch.equal(batch.durations.sum(dim=1), batch.mel_lengths)),
        "exact_pitch_to_mel_grid": bool(batch.f0_hz.shape == batch.mel_mask.shape),
        "real_voiced_fraction": round(float(voiced_real.mean()), 6),
        "real_voiced_f0_min_hz": round(float(active_f0.min()), 4) if active_f0.numel() else None,
        "real_voiced_f0_max_hz": round(float(active_f0.max()), 4) if active_f0.numel() else None,
        "duration_weight": duration_weight,
        "f0_weight": f0_weight,
        "voicing_weight": voicing_weight,
        "probe_before": {key: round(value, 6) for key, value in before.items()},
        "probe_after": {key: round(value, 6) for key, value in after.items()},
        "probe_decreased": decreases,
        "first_training": None if first_training is None else {key: round(value, 6) for key, value in first_training.items()},
        "last_training": None if last_training is None else {key: round(value, 6) for key, value in last_training.items()},
        "mean_seconds_per_step": round(sum(timings) / len(timings), 4),
        "max_seconds_per_step": round(max(timings), 4),
        "next_gate": (
            "build_bounded_resumable_acoustic_trainer_with_prosody"
            if status == "pass"
            else "review_acoustic_prosody_heads_before_training"
        ),
        "warning": (
            "A pass proves joint supervised optimization and exact frame contracts only. "
            "It does not prove unseen-text inference, predicted-duration semantics, final "
            "pitch accuracy, intelligibility, identity, or runtime export."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-mel-frames", type=int, default=900)
    parser.add_argument("--duration-weight", type=float, default=0.10)
    parser.add_argument("--f0-weight", type=float, default=0.25)
    parser.add_argument("--voicing-weight", type=float, default=0.25)
    args = parser.parse_args()
    print(
        json.dumps(
            run_acoustic_prosody_smoke(
                args.root,
                steps=args.steps,
                batch_size=args.batch_size,
                max_mel_frames=args.max_mel_frames,
                duration_weight=args.duration_weight,
                f0_weight=args.f0_weight,
                voicing_weight=args.voicing_weight,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
