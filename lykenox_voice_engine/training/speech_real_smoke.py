"""Tiny real-data smoke test for LYKENOX Speech.

This validates the real WAV -> mel -> text -> acoustic model -> backprop path on
CPU. It deliberately uses temporary uniform durations; it is not a quality
training recipe and must not be used for a final voice.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset, uniform_bootstrap_durations


def run_smoke(root: Path, steps: int = 20, max_items: int = 8) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    prepared = root / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech_segmented"
    csv_path = prepared / "train.segmented.csv"
    if not csv_path.exists():
        csv_path = root / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech" / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No speech training manifest found: {csv_path}")

    cache_dir = root / "datasets" / "lykenox" / "identity_voice" / "features" / "speech" / "mel-v1" / "train"
    config = LykenoxSpeechConfig()
    dataset = LykenoxSpeechDataset(csv_path, cache_dir, config)
    if len(dataset) == 0:
        raise RuntimeError("Speech dataset is empty")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = LykenoxSpeechAcousticModel(config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    losses: list[float] = []
    timings: list[float] = []
    used_ids: list[str] = []

    for step in range(steps):
        item = dataset[step % min(len(dataset), max_items)]
        token_ids = item["token_ids"]
        mel_target = item["mel"]
        # Bound smoke-test work. Long examples are intentionally clipped here;
        # production training will use proper batching/cropping.
        mel_target = mel_target[: min(1200, mel_target.shape[0])]
        token_ids = token_ids[: min(160, token_ids.shape[0])]
        durations = uniform_bootstrap_durations(int(token_ids.shape[0]), int(mel_target.shape[0]))

        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(token_ids.unsqueeze(0), torch.ones_like(token_ids, dtype=torch.bool).unsqueeze(0), durations.unsqueeze(0))
        mel_pred = output["mel"].squeeze(0)
        usable = min(mel_pred.shape[0], mel_target.shape[0])
        acoustic_loss = F.l1_loss(mel_pred[:usable], mel_target[:usable])
        duration_loss = F.l1_loss(output["duration_prediction"].squeeze(0), durations.float())
        loss = acoustic_loss + 0.0025 * duration_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        elapsed = time.perf_counter() - started

        losses.append(float(loss.detach()))
        timings.append(elapsed)
        used_ids.append(str(item["utterance_id"]))
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")

    trend = losses[-1] < losses[0]
    return {
        "status": "pass" if trend else "pass_no_loss_drop",
        "device": "cpu",
        "manifest": str(csv_path),
        "items_available": len(dataset),
        "items_used": len(set(used_ids)),
        "steps": steps,
        "parameters": model.parameter_count(),
        "first_loss": round(losses[0], 6),
        "last_loss": round(losses[-1], 6),
        "loss_decreased": trend,
        "mean_seconds_per_step": round(sum(timings) / len(timings), 4),
        "min_seconds_per_step": round(min(timings), 4),
        "max_seconds_per_step": round(max(timings), 4),
        "alignment": "uniform_bootstrap_only",
        "warning": "This validates real-data plumbing only. Uniform durations are not acceptable for production voice training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-items", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.root.resolve(), args.steps, args.max_items), indent=2))


if __name__ == "__main__":
    main()
