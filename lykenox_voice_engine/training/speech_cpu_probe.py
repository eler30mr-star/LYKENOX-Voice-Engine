"""CPU feasibility probe for the LYKENOX-owned speech acoustic model.

The probe is intentionally synthetic: it measures whether the architecture can
perform forward/backward/update on the target machine before real dataset
training is attempted. It does not claim voice quality.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from dataclasses import asdict

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig


def _peak_rss_mb() -> float | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.name == "nt":
            return value / (1024 * 1024)
        return value / 1024
    except Exception:
        return None


def run_probe(steps: int = 10, batch_size: int = 1, text_steps: int = 32) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")
    config = LykenoxSpeechConfig()
    model = LykenoxSpeechAcousticModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    token_ids = torch.randint(5, min(config.vocab_size, 60), (batch_size, text_steps), device=device)
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    durations = torch.randint(2, 7, (batch_size, text_steps), device=device)

    timings: list[float] = []
    losses: list[float] = []
    for _ in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(token_ids, mask, durations)
        mel = output["mel"]
        duration_prediction = output["duration_prediction"]
        target_mel = torch.zeros_like(mel)
        loss = torch.nn.functional.l1_loss(mel, target_mel)
        loss = loss + 0.01 * torch.nn.functional.l1_loss(duration_prediction, durations.float())
        loss.backward()
        optimizer.step()
        timings.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))

    mean_step = sum(timings) / len(timings)
    return {
        "probe": "LYKENOX speech acoustic CPU forward/backward",
        "status": "pass",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "config": asdict(config),
        "parameters": model.parameter_count(),
        "steps": steps,
        "batch_size": batch_size,
        "text_steps": text_steps,
        "mean_seconds_per_step": round(mean_step, 4),
        "min_seconds_per_step": round(min(timings), 4),
        "max_seconds_per_step": round(max(timings), 4),
        "last_loss": round(losses[-1], 6),
        "peak_rss_mb": _peak_rss_mb(),
        "note": "Synthetic feasibility only; this does not validate identity or audio quality.",
    }


def main() -> None:
    print(json.dumps(run_probe(), indent=2))


if __name__ == "__main__":
    main()
