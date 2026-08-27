"""Batched/masked acoustic-training contract gate for LYKENOX Speech.

This gate runs only a short CPU smoke. It validates the pieces that must be fixed before
any long acoustic run:

- exact alignment-v3 teacher durations
- variable-length batching with token/mel masks
- padded mel frames excluded from loss
- padded tokens excluded from duration loss
- tensorized length regulation with exact per-item frame lengths
- exact frontend vocabulary metadata in a resumable checkpoint round-trip

It deliberately does not start production training and does not validate a vocoder.
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
from lykenox_voice_engine.training.speech_acoustic_artifact import (
    build_training_provenance,
    load_speech_acoustic_checkpoint,
    save_speech_acoustic_checkpoint,
    vocabulary_sha256,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_losses import (
    masked_l1_loss,
    speech_training_losses,
)


def _compute_losses(
    model: LykenoxSpeechAcousticModel,
    batch,
    *,
    duration_weight: float,
):
    output = model(batch.token_ids, batch.token_mask, batch.durations)
    if not torch.equal(output["mel_lengths"].cpu(), batch.mel_lengths.cpu()):
        raise RuntimeError("Model regulated mel lengths do not match teacher mel lengths")
    if not torch.equal(output["mel_mask"].cpu(), batch.mel_mask.cpu()):
        raise RuntimeError("Model regulated mel mask does not match padded batch mel mask")
    losses = speech_training_losses(
        mel_prediction=output["mel"],
        mel_target=batch.mel,
        mel_mask=batch.mel_mask,
        duration_prediction=output["duration_prediction"],
        duration_target=batch.durations,
        token_mask=batch.token_mask,
        duration_weight=duration_weight,
    )
    return output, losses


def _probe(model, batch, *, duration_weight: float) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        _, losses = _compute_losses(model, batch, duration_weight=duration_weight)
    model.train()
    return (
        float(losses.total.detach().cpu()),
        float(losses.acoustic.detach().cpu()),
        float(losses.duration.detach().cpu()),
    )


def _regulator_reference_check() -> bool:
    encoded = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    durations = torch.tensor([[0, 2, 0, 3], [1, 0, 4, 0]], dtype=torch.long)
    expanded, mask, lengths = LykenoxSpeechAcousticModel._length_regulate(
        encoded,
        durations,
    )
    expected_rows: list[torch.Tensor] = []
    for batch_index in range(encoded.shape[0]):
        chunks: list[torch.Tensor] = []
        for token_index in range(encoded.shape[1]):
            count = int(durations[batch_index, token_index])
            if count:
                chunks.append(encoded[batch_index, token_index].repeat(count, 1))
        expected_rows.append(torch.cat(chunks, dim=0))
    expected_lengths = torch.tensor([row.shape[0] for row in expected_rows], dtype=torch.long)
    if not torch.equal(lengths, expected_lengths):
        return False
    for batch_index, row in enumerate(expected_rows):
        count = row.shape[0]
        if not torch.equal(expanded[batch_index, :count], row):
            return False
        if not bool(mask[batch_index, :count].all().item()):
            return False
        if count < mask.shape[1] and bool(mask[batch_index, count:].any().item()):
            return False
    return True


def run_training_contract_smoke(
    root: Path,
    *,
    steps: int = 30,
    max_items: int = 8,
    batch_size: int = 2,
    max_mel_frames: int = 1200,
    duration_weight: float = 0.10,
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if max_items < 2:
        raise ValueError("max_items must be >= 2")
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 so padding behavior is exercised")
    if duration_weight < 0:
        raise ValueError("duration_weight must be non-negative")

    root = Path(root).resolve()
    torch.manual_seed(1337)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    frontend = SpanishTextFrontend()
    config = LykenoxSpeechConfig(vocab_size=frontend.vocab_size)
    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "train",
        config,
        duration_root=duration_root,
    )

    items: list[dict[str, object]] = []
    skipped_too_long = 0
    for index in range(len(dataset)):
        item = dataset[index]
        if int(item["mel"].shape[0]) > max_mel_frames:
            skipped_too_long += 1
            continue
        items.append(item)
        if len(items) >= max_items:
            break
    if len(items) < 2:
        raise RuntimeError("Not enough aligned examples fit the contract-smoke length bound")

    # Force the probe to contain unequal frame lengths so real padding exists.
    ordered = sorted(items, key=lambda item: int(item["mel"].shape[0]))
    probe_items = [ordered[0], ordered[-1]]
    probe_batch = collate_aligned_speech(probe_items)
    if int(probe_batch.mel_lengths[0]) == int(probe_batch.mel_lengths[1]):
        raise RuntimeError("Contract smoke needs unequal probe mel lengths to test masking")

    batches = [
        collate_aligned_speech(items[offset : offset + batch_size])
        for offset in range(0, len(items), batch_size)
        if len(items[offset : offset + batch_size]) >= 2
    ]
    if not batches:
        raise RuntimeError("Could not construct a padded aligned training batch")

    model = LykenoxSpeechAcousticModel(config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    probe_before = _probe(model, probe_batch, duration_weight=duration_weight)

    # The masked acoustic loss must be invariant to arbitrary values in padded target frames.
    model.eval()
    with torch.no_grad():
        probe_output = model(
            probe_batch.token_ids,
            probe_batch.token_mask,
            probe_batch.durations,
        )
        normal_masked = masked_l1_loss(
            probe_output["mel"],
            probe_batch.mel,
            probe_batch.mel_mask,
        )
        corrupted = probe_batch.mel.clone()
        corrupted[~probe_batch.mel_mask] = 1234.5
        corrupted_masked = masked_l1_loss(
            probe_output["mel"],
            corrupted,
            probe_batch.mel_mask,
        )
    model.train()
    padding_loss_delta = abs(float(normal_masked) - float(corrupted_masked))
    padding_loss_invariant = padding_loss_delta < 1e-7

    training_losses: list[float] = []
    acoustic_losses: list[float] = []
    duration_losses: list[float] = []
    timings: list[float] = []
    max_gradient_norm = 0.0

    for step in range(steps):
        batch = batches[step % len(batches)]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _, losses = _compute_losses(model, batch, duration_weight=duration_weight)
        if not torch.isfinite(losses.total):
            raise RuntimeError(f"Non-finite batched loss at step {step}")
        losses.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        timings.append(time.perf_counter() - started)
        training_losses.append(float(losses.total.detach().cpu()))
        acoustic_losses.append(float(losses.acoustic.detach().cpu()))
        duration_losses.append(float(losses.duration.detach().cpu()))
        max_gradient_norm = max(max_gradient_norm, float(grad_norm))

    probe_after = _probe(model, probe_batch, duration_weight=duration_weight)

    provenance = build_training_provenance(root, duration_root)
    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "speech_acoustic_contract_smoke"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "roundtrip.pt"
    save_speech_acoustic_checkpoint(
        checkpoint_path,
        model,
        frontend=frontend,
        epoch=0,
        global_step=steps,
        validation_loss=None,
        training_provenance=provenance,
        optimizer=optimizer,
        training_metadata={
            "purpose": "training_contract_smoke_only",
            "batch_size": batch_size,
            "steps": steps,
            "duration_weight": duration_weight,
        },
    )
    loaded_model, payload = load_speech_acoustic_checkpoint(checkpoint_path)

    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        original = model(
            probe_batch.token_ids,
            probe_batch.token_mask,
            probe_batch.durations,
        )
        restored = loaded_model(
            probe_batch.token_ids,
            probe_batch.token_mask,
            probe_batch.durations,
        )
    mel_roundtrip_delta = float(
        torch.max(torch.abs(original["mel"] - restored["mel"])).detach().cpu()
    )
    duration_roundtrip_delta = float(
        torch.max(
            torch.abs(
                original["duration_prediction"] - restored["duration_prediction"]
            )
        ).detach().cpu()
    )
    checkpoint_roundtrip_exact = (
        mel_roundtrip_delta == 0.0 and duration_roundtrip_delta == 0.0
    )

    regulator_reference_pass = _regulator_reference_check()
    probe_total_decreased = probe_after[0] < probe_before[0]
    probe_acoustic_decreased = probe_after[1] < probe_before[1]
    probe_duration_decreased = probe_after[2] < probe_before[2]
    vocab_exact = (
        int(payload["model_config"]["vocab_size"]) == frontend.vocab_size
        and payload["vocabulary_sha256"] == vocabulary_sha256(frontend.vocabulary())
    )
    gate_pass = all(
        (
            probe_total_decreased,
            probe_acoustic_decreased,
            probe_duration_decreased,
            padding_loss_invariant,
            regulator_reference_pass,
            checkpoint_roundtrip_exact,
            vocab_exact,
        )
    )

    report = {
        "status": "pass" if gate_pass else "needs_review",
        "device": "cpu",
        "alignment": "alignment-v3",
        "duration_root": str(duration_root),
        "items_available": len(dataset),
        "items_used": len(items),
        "skipped_too_long_before_selection": skipped_too_long,
        "steps": steps,
        "batch_size": batch_size,
        "parameters": model.parameter_count(),
        "frontend_version": frontend.version,
        "vocab_size": frontend.vocab_size,
        "vocabulary_sha256": payload["vocabulary_sha256"],
        "first_training_loss": round(training_losses[0], 6),
        "last_training_loss": round(training_losses[-1], 6),
        "first_acoustic_loss": round(acoustic_losses[0], 6),
        "last_acoustic_loss": round(acoustic_losses[-1], 6),
        "first_duration_loss": round(duration_losses[0], 6),
        "last_duration_loss": round(duration_losses[-1], 6),
        "probe_total_loss_before": round(probe_before[0], 6),
        "probe_total_loss_after": round(probe_after[0], 6),
        "probe_total_loss_decreased": probe_total_decreased,
        "probe_acoustic_loss_before": round(probe_before[1], 6),
        "probe_acoustic_loss_after": round(probe_after[1], 6),
        "probe_acoustic_loss_decreased": probe_acoustic_decreased,
        "probe_duration_loss_before": round(probe_before[2], 6),
        "probe_duration_loss_after": round(probe_after[2], 6),
        "probe_duration_loss_decreased": probe_duration_decreased,
        "probe_mel_lengths": [int(value) for value in probe_batch.mel_lengths.tolist()],
        "probe_token_lengths": [int(value) for value in probe_batch.token_lengths.tolist()],
        "padding_loss_invariant": padding_loss_invariant,
        "padding_loss_delta": padding_loss_delta,
        "length_regulator_tensor_reference_pass": regulator_reference_pass,
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "checkpoint_mel_max_abs_delta": mel_roundtrip_delta,
        "checkpoint_duration_max_abs_delta": duration_roundtrip_delta,
        "checkpoint": str(checkpoint_path),
        "checkpoint_vocab_exact": vocab_exact,
        "mean_seconds_per_step": round(sum(timings) / len(timings), 4),
        "min_seconds_per_step": round(min(timings), 4),
        "max_seconds_per_step": round(max(timings), 4),
        "max_gradient_norm": round(max_gradient_norm, 6),
        "next_gate": (
            "benchmark_lykenox_vocoder_cpu"
            if gate_pass
            else "fix_acoustic_training_contract"
        ),
        "warning": (
            "This gate validates the acoustic training contract only. It is not a long "
            "identity training run and does not prove intelligibility or waveform quality."
        ),
    }
    report_path = artifact_dir / "contract_smoke_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-mel-frames", type=int, default=1200)
    parser.add_argument("--duration-weight", type=float, default=0.10)
    args = parser.parse_args()
    result = run_training_contract_smoke(
        args.root,
        steps=args.steps,
        max_items=args.max_items,
        batch_size=args.batch_size,
        max_mel_frames=args.max_mel_frames,
        duration_weight=args.duration_weight,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
