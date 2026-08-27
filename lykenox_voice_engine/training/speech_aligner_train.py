"""Train and validate the persistent LYKENOX speech CTC aligner on CPU."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
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
from lykenox_voice_engine.training.alignment_artifact import (
    load_aligner_checkpoint,
    save_aligner_checkpoint,
)
from lykenox_voice_engine.training.speech_dataset import LykenoxSpeechDataset


def _manifest_path(root: Path, split: str) -> Path:
    segmented = (
        root
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "prepared"
        / "speech_segmented"
        / f"{split}.segmented.csv"
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
        / f"{split}.csv"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No LYKENOX speech manifest found for {split}: {fallback}")


def _dataset(root: Path, split: str, config: LykenoxSpeechConfig) -> LykenoxSpeechDataset:
    csv_path = _manifest_path(root, split)
    cache_dir = (
        root
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "features"
        / "speech"
        / "mel-v1"
        / split
    )
    return LykenoxSpeechDataset(csv_path, cache_dir, config)


def _eligible_indices(
    dataset: LykenoxSpeechDataset,
    aligner_config: LykenoxCTCAlignerConfig,
    max_mel_frames: int,
) -> tuple[list[int], list[dict[str, object]]]:
    eligible: list[int] = []
    excluded: list[dict[str, object]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        mel = item["mel"]
        token_ids = item["token_ids"]
        targets, _ = ctc_targets(token_ids)
        output_steps = int(
            (int(mel.shape[0]) + aligner_config.frame_stride - 1)
            // aligner_config.frame_stride
        )
        reason: str | None = None
        if int(mel.shape[0]) > max_mel_frames:
            reason = "mel_too_long"
        elif output_steps < minimum_ctc_steps(targets):
            reason = "ctc_path_impossible"

        if reason is None:
            eligible.append(index)
        else:
            excluded.append(
                {
                    "utterance_id": str(item["utterance_id"]),
                    "mel_frames": int(mel.shape[0]),
                    "content_tokens": int(targets.numel()),
                    "output_steps": output_steps,
                    "reason": reason,
                }
            )
    return eligible, excluded


def _item_ctc_loss(
    model: LykenoxCTCAligner,
    item: dict[str, object],
    criterion: torch.nn.CTCLoss,
) -> torch.Tensor:
    mel = item["mel"]
    token_ids = item["token_ids"]
    targets, _ = ctc_targets(token_ids)
    logits = model(mel.unsqueeze(0))
    log_probs = F.log_softmax(logits, dim=-1)
    return criterion(
        log_probs.transpose(0, 1),
        targets,
        torch.tensor([log_probs.shape[1]], dtype=torch.long),
        torch.tensor([targets.numel()], dtype=torch.long),
    )


def _mean_ctc_loss(
    model: LykenoxCTCAligner,
    dataset: LykenoxSpeechDataset,
    indices: list[int],
    criterion: torch.nn.CTCLoss,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for index in indices:
            value = _item_ctc_loss(model, dataset[index], criterion)
            if not torch.isfinite(value):
                raise RuntimeError(f"Non-finite validation CTC loss for index {index}")
            total += float(value.detach().cpu())
    model.train()
    return total / max(1, len(indices))


def _audit_forced_alignments(
    model: LykenoxCTCAligner,
    dataset: LykenoxSpeechDataset,
    indices: list[int],
) -> dict[str, object]:
    exact = 0
    nonzero = 0
    failures: list[dict[str, object]] = []
    scores: list[float] = []
    max_content_duration = 0
    leading_boundary_frames: list[int] = []
    trailing_boundary_frames: list[int] = []

    model.eval()
    with torch.no_grad():
        for index in indices:
            item = dataset[index]
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
                    leading_boundary_frames=alignment.leading_boundary_frames,
                    trailing_boundary_frames=alignment.trailing_boundary_frames,
                )
                if int(full.sum().item()) == int(mel.shape[0]):
                    exact += 1
                if bool((alignment.target_durations > 0).all().item()):
                    nonzero += 1
                scores.append(float(alignment.score_per_step))
                leading_boundary_frames.append(alignment.leading_boundary_frames)
                trailing_boundary_frames.append(alignment.trailing_boundary_frames)
                max_content_duration = max(
                    max_content_duration,
                    int(alignment.target_durations.max().item()),
                )
            except Exception as error:
                failures.append(
                    {
                        "utterance_id": str(item["utterance_id"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    count = len(indices)
    return {
        "items": count,
        "forced_alignment_success": count - len(failures),
        "duration_sum_exact": exact,
        "all_content_nonzero": nonzero,
        "mean_alignment_score_per_step": (
            round(sum(scores) / len(scores), 6) if scores else None
        ),
        "max_content_duration_frames": max_content_duration,
        "max_leading_boundary_frames": max(leading_boundary_frames) if leading_boundary_frames else 0,
        "max_trailing_boundary_frames": max(trailing_boundary_frames) if trailing_boundary_frames else 0,
        "boundary_blank_policy": "leading_to_bos_trailing_to_eos",
        "failures": failures[:20],
        "pass": (
            count > 0
            and not failures
            and exact == count
            and nonzero == count
        ),
    }


def train_persistent_aligner(
    root: Path,
    *,
    epochs: int = 20,
    patience: int = 4,
    min_delta: float = 0.01,
    max_mel_frames: int = 1800,
    seed: int = 1337,
) -> dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if max_mel_frames < 100:
        raise ValueError("max_mel_frames is unrealistically small")

    root = Path(root).resolve()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    speech_config = LykenoxSpeechConfig()
    frontend = SpanishTextFrontend()
    aligner_config = LykenoxCTCAlignerConfig(
        num_symbols=frontend.vocab_size,
        mel_bins=speech_config.mel_bins,
    )
    train_dataset = _dataset(root, "train", speech_config)
    val_dataset = _dataset(root, "val", speech_config)
    train_indices, train_excluded = _eligible_indices(
        train_dataset, aligner_config, max_mel_frames
    )
    val_indices, val_excluded = _eligible_indices(
        val_dataset, aligner_config, max_mel_frames
    )
    if not train_indices:
        raise RuntimeError("No eligible train utterances for persistent aligner training")
    if not val_indices:
        raise RuntimeError("No eligible validation utterances for persistent aligner training")

    model = LykenoxCTCAligner(aligner_config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = torch.nn.CTCLoss(blank=aligner_config.blank_id, zero_infinity=True)

    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "speech_aligner"
        / frontend.version
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = artifact_dir / "best.pt"
    last_path = artifact_dir / "last.pt"
    report_path = artifact_dir / "training_report.json"

    initial_val = _mean_ctc_loss(model, val_dataset, val_indices, criterion)
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    epoch_rows: list[dict[str, object]] = []
    total_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        order = list(train_indices)
        random.Random(seed + epoch).shuffle(order)
        train_losses: list[float] = []

        model.train()
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = _item_ctc_loss(model, train_dataset[index], criterion)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite train CTC loss at epoch {epoch}, index {index}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(
                    f"Non-finite gradient norm at epoch {epoch}, index {index}"
                )
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_mean = sum(train_losses) / len(train_losses)
        val_mean = _mean_ctc_loss(model, val_dataset, val_indices, criterion)
        improved = val_mean < (best_val - min_delta)
        if improved:
            best_val = val_mean
            best_epoch = epoch
            bad_epochs = 0
            save_aligner_checkpoint(
                best_path,
                model,
                frontend=frontend,
                speech_config=speech_config.to_dict(),
                epoch=epoch,
                validation_ctc_loss=val_mean,
                training_metadata={
                    "seed": seed,
                    "train_manifest": str(train_dataset.csv_path),
                    "val_manifest": str(val_dataset.csv_path),
                    "train_eligible": len(train_indices),
                    "val_eligible": len(val_indices),
                    "max_mel_frames": max_mel_frames,
                },
            )
        else:
            bad_epochs += 1

        save_aligner_checkpoint(
            last_path,
            model,
            frontend=frontend,
            speech_config=speech_config.to_dict(),
            epoch=epoch,
            validation_ctc_loss=val_mean,
            training_metadata={
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_ctc_loss": best_val,
            },
        )

        row = {
            "epoch": epoch,
            "train_ctc_loss": round(train_mean, 6),
            "val_ctc_loss": round(val_mean, 6),
            "improved": improved,
            "seconds": round(time.perf_counter() - epoch_started, 2),
        }
        epoch_rows.append(row)
        print(
            f"[LYKENOX aligner] epoch {epoch}/{epochs} "
            f"train={train_mean:.4f} val={val_mean:.4f} "
            f"{'best' if improved else f'patience={bad_epochs}/{patience}'}",
            file=sys.stderr,
            flush=True,
        )
        if bad_epochs >= patience:
            break

    if not best_path.exists():
        raise RuntimeError("Persistent aligner training did not produce a best checkpoint")

    best_model, best_payload = load_aligner_checkpoint(best_path)
    final_audit = _audit_forced_alignments(best_model, val_dataset, val_indices)
    val_improved = best_val < initial_val
    status = "pass" if val_improved and bool(final_audit["pass"]) else "needs_review"

    report = {
        "status": status,
        "device": "cpu",
        "frontend_version": frontend.version,
        "parameters": best_model.parameter_count(),
        "train_manifest": str(train_dataset.csv_path),
        "val_manifest": str(val_dataset.csv_path),
        "train_items_total": len(train_dataset),
        "val_items_total": len(val_dataset),
        "train_items_eligible": len(train_indices),
        "val_items_eligible": len(val_indices),
        "train_excluded": train_excluded,
        "val_excluded": val_excluded,
        "epochs_requested": epochs,
        "epochs_completed": len(epoch_rows),
        "patience": patience,
        "best_epoch": best_epoch,
        "initial_val_ctc_loss": round(initial_val, 6),
        "best_val_ctc_loss": round(best_val, 6),
        "validation_loss_decreased": val_improved,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "checkpoint_epoch": int(best_payload["epoch"]),
        "validation_alignment_audit": final_audit,
        "elapsed_seconds": round(time.perf_counter() - total_started, 2),
        "history": epoch_rows,
        "next_gate": (
            "generate_and_audit_duration_cache"
            if status == "pass"
            else "review_alignment_training"
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--max-mel-frames", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    print(
        json.dumps(
            train_persistent_aligner(
                args.root,
                epochs=args.epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                max_mel_frames=args.max_mel_frames,
                seed=args.seed,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()