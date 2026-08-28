"""Held-out audit for the persistent LYKENOX acoustic prosody checkpoint.

This gate deliberately uses teacher durations on the validation split so it isolates the
learned acoustic/prosody heads from the still-unresolved inference-duration semantics.
It measures mel reconstruction, F0 accuracy, voicing classification, and—critically—
whether the model can express frame-to-frame variation *inside* a token duration.

The current acoustic bootstrap length-regulates by repeating one encoded token vector.
A frame decoder with no post-regulation temporal context can therefore become exactly
piecewise-constant inside each token. The persistent training smoke cannot detect that
expressivity limit because all supervised losses may still decrease. This audit makes the
limitation explicit before any end-to-end unseen-text synthesis is attempted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    build_acoustic_prosody_provenance,
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)


AUDIT_VERSION = "acoustic-prosody-heldout-audit-v1"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() != x.numel():
        return None
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denom) <= 1e-12:
        return None
    return float((x * y).sum() / denom)


def _token_internal_pair_mask(durations: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Return mask over adjacent frame pairs that remain inside the same token.

    Output length is ``max(frame_count - 1, 0)``. Zero-duration structural tokens vanish
    naturally because they contribute no frame positions.
    """

    if durations.ndim != 1:
        raise ValueError("durations must be one-dimensional")
    if frame_count < 0 or int(durations.sum()) != int(frame_count):
        raise ValueError("duration sum must equal frame_count")
    if frame_count <= 1:
        return torch.zeros((0,), dtype=torch.bool)

    ends = torch.cumsum(durations.to(torch.long), dim=0)
    # A pair (t, t+1) crosses a token boundary exactly when t+1 equals a positive
    # cumulative end. Build those boundaries without Python expansion by frame count.
    next_frames = torch.arange(1, frame_count, dtype=torch.long)
    positive_ends = ends[(durations > 0) & (ends < frame_count)]
    crosses = (next_frames.unsqueeze(1) == positive_ends.unsqueeze(0)).any(dim=1)
    return ~crosses


def _voicing_counts(predicted: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int, int]:
    predicted = predicted.bool()
    target = target.bool()
    tp = int((predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    return tp, tn, fp, fn


def _classification_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    specificity = tn / max(1, tn + fp)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    balanced = 0.5 * (recall + specificity)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
    }


def run_acoustic_prosody_audit(
    root: Path,
    *,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_prosody_v1"
        / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Persistent acoustic best checkpoint not found: {checkpoint}")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model, payload = load_acoustic_prosody_checkpoint(checkpoint)
    duration_root = find_clean_duration_root(root)
    current_provenance = build_acoustic_prosody_provenance(
        root,
        duration_root=duration_root,
        config=model.config,
    )
    if payload.get("training_provenance") != current_provenance:
        raise RuntimeError("Persistent acoustic checkpoint provenance no longer matches current data")

    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        model.config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(dataset) < 1:
        raise RuntimeError("Validation dataset is empty")

    model.cpu().eval()
    total_mel_abs = 0.0
    total_mel_values = 0
    total_internal_pred_mel_delta = 0.0
    total_internal_target_mel_delta = 0.0
    total_internal_mel_values = 0
    total_internal_pred_f0_delta_cents = 0.0
    total_internal_target_f0_delta_cents = 0.0
    total_internal_f0_pairs = 0
    all_pred_log_f0: list[torch.Tensor] = []
    all_target_log_f0: list[torch.Tensor] = []
    all_cents_error: list[torch.Tensor] = []
    tp = tn = fp = fn = 0
    predicted_voiced_frames = 0
    target_voiced_frames = 0
    real_frames = 0
    exact_contract_count = 0
    per_utterance: list[dict[str, object]] = []

    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("Held-out audit requires cached pitch targets")
            output = model(batch.token_ids, batch.token_mask, batch.durations)

            exact = (
                output["mel"].shape == batch.mel.shape
                and output["f0_prediction_hz"].shape == batch.f0_hz.shape
                and output["voicing_logits"].shape == batch.voiced.shape
                and torch.equal(output["mel_mask"], batch.mel_mask)
                and torch.equal(output["mel_lengths"], batch.mel_lengths)
            )
            if not exact:
                raise RuntimeError(f"Held-out frame contract failed for {item['utterance_id']}")
            exact_contract_count += 1

            frames = int(batch.mel_lengths[0])
            pred_mel = output["mel"][0, :frames]
            target_mel = batch.mel[0, :frames]
            pred_f0 = output["f0_prediction_hz"][0, :frames]
            target_f0 = batch.f0_hz[0, :frames]
            target_voiced = batch.voiced[0, :frames] > 0.5
            predicted_voiced = torch.sigmoid(output["voicing_logits"][0, :frames]) >= 0.5

            if not (
                torch.isfinite(pred_mel).all()
                and torch.isfinite(pred_f0).all()
                and torch.isfinite(output["voicing_logits"][0, :frames]).all()
            ):
                raise RuntimeError(f"Non-finite held-out output for {item['utterance_id']}")

            mel_abs = torch.abs(pred_mel - target_mel)
            total_mel_abs += float(mel_abs.sum())
            total_mel_values += int(mel_abs.numel())

            durations = batch.durations[0, : int(batch.token_lengths[0])].cpu()
            internal_pairs = _token_internal_pair_mask(durations, frames)
            if internal_pairs.numel():
                pred_mel_delta = torch.abs(pred_mel[1:] - pred_mel[:-1])
                target_mel_delta = torch.abs(target_mel[1:] - target_mel[:-1])
                selected_pred_mel = pred_mel_delta[internal_pairs]
                selected_target_mel = target_mel_delta[internal_pairs]
                total_internal_pred_mel_delta += float(selected_pred_mel.sum())
                total_internal_target_mel_delta += float(selected_target_mel.sum())
                total_internal_mel_values += int(selected_pred_mel.numel())

                both_target_voiced = target_voiced[1:] & target_voiced[:-1] & internal_pairs
                if bool(both_target_voiced.any()):
                    pred_a = torch.clamp(pred_f0[:-1][both_target_voiced], min=1e-6)
                    pred_b = torch.clamp(pred_f0[1:][both_target_voiced], min=1e-6)
                    target_a = torch.clamp(target_f0[:-1][both_target_voiced], min=1e-6)
                    target_b = torch.clamp(target_f0[1:][both_target_voiced], min=1e-6)
                    pred_delta_cents = torch.abs(1200.0 * torch.log2(pred_b / pred_a))
                    target_delta_cents = torch.abs(1200.0 * torch.log2(target_b / target_a))
                    total_internal_pred_f0_delta_cents += float(pred_delta_cents.sum())
                    total_internal_target_f0_delta_cents += float(target_delta_cents.sum())
                    total_internal_f0_pairs += int(pred_delta_cents.numel())

            if bool(target_voiced.any()):
                pred_active = torch.clamp(pred_f0[target_voiced], min=1e-6)
                target_active = torch.clamp(target_f0[target_voiced], min=1e-6)
                pred_log = torch.log(pred_active)
                target_log = torch.log(target_active)
                cents = 1200.0 * torch.log2(pred_active / target_active)
                all_pred_log_f0.append(pred_log.cpu())
                all_target_log_f0.append(target_log.cpu())
                all_cents_error.append(cents.cpu())

            item_tp, item_tn, item_fp, item_fn = _voicing_counts(predicted_voiced, target_voiced)
            tp += item_tp
            tn += item_tn
            fp += item_fp
            fn += item_fn
            predicted_voiced_frames += int(predicted_voiced.sum())
            target_voiced_frames += int(target_voiced.sum())
            real_frames += frames

            item_metrics = _classification_metrics(item_tp, item_tn, item_fp, item_fn)
            per_utterance.append(
                {
                    "utterance_id": str(item["utterance_id"]),
                    "mel_frames": frames,
                    "mel_l1": round(float(mel_abs.mean()), 6),
                    "target_voiced_fraction": round(float(target_voiced.float().mean()), 6),
                    "predicted_voiced_fraction": round(float(predicted_voiced.float().mean()), 6),
                    "voicing_f1": round(item_metrics["f1"], 6),
                }
            )

    if not all_pred_log_f0 or not all_cents_error:
        raise RuntimeError("Held-out audit found no voiced F0 frames")

    pred_log_f0 = torch.cat(all_pred_log_f0)
    target_log_f0 = torch.cat(all_target_log_f0)
    cents_error = torch.cat(all_cents_error)
    abs_cents = torch.abs(cents_error)
    classification = _classification_metrics(tp, tn, fp, fn)

    mel_l1 = total_mel_abs / max(1, total_mel_values)
    pred_internal_mel_delta = total_internal_pred_mel_delta / max(1, total_internal_mel_values)
    target_internal_mel_delta = total_internal_target_mel_delta / max(1, total_internal_mel_values)
    pred_internal_f0_delta = (
        total_internal_pred_f0_delta_cents / max(1, total_internal_f0_pairs)
    )
    target_internal_f0_delta = (
        total_internal_target_f0_delta_cents / max(1, total_internal_f0_pairs)
    )

    # This is an architectural expressivity gate, not an arbitrary naturalness score.
    # If real held-out mel/F0 targets change inside token spans but predictions are
    # effectively constant there, end-to-end synthesis must not proceed yet.
    target_has_internal_mel_motion = target_internal_mel_delta > 1e-4
    predicted_has_internal_mel_motion = pred_internal_mel_delta > 1e-7
    target_has_internal_f0_motion = target_internal_f0_delta > 0.5
    predicted_has_internal_f0_motion = pred_internal_f0_delta > 0.05
    frame_expressivity_pass = (
        (not target_has_internal_mel_motion or predicted_has_internal_mel_motion)
        and (not target_has_internal_f0_motion or predicted_has_internal_f0_motion)
    )
    contracts_pass = exact_contract_count == len(dataset)
    status = "pass" if contracts_pass and frame_expressivity_pass else "needs_review"

    metadata = payload.get("training_metadata")
    best_epoch = None
    if isinstance(metadata, dict):
        value = metadata.get("best_epoch")
        if value is not None:
            best_epoch = int(value)

    report: dict[str, object] = {
        "status": status,
        "audit_version": AUDIT_VERSION,
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "validation_items": len(dataset),
        "exact_frame_contract_count": exact_contract_count,
        "all_frame_contracts_exact": contracts_pass,
        "mel_l1": round(mel_l1, 6),
        "f0_target_voiced_frames": int(target_log_f0.numel()),
        "f0_log_pearson": (
            None if _pearson(pred_log_f0, target_log_f0) is None
            else round(float(_pearson(pred_log_f0, target_log_f0)), 6)
        ),
        "f0_mae_cents": round(float(abs_cents.mean()), 3),
        "f0_rmse_cents": round(float(torch.sqrt(cents_error.square().mean())), 3),
        "f0_median_abs_cents": round(float(abs_cents.median()), 3),
        "target_voiced_fraction": round(target_voiced_frames / max(1, real_frames), 6),
        "predicted_voiced_fraction": round(predicted_voiced_frames / max(1, real_frames), 6),
        "voicing_precision": round(classification["precision"], 6),
        "voicing_recall": round(classification["recall"], 6),
        "voicing_f1": round(classification["f1"], 6),
        "voicing_specificity": round(classification["specificity"], 6),
        "voicing_accuracy": round(classification["accuracy"], 6),
        "voicing_balanced_accuracy": round(classification["balanced_accuracy"], 6),
        "intra_token_mel_delta_l1_target": round(target_internal_mel_delta, 8),
        "intra_token_mel_delta_l1_predicted": round(pred_internal_mel_delta, 8),
        "intra_token_f0_delta_cents_target": round(target_internal_f0_delta, 4),
        "intra_token_f0_delta_cents_predicted": round(pred_internal_f0_delta, 4),
        "target_has_intra_token_mel_motion": target_has_internal_mel_motion,
        "predicted_has_intra_token_mel_motion": predicted_has_internal_mel_motion,
        "target_has_intra_token_f0_motion": target_has_internal_f0_motion,
        "predicted_has_intra_token_f0_motion": predicted_has_internal_f0_motion,
        "frame_expressivity_pass": frame_expressivity_pass,
        "per_utterance": per_utterance,
        "next_gate": (
            "fix_post_regulation_frame_context_before_end_to_end"
            if not frame_expressivity_pass
            else "fix_predicted_duration_semantics_before_end_to_end"
        ),
        "interpretation": (
            "Teacher durations are intentionally used here to isolate the acoustic/prosody "
            "checkpoint. A needs_review caused by zero intra-token motion is structural and "
            "must be fixed before unseen-text synthesis; training longer is not the remedy."
        ),
    }

    report_path = checkpoint.parent / "heldout_audit.json"
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_acoustic_prosody_audit(args.root, checkpoint=args.checkpoint),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
