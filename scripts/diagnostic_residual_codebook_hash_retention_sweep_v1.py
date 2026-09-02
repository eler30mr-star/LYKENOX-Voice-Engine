"""Resumable TRAIN-only retention-cap sweep for the owned residual codebook.

Completed full-TRAIN retrieval coverage showed that the current 128-per-conditioning-bucket hash
retention is a dominant capacity bottleneck.  This diagnostic changes only that retention cap.
It keeps the same deterministic hash score, the same real TRAIN residual vectors, the same
512-sample / 256-hop representation, the same candidate compatibility rule, and the same exact
frozen-renderer synthesis-domain scoring used by the preceding coverage gates.

One diagnostic bank is built with ``max_per_bucket=1024`` using the existing v1 builder.  Because the
builder orders each bucket by the same deterministic hash score, its first 256, 512, and 1024 vectors
are exact nested TRAIN-only retention subsets.  Each held-out window is scored against the 1024 bank
once; the best candidates at caps 256/512/1024 are recovered from that single scoring pass.  The
completed 128-codeword and full-TRAIN per-window coverage CSVs are reused as fixed baselines.

This is not a production codebook replacement and does not train a model, create an optimizer or
checkpoint, modify the renderer, or put held-out residuals into TRAIN.  Metrics can reject but cannot
accept product quality under LYX-POL-001.  CPU only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostic_residual_codebook_oracle_v1 import (
    F0_SEARCH_RADIUS_HZ,
    PERIODICITY_BIN_RADIUS,
)
from scripts.diagnostic_residual_codebook_oracle_v4_synthesis_domain import (
    ENERGY_EPSILON,
    LOCAL_CONV_FFT_SIZE,
    _local_filtered_vector_response,
)
from scripts.diagnostic_residual_codebook_synthesis_coverage_v1 import _coverage_summary
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    POLICY_ID,
    RESIDUAL_CODEBOOK_VERSION,
    _conditioning_bucket,
    build_owned_residual_codebook,
    load_owned_residual_codebook,
    residual_analysis_vectors,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    N_FFT,
    RENDERER_VERSION,
    one_sided_real_cepstrum_to_minimum_phase_fir,
)


DIAGNOSTIC_VERSION = "owned-residual-codebook-hash-retention-sweep-v1"
MAX_RETENTION_CAP = 1024
RETENTION_CAPS = (256, 512, 1024)
DEFAULT_HELDOUT_ITEMS = 3
DEFAULT_TRAIN_ITEMS = 1_000_000
DEFAULT_CHUNK_SIZE = 512
CHECKPOINT_VERSION = "owned-residual-codebook-hash-retention-sweep-window-checkpoint-v1"
MATCH_TOLERANCE = 1.0e-5

ROW_FIELDS = [
    "utterance_id",
    "vector_index",
    "conditioning_index",
    "f0_hz",
    "voiced",
    "periodicity",
    "cap128_best_filtered_response_cosine",
    "cap128_best_normalized_mse",
    "cap256_compatible_candidate_count",
    "cap256_best_filtered_response_cosine",
    "cap256_best_normalized_mse",
    "cap512_compatible_candidate_count",
    "cap512_best_filtered_response_cosine",
    "cap512_best_normalized_mse",
    "cap1024_compatible_candidate_count",
    "cap1024_best_filtered_response_cosine",
    "cap1024_best_normalized_mse",
    "full_train_best_filtered_response_cosine",
    "full_train_best_normalized_mse",
    "cap128_reproduced_by_nested_bank",
]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _load_or_build_nested_bank(
    root: Path,
    output_dir: Path,
    *,
    max_train_items: int,
) -> tuple[torch.Tensor, dict[str, Any], bool]:
    tensor_path = output_dir / "residual_codebook_hash_retention_1024_v1.pt"
    index_path = output_dir / "residual_codebook_hash_retention_1024_v1.json"
    if tensor_path.exists() and index_path.exists():
        bank, metadata = load_owned_residual_codebook(tensor_path, index_path)
        if int(metadata.get("max_per_bucket", -1)) != MAX_RETENTION_CAP:
            raise RuntimeError("existing nested retention bank cap mismatch")
        return bank, metadata, True

    build_owned_residual_codebook(
        root,
        split="train",
        max_items=max_train_items,
        max_per_bucket=MAX_RETENTION_CAP,
        tensor_path=tensor_path,
        index_path=index_path,
    )
    bank, metadata = load_owned_residual_codebook(tensor_path, index_path)
    if int(metadata.get("max_per_bucket", -1)) != MAX_RETENTION_CAP:
        raise RuntimeError("nested retention bank was built with the wrong cap")
    return bank, metadata, False


def _bucket_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("buckets")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("retention bank has no bucket records")
    return [dict(item) for item in raw]


def _selected_bucket_records(
    metadata: dict[str, Any],
    *,
    f0_hz: float,
    voiced: float,
    periodicity: float,
) -> list[dict[str, Any]]:
    """Use exactly the existing v1 compatibility/fallback rule, returning bucket records."""

    target_state, target_f0_bin, target_p_bin = _conditioning_bucket(
        f0_hz=f0_hz,
        voiced=voiced,
        periodicity=periodicity,
    )
    buckets = _bucket_records(metadata)

    def compatible(item: dict[str, Any]) -> bool:
        if item["voicing_state"] != target_state:
            return False
        if abs(int(item["periodicity_bin"]) - target_p_bin) > PERIODICITY_BIN_RADIUS:
            return False
        if (
            target_state != "unvoiced"
            and abs(int(item["f0_bin_hz"]) - target_f0_bin) > F0_SEARCH_RADIUS_HZ
        ):
            return False
        return True

    selected = [item for item in buckets if compatible(item)]
    if not selected:
        selected = [item for item in buckets if item["voicing_state"] == target_state]
    if not selected:
        selected = buckets
    if not selected:
        raise RuntimeError("retention sweep has no compatible bucket records")
    return selected


def _allowed_indices_and_ranks(
    selected_buckets: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    indices: list[torch.Tensor] = []
    ranks: list[torch.Tensor] = []
    for item in selected_buckets:
        start = int(item["start_index"])
        count = int(item["count"])
        if count < 1:
            continue
        indices.append(torch.arange(start, start + count, dtype=torch.long))
        ranks.append(torch.arange(count, dtype=torch.long))
    if not indices:
        raise RuntimeError("retention sweep has no compatible candidate indices")
    return torch.cat(indices, dim=0), torch.cat(ranks, dim=0)


def _score_nested_caps(
    *,
    target_response: torch.Tensor,
    bank: torch.Tensor,
    allowed: torch.Tensor,
    ranks: torch.Tensor,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
    chunk_size: int,
) -> dict[int, dict[str, object]]:
    if allowed.shape != ranks.shape or int(allowed.numel()) < 1:
        raise ValueError("allowed indices and ranks must be non-empty and aligned")
    target_response = target_response.to(torch.float32).contiguous()
    target_energy = target_response.square().sum().clamp_min(ENERGY_EPSILON)
    best: dict[int, dict[str, object] | None] = {128: None, 256: None, 512: None, 1024: None}
    counts = {128: 0, 256: 0, 512: 0, 1024: 0}

    for start in range(0, int(allowed.numel()), chunk_size):
        indices = allowed[start : start + chunk_size]
        local_ranks = ranks[start : start + chunk_size]
        candidate_responses = _local_filtered_vector_response(
            bank[indices],
            vector_index=vector_index,
            filter_fft=filter_fft,
            frame_count=frame_count,
        )
        candidate_energy = candidate_responses.square().sum(dim=1).clamp_min(ENERGY_EPSILON)
        dots = torch.mv(candidate_responses, target_response)
        gains = (dots / candidate_energy).clamp_min(0.0)
        error_energy = (
            candidate_responses * gains.unsqueeze(1) - target_response.unsqueeze(0)
        ).square().sum(dim=1)
        normalized_mse = error_energy / target_energy
        cosine = dots / torch.sqrt(candidate_energy * target_energy)

        for cap in best:
            mask = local_ranks < cap
            count = int(mask.sum())
            counts[cap] += count
            if count < 1:
                continue
            positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
            masked_mse = normalized_mse[positions]
            local = int(torch.argmin(masked_mse))
            position = int(positions[local])
            record = {
                "index": int(indices[position]),
                "gain": float(gains[position]),
                "cosine": float(cosine[position]),
                "normalized_mse": float(normalized_mse[position]),
            }
            previous = best[cap]
            if previous is None or float(record["normalized_mse"]) < float(previous["normalized_mse"]):
                best[cap] = record

    result: dict[int, dict[str, object]] = {}
    for cap, record in best.items():
        if record is None or counts[cap] < 1:
            raise RuntimeError(f"retention cap {cap} produced no candidate")
        record["candidate_count"] = counts[cap]
        result[cap] = record
    return result


def _load_baseline_csv(path: Path, *, cosine_field: str, mse_field: str) -> dict[int, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows[int(raw["vector_index"])] = {
                "cosine": float(raw[cosine_field]),
                "normalized_mse": float(raw[mse_field]),
            }
    if not rows:
        raise RuntimeError(f"baseline CSV contains no rows: {path}")
    return rows


def _load_cap128_baseline(root: Path, utterance_id: str) -> dict[int, dict[str, float]]:
    path = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_residual_codebook_synthesis_coverage_v1"
        / f"{utterance_id}__synthesis_coverage_windows.csv"
    )
    return _load_baseline_csv(
        path,
        cosine_field="all_compatible_best_filtered_response_cosine",
        mse_field="all_compatible_best_normalized_mse",
    )


def _load_full_train_baseline(root: Path, utterance_id: str) -> dict[int, dict[str, float]]:
    path = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_full_train_residual_retrieval_coverage_v1"
        / f"{utterance_id}__full_train_coverage_windows.csv"
    )
    return _load_baseline_csv(
        path,
        cosine_field="full_train_best_filtered_response_cosine",
        mse_field="full_train_best_normalized_mse",
    )


def _load_checkpoint(path: Path, *, utterance_id: str) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    result: dict[int, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise RuntimeError("retention sweep checkpoint schema mismatch")
        for raw in reader:
            if raw["utterance_id"] != utterance_id:
                raise RuntimeError("retention sweep checkpoint utterance mismatch")
            row: dict[str, object] = {key: raw[key] for key in ROW_FIELDS}
            row["vector_index"] = int(raw["vector_index"])
            row["conditioning_index"] = int(raw["conditioning_index"])
            for key in ROW_FIELDS:
                if key in {
                    "utterance_id",
                    "vector_index",
                    "conditioning_index",
                    "cap128_reproduced_by_nested_bank",
                }:
                    continue
                if key.endswith("candidate_count"):
                    row[key] = int(raw[key])
                else:
                    row[key] = float(raw[key])
            row["cap128_reproduced_by_nested_bank"] = raw["cap128_reproduced_by_nested_bank"] == "True"
            result[int(row["vector_index"])] = row
    return result


def _append_checkpoint(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _coverage_for_rows(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    return _coverage_summary(
        [float(row[f"{prefix}_best_filtered_response_cosine"]) for row in rows],
        [float(row[f"{prefix}_best_normalized_mse"]) for row in rows],
    )


def _gap_recovery(rows: list[dict[str, object]], cap_prefix: str) -> dict[str, float]:
    baseline = sum(float(row["cap128_best_normalized_mse"]) for row in rows)
    candidate = sum(float(row[f"{cap_prefix}_best_normalized_mse"]) for row in rows)
    full = sum(float(row["full_train_best_normalized_mse"]) for row in rows)
    denominator = baseline - full
    recovered = baseline - candidate
    return {
        "total_nmse_cap128": baseline,
        "total_nmse_candidate": candidate,
        "total_nmse_full_train": full,
        "fraction_of_128_to_full_train_nmse_gap_recovered": (
            recovered / denominator if denominator > 1.0e-12 else 0.0
        ),
        "fraction_windows_improved_vs_128": sum(
            1
            for row in rows
            if float(row[f"{cap_prefix}_best_normalized_mse"])
            < float(row["cap128_best_normalized_mse"]) - 1.0e-9
        )
        / float(len(rows)),
    }


def _finalize_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_retention_sweep(
    root: Path,
    *,
    heldout_items: int = DEFAULT_HELDOUT_ITEMS,
    max_train_items: int = DEFAULT_TRAIN_ITEMS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    utterance_index: int | None = None,
    max_new_windows: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if heldout_items < 1 or heldout_items > 3:
        raise ValueError("heldout_items must be between 1 and 3")
    if chunk_size < 1 or chunk_size > 2048:
        raise ValueError("chunk_size must be in [1, 2048]")
    if utterance_index is not None and not 0 <= utterance_index < heldout_items:
        raise ValueError("utterance_index must select one requested held-out item")
    if max_new_windows is not None and max_new_windows < 1:
        raise ValueError("max_new_windows must be positive")

    root = Path(root).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_residual_codebook_hash_retention_sweep_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bank, metadata, reused_bank = _load_or_build_nested_bank(
        root,
        output_dir,
        max_train_items=max_train_items,
    )
    if metadata.get("codebook_version") != RESIDUAL_CODEBOOK_VERSION:
        raise RuntimeError("nested retention bank codebook version mismatch")
    if metadata.get("source_split") != "train":
        raise RuntimeError("nested retention bank is not TRAIN-only")

    heldout = collect_owned_vocoder_utterances(root, split="val", max_items=heldout_items)
    selected_items = [utterance_index] if utterance_index is not None else list(range(len(heldout)))
    new_windows_done = 0
    incomplete: list[dict[str, object]] = []

    with torch.no_grad():
        for item_index in selected_items:
            utterance = heldout[item_index]
            frame_count = int(utterance.mel_frames)
            expected_windows = frame_count + 1
            cap128_baseline = _load_cap128_baseline(root, utterance.utterance_id)
            full_baseline = _load_full_train_baseline(root, utterance.utterance_id)
            if len(cap128_baseline) != expected_windows or len(full_baseline) != expected_windows:
                raise RuntimeError("retention sweep baseline geometry mismatch")

            checkpoint = output_dir / f"{utterance.utterance_id}__retention_sweep_windows.partial.csv"
            completed = _load_checkpoint(checkpoint, utterance_id=utterance.utterance_id)

            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            target_residual, cepstrum, _ = extract_owned_real_residual(
                reference,
                frame_count=frame_count,
            )
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("retention sweep cepstrum geometry changed")
            targets = residual_analysis_vectors(target_residual)
            if int(targets.shape[0]) != expected_windows:
                raise RuntimeError("retention sweep residual vector geometry changed")

            filters = one_sided_real_cepstrum_to_minimum_phase_fir(
                cepstrum,
                n_fft=N_FFT,
            ).to(torch.float32)
            filter_fft = torch.fft.rfft(filters, n=LOCAL_CONV_FFT_SIZE, dim=-1).contiguous()

            for vector_index in range(expected_windows):
                if vector_index in completed:
                    continue
                if max_new_windows is not None and new_windows_done >= max_new_windows:
                    break

                conditioning_index = min(vector_index, frame_count - 1)
                f0_hz = float(utterance.f0_hz[conditioning_index])
                voiced = float(utterance.voiced[conditioning_index])
                periodicity = float(utterance.periodicity[conditioning_index])
                buckets = _selected_bucket_records(
                    metadata,
                    f0_hz=f0_hz,
                    voiced=voiced,
                    periodicity=periodicity,
                )
                allowed, ranks = _allowed_indices_and_ranks(buckets)
                target_response = _local_filtered_vector_response(
                    targets[vector_index].unsqueeze(0),
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                ).squeeze(0)
                nested = _score_nested_caps(
                    target_response=target_response,
                    bank=bank,
                    allowed=allowed,
                    ranks=ranks,
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                    chunk_size=chunk_size,
                )
                baseline128 = cap128_baseline[vector_index]
                baseline_full = full_baseline[vector_index]
                nested128 = nested[128]
                reproduced = (
                    abs(float(nested128["normalized_mse"]) - baseline128["normalized_mse"])
                    <= MATCH_TOLERANCE
                    and abs(float(nested128["cosine"]) - baseline128["cosine"])
                    <= MATCH_TOLERANCE
                )
                row = {
                    "utterance_id": utterance.utterance_id,
                    "vector_index": vector_index,
                    "conditioning_index": conditioning_index,
                    "f0_hz": f0_hz,
                    "voiced": voiced,
                    "periodicity": periodicity,
                    "cap128_best_filtered_response_cosine": baseline128["cosine"],
                    "cap128_best_normalized_mse": baseline128["normalized_mse"],
                    "cap256_compatible_candidate_count": int(nested[256]["candidate_count"]),
                    "cap256_best_filtered_response_cosine": float(nested[256]["cosine"]),
                    "cap256_best_normalized_mse": float(nested[256]["normalized_mse"]),
                    "cap512_compatible_candidate_count": int(nested[512]["candidate_count"]),
                    "cap512_best_filtered_response_cosine": float(nested[512]["cosine"]),
                    "cap512_best_normalized_mse": float(nested[512]["normalized_mse"]),
                    "cap1024_compatible_candidate_count": int(nested[1024]["candidate_count"]),
                    "cap1024_best_filtered_response_cosine": float(nested[1024]["cosine"]),
                    "cap1024_best_normalized_mse": float(nested[1024]["normalized_mse"]),
                    "full_train_best_filtered_response_cosine": baseline_full["cosine"],
                    "full_train_best_normalized_mse": baseline_full["normalized_mse"],
                    "cap128_reproduced_by_nested_bank": reproduced,
                }
                _append_checkpoint(checkpoint, row)
                completed[vector_index] = row
                new_windows_done += 1
                print(
                    json.dumps(
                        {
                            "status": "checkpointed_retention_sweep_window",
                            "utterance_index": item_index,
                            "utterance_id": utterance.utterance_id,
                            "vector_index": vector_index,
                            "completed_windows": len(completed),
                            "expected_windows": expected_windows,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if len(completed) == expected_windows:
                ordered = [completed[index] for index in range(expected_windows)]
                _finalize_csv(
                    output_dir / f"{utterance.utterance_id}__retention_sweep_windows.csv",
                    ordered,
                )
            else:
                incomplete.append(
                    {
                        "utterance_index": item_index,
                        "utterance_id": utterance.utterance_id,
                        "completed_windows": len(completed),
                        "expected_windows": expected_windows,
                        "checkpoint": str(checkpoint),
                    }
                )

            if max_new_windows is not None and new_windows_done >= max_new_windows:
                break

    progress: dict[str, object] = {
        "status": "retention_sweep_in_progress" if incomplete else "selected_items_complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "reused_existing_nested_bank": reused_bank,
        "nested_bank_retained_count": int(bank.shape[0]),
        "nested_bank_max_per_bucket": metadata.get("max_per_bucket"),
        "original_train_candidate_window_count": metadata.get("candidate_window_count"),
        "new_windows_completed_this_run": new_windows_done,
        "incomplete_items": incomplete,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "production_codebook_replaced": False,
    }
    _atomic_json(output_dir / "residual_codebook_hash_retention_sweep_v1_progress.json", progress)

    all_rows: list[dict[str, object]] = []
    complete = True
    for utterance in heldout:
        final_csv = output_dir / f"{utterance.utterance_id}__retention_sweep_windows.csv"
        if not final_csv.exists():
            complete = False
            break
        all_rows.extend(_load_checkpoint(final_csv, utterance_id=utterance.utterance_id).values())

    if not complete:
        return progress

    all_rows.sort(key=lambda row: (str(row["utterance_id"]), int(row["vector_index"])))
    reproduction_failures = sum(
        1 for row in all_rows if not bool(row["cap128_reproduced_by_nested_bank"])
    )
    report: dict[str, object] = {
        "status": "ready_for_hash_retention_capacity_curve_review",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "heldout_window_count": len(all_rows),
        "original_train_candidate_window_count": metadata.get("candidate_window_count"),
        "nested_bank_retained_count": int(bank.shape[0]),
        "nested_bank_bucket_count": metadata.get("bucket_count"),
        "nested_bank_max_per_bucket": metadata.get("max_per_bucket"),
        "cap128_baseline_reproduction_failure_count": reproduction_failures,
        "cap128_baseline_reproduction_pass": reproduction_failures == 0,
        "coverage": {
            "cap128": _coverage_for_rows(all_rows, "cap128"),
            "cap256": _coverage_for_rows(all_rows, "cap256"),
            "cap512": _coverage_for_rows(all_rows, "cap512"),
            "cap1024": _coverage_for_rows(all_rows, "cap1024"),
            "full_train": _coverage_for_rows(all_rows, "full_train"),
        },
        "gap_recovery": {
            "cap256": _gap_recovery(all_rows, "cap256"),
            "cap512": _gap_recovery(all_rows, "cap512"),
            "cap1024": _gap_recovery(all_rows, "cap1024"),
        },
        "retention_rule": "existing_deterministic_hash_rule_only_cap_changes",
        "heldout_used_to_select_individual_codewords": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "production_codebook_replaced": False,
        "metrics_can_accept_product_quality": False,
        "next_action": (
            "review_size_coverage_curve;_if_1024_recovers_most_full_train_gap_then_test_audio_with_"
            "larger_train_only_codebook;_otherwise_design_train_only_signal_diversity_retention"
        ),
    }
    _atomic_json(output_dir / "residual_codebook_hash_retention_sweep_v1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_HELDOUT_ITEMS)
    parser.add_argument("--max-train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--utterance-index", type=int, default=None)
    parser.add_argument("--max-new-windows", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_retention_sweep(
                args.root,
                heldout_items=args.heldout_items,
                max_train_items=args.max_train_items,
                chunk_size=args.chunk_size,
                utterance_index=args.utterance_index,
                max_new_windows=args.max_new_windows,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
