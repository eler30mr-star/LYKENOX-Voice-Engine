"""Resumable CPU audit for full-TRAIN residual retrieval coverage.

V1 established the correct full-TRAIN isolation but is too expensive to restart from zero on the
project laptop.  This V2 preserves the same exact candidate compatibility, frozen-renderer local
response, and non-negative least-squares synthesis-domain scoring while changing only execution
engineering:

* reuse the already-built diagnostic full-TRAIN bank when present;
* reuse the completed retained-codebook exhaustive coverage CSVs as the sampled baseline;
* compute the held-out target filtered response once per window rather than once per candidate chunk;
* checkpoint every completed held-out window to a resumable CSV;
* skip completed windows on restart;
* optionally process one held-out utterance at a time;
* never replace the production codebook, train a model, or create a checkpoint.

The full TRAIN bank remains diagnostic-only and contains owned TRAIN residual vectors only. Held-out
residuals are targets only. Metrics can reject but cannot accept product quality under LYX-POL-001.
CPU only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostic_full_train_residual_retrieval_coverage_v1 import (
    FULL_BANK_VERSION,
    _build_full_train_bank,
)
from scripts.diagnostic_residual_codebook_oracle_v1 import _candidate_indices
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


DIAGNOSTIC_VERSION = "owned-full-train-residual-retrieval-synthesis-coverage-v2-resumable"
DEFAULT_HELDOUT_ITEMS = 3
DEFAULT_TRAIN_ITEMS = 1_000_000
DEFAULT_CHUNK_SIZE = 512
CHECKPOINT_VERSION = "owned-full-train-coverage-window-checkpoint-v1"

ROW_FIELDS = [
    "utterance_id",
    "vector_index",
    "conditioning_index",
    "f0_hz",
    "voiced",
    "periodicity",
    "sampled_compatible_candidate_count",
    "full_train_compatible_candidate_count",
    "sampled_best_filtered_response_cosine",
    "sampled_best_normalized_mse",
    "full_train_best_filtered_response_cosine",
    "full_train_best_normalized_mse",
    "normalized_mse_improvement_full_vs_sampled",
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


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError("full TRAIN retrieval bank tensor has invalid geometry")
    value = value.detach().cpu().to(torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("full TRAIN retrieval bank contains non-finite values")
    return value


def _load_or_build_full_bank(
    root: Path,
    output_dir: Path,
    *,
    max_train_items: int,
) -> tuple[torch.Tensor, dict[str, Any], bool]:
    bank_path = output_dir / "full_train_residual_retrieval_bank_v1.pt"
    metadata_path = output_dir / "full_train_residual_retrieval_bank_v1.json"
    if bank_path.exists() and metadata_path.exists():
        metadata = _load_json(metadata_path)
        if metadata.get("bank_version") != FULL_BANK_VERSION:
            raise RuntimeError("existing full TRAIN bank version mismatch")
        if metadata.get("policy_id") != POLICY_ID or metadata.get("source_split") != "train":
            raise RuntimeError("existing full TRAIN bank provenance mismatch")
        if metadata.get("heldout_data_in_bank") is not False:
            raise RuntimeError("existing full TRAIN bank does not prove held-out exclusion")
        bank = _load_tensor(bank_path)
        expected = int(metadata.get("retained_window_count", -1))
        if expected != int(bank.shape[0]):
            raise RuntimeError("existing full TRAIN bank tensor/metadata count mismatch")
        return bank, metadata, True

    bank, metadata = _build_full_train_bank(root, max_train_items=max_train_items)
    temporary = bank_path.with_suffix(bank_path.suffix + ".tmp")
    torch.save(bank, temporary)
    os.replace(temporary, bank_path)
    _atomic_json(metadata_path, metadata)
    return bank, metadata, False


def _load_sampled_baseline(
    root: Path,
    utterance_id: str,
) -> dict[int, dict[str, object]]:
    csv_path = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_residual_codebook_synthesis_coverage_v1"
        / f"{utterance_id}__synthesis_coverage_windows.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(
            "completed retained-codebook coverage CSV is required for resumable full-TRAIN audit: "
            + str(csv_path)
        )
    result: dict[int, dict[str, object]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["vector_index"])
            result[index] = {
                "candidate_count": int(row["compatible_candidate_count"]),
                "cosine": float(row["all_compatible_best_filtered_response_cosine"]),
                "normalized_mse": float(row["all_compatible_best_normalized_mse"]),
            }
    if not result:
        raise RuntimeError("sampled baseline CSV contains no windows")
    return result


def _load_checkpoint_rows(path: Path, *, utterance_id: str) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise RuntimeError("resumable full-TRAIN checkpoint schema mismatch")
        for raw in reader:
            if raw["utterance_id"] != utterance_id:
                raise RuntimeError("resumable checkpoint contains a different utterance")
            index = int(raw["vector_index"])
            rows[index] = {
                "utterance_id": raw["utterance_id"],
                "vector_index": index,
                "conditioning_index": int(raw["conditioning_index"]),
                "f0_hz": float(raw["f0_hz"]),
                "voiced": float(raw["voiced"]),
                "periodicity": float(raw["periodicity"]),
                "sampled_compatible_candidate_count": int(raw["sampled_compatible_candidate_count"]),
                "full_train_compatible_candidate_count": int(raw["full_train_compatible_candidate_count"]),
                "sampled_best_filtered_response_cosine": float(raw["sampled_best_filtered_response_cosine"]),
                "sampled_best_normalized_mse": float(raw["sampled_best_normalized_mse"]),
                "full_train_best_filtered_response_cosine": float(raw["full_train_best_filtered_response_cosine"]),
                "full_train_best_normalized_mse": float(raw["full_train_best_normalized_mse"]),
                "normalized_mse_improvement_full_vs_sampled": float(raw["normalized_mse_improvement_full_vs_sampled"]),
            }
    return rows


def _append_checkpoint_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _best_full_train_against_target_response(
    *,
    target_response: torch.Tensor,
    bank: torch.Tensor,
    allowed: torch.Tensor,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
    chunk_size: int,
) -> dict[str, object]:
    if int(allowed.numel()) < 1:
        raise RuntimeError("full TRAIN coverage has no compatible candidates")
    target_response = target_response.to(torch.float32).contiguous()
    target_energy = target_response.square().sum().clamp_min(ENERGY_EPSILON)
    best: dict[str, object] | None = None

    for start in range(0, int(allowed.numel()), chunk_size):
        indices = allowed[start : start + chunk_size]
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
        local = int(torch.argmin(normalized_mse))
        record = {
            "index": int(indices[local]),
            "gain": float(gains[local]),
            "cosine": float(cosine[local]),
            "normalized_mse": float(normalized_mse[local]),
        }
        if best is None or float(record["normalized_mse"]) < float(best["normalized_mse"]):
            best = record

    if best is None:
        raise RuntimeError("full TRAIN coverage failed to select a candidate")
    best["candidate_count"] = int(allowed.numel())
    return best


def _finalize_utterance(
    *,
    output_dir: Path,
    utterance_id: str,
    frame_count: int,
    extension_frames: int,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    final_csv = output_dir / f"{utterance_id}__full_train_coverage_windows.csv"
    temporary = final_csv.with_suffix(final_csv.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, final_csv)

    worst = sorted(
        rows,
        key=lambda item: (
            float(item["full_train_best_filtered_response_cosine"]),
            -float(item["full_train_best_normalized_mse"]),
            int(item["vector_index"]),
        ),
    )[:20]
    return {
        "utterance_id": utterance_id,
        "conditioning_frames": frame_count,
        "terminal_transfer_extension_frames": extension_frames,
        "window_count": len(rows),
        "sampled_codebook_coverage": _coverage_summary(
            [float(row["sampled_best_filtered_response_cosine"]) for row in rows],
            [float(row["sampled_best_normalized_mse"]) for row in rows],
        ),
        "full_train_retrieval_coverage": _coverage_summary(
            [float(row["full_train_best_filtered_response_cosine"]) for row in rows],
            [float(row["full_train_best_normalized_mse"]) for row in rows],
        ),
        "worst_20_full_train_windows": worst,
        "csv": str(final_csv),
    }


def run_resumable_full_train_coverage(
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
        / "vocoder_minimum_phase_full_train_residual_retrieval_coverage_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    full_bank, full_metadata, reused_existing_bank = _load_or_build_full_bank(
        root,
        output_dir,
        max_train_items=max_train_items,
    )
    sampled_metadata = _load_json(
        root / "models" / "lykenox_identity" / "calibration" / "residual_codebook_v1.json"
    )
    if sampled_metadata.get("codebook_version") != RESIDUAL_CODEBOOK_VERSION:
        raise RuntimeError("sampled codebook metadata version mismatch")

    heldout = collect_owned_vocoder_utterances(root, split="val", max_items=heldout_items)
    selected_indices = [utterance_index] if utterance_index is not None else list(range(len(heldout)))
    new_windows_done = 0
    item_reports: list[dict[str, object]] = []
    incomplete_items: list[dict[str, object]] = []

    with torch.no_grad():
        for item_index in selected_indices:
            utterance = heldout[item_index]
            frame_count = int(utterance.mel_frames)
            expected_windows = frame_count + 1
            baseline = _load_sampled_baseline(root, utterance.utterance_id)
            if len(baseline) != expected_windows:
                raise RuntimeError("sampled baseline window count does not match held-out geometry")

            checkpoint_path = output_dir / f"{utterance.utterance_id}__full_train_coverage_windows.partial.csv"
            completed = _load_checkpoint_rows(checkpoint_path, utterance_id=utterance.utterance_id)

            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            target_residual, cepstrum, extension_frames = extract_owned_real_residual(
                reference,
                frame_count=frame_count,
            )
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("held-out resumable coverage cepstrum geometry changed")
            targets = residual_analysis_vectors(target_residual)
            if int(targets.shape[0]) != expected_windows:
                raise RuntimeError("held-out resumable coverage residual geometry changed")

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
                full_allowed = _candidate_indices(
                    full_metadata,
                    f0_hz=f0_hz,
                    voiced=voiced,
                    periodicity=periodicity,
                )

                target_response = _local_filtered_vector_response(
                    targets[vector_index].unsqueeze(0),
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                ).squeeze(0)
                full = _best_full_train_against_target_response(
                    target_response=target_response,
                    bank=full_bank,
                    allowed=full_allowed,
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                    chunk_size=chunk_size,
                )
                sampled = baseline[vector_index]
                improvement = float(sampled["normalized_mse"]) - float(full["normalized_mse"])
                row = {
                    "utterance_id": utterance.utterance_id,
                    "vector_index": vector_index,
                    "conditioning_index": conditioning_index,
                    "f0_hz": f0_hz,
                    "voiced": voiced,
                    "periodicity": periodicity,
                    "sampled_compatible_candidate_count": int(sampled["candidate_count"]),
                    "full_train_compatible_candidate_count": int(full["candidate_count"]),
                    "sampled_best_filtered_response_cosine": float(sampled["cosine"]),
                    "sampled_best_normalized_mse": float(sampled["normalized_mse"]),
                    "full_train_best_filtered_response_cosine": float(full["cosine"]),
                    "full_train_best_normalized_mse": float(full["normalized_mse"]),
                    "normalized_mse_improvement_full_vs_sampled": improvement,
                }
                _append_checkpoint_row(checkpoint_path, row)
                completed[vector_index] = row
                new_windows_done += 1
                print(
                    json.dumps(
                        {
                            "status": "checkpointed_window",
                            "utterance_index": item_index,
                            "utterance_id": utterance.utterance_id,
                            "vector_index": vector_index,
                            "completed_windows": len(completed),
                            "expected_windows": expected_windows,
                            "chunk_size": chunk_size,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if len(completed) == expected_windows:
                ordered_rows = [completed[index] for index in range(expected_windows)]
                item_reports.append(
                    _finalize_utterance(
                        output_dir=output_dir,
                        utterance_id=utterance.utterance_id,
                        frame_count=frame_count,
                        extension_frames=extension_frames,
                        rows=ordered_rows,
                    )
                )
            else:
                incomplete_items.append(
                    {
                        "utterance_index": item_index,
                        "utterance_id": utterance.utterance_id,
                        "completed_windows": len(completed),
                        "expected_windows": expected_windows,
                        "checkpoint": str(checkpoint_path),
                    }
                )

            if max_new_windows is not None and new_windows_done >= max_new_windows:
                break

    progress = {
        "status": "full_train_residual_retrieval_coverage_in_progress"
        if incomplete_items
        else "selected_heldout_items_complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "reused_existing_full_train_bank": reused_existing_bank,
        "full_train_retrieval_window_count": int(full_bank.shape[0]),
        "chunk_size": chunk_size,
        "new_windows_completed_this_run": new_windows_done,
        "completed_items": [item["utterance_id"] for item in item_reports],
        "incomplete_items": incomplete_items,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "production_codebook_replaced": False,
    }
    _atomic_json(output_dir / "full_train_residual_retrieval_coverage_v2_progress.json", progress)

    # Build the final report only when all requested held-out utterances have completed final CSVs.
    all_item_reports: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    all_complete = True
    for utterance in heldout:
        final_csv = output_dir / f"{utterance.utterance_id}__full_train_coverage_windows.csv"
        if not final_csv.exists():
            all_complete = False
            break
        rows: list[dict[str, object]] = []
        with final_csv.open("r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                rows.append(
                    {
                        "utterance_id": raw["utterance_id"],
                        "vector_index": int(raw["vector_index"]),
                        "conditioning_index": int(raw["conditioning_index"]),
                        "f0_hz": float(raw["f0_hz"]),
                        "voiced": float(raw["voiced"]),
                        "periodicity": float(raw["periodicity"]),
                        "sampled_compatible_candidate_count": int(raw["sampled_compatible_candidate_count"]),
                        "full_train_compatible_candidate_count": int(raw["full_train_compatible_candidate_count"]),
                        "sampled_best_filtered_response_cosine": float(raw["sampled_best_filtered_response_cosine"]),
                        "sampled_best_normalized_mse": float(raw["sampled_best_normalized_mse"]),
                        "full_train_best_filtered_response_cosine": float(raw["full_train_best_filtered_response_cosine"]),
                        "full_train_best_normalized_mse": float(raw["full_train_best_normalized_mse"]),
                        "normalized_mse_improvement_full_vs_sampled": float(raw["normalized_mse_improvement_full_vs_sampled"]),
                    }
                )
        if len(rows) != int(utterance.mel_frames) + 1:
            all_complete = False
            break
        rows.sort(key=lambda item: int(item["vector_index"]))
        all_rows.extend(rows)
        all_item_reports.append(
            {
                "utterance_id": utterance.utterance_id,
                "conditioning_frames": int(utterance.mel_frames),
                "window_count": len(rows),
                "sampled_codebook_coverage": _coverage_summary(
                    [float(row["sampled_best_filtered_response_cosine"]) for row in rows],
                    [float(row["sampled_best_normalized_mse"]) for row in rows],
                ),
                "full_train_retrieval_coverage": _coverage_summary(
                    [float(row["full_train_best_filtered_response_cosine"]) for row in rows],
                    [float(row["full_train_best_normalized_mse"]) for row in rows],
                ),
                "csv": str(final_csv),
            }
        )

    if all_complete and all_rows:
        sampled_cosines = [float(row["sampled_best_filtered_response_cosine"]) for row in all_rows]
        sampled_mse = [float(row["sampled_best_normalized_mse"]) for row in all_rows]
        full_cosines = [float(row["full_train_best_filtered_response_cosine"]) for row in all_rows]
        full_mse = [float(row["full_train_best_normalized_mse"]) for row in all_rows]
        improvement_count = sum(
            1 for row in all_rows if float(row["normalized_mse_improvement_full_vs_sampled"]) > 1.0e-9
        )
        report: dict[str, object] = {
            "status": "ready_for_full_train_residual_retrieval_coverage_review",
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "policy_id": POLICY_ID,
            "device": "cpu",
            "renderer_version": RENDERER_VERSION,
            "sampled_codebook_version": RESIDUAL_CODEBOOK_VERSION,
            "sampled_codebook_retained_count": int(sampled_metadata.get("retained_codeword_count", -1)),
            "sampled_codebook_max_per_bucket": sampled_metadata.get("max_per_bucket"),
            "sampled_codebook_original_candidate_window_count": sampled_metadata.get("candidate_window_count"),
            "full_train_retrieval_window_count": int(full_bank.shape[0]),
            "full_train_bucket_count": int(full_metadata["bucket_count"]),
            "heldout_window_count": len(all_rows),
            "full_train_improvement_count": improvement_count,
            "full_train_improvement_fraction": improvement_count / float(len(all_rows)),
            "sampled_codebook_coverage": _coverage_summary(sampled_cosines, sampled_mse),
            "full_train_retrieval_coverage": _coverage_summary(full_cosines, full_mse),
            "full_bank_artifact": str(output_dir / "full_train_residual_retrieval_bank_v1.pt"),
            "reused_existing_full_train_bank": reused_existing_bank,
            "resumable_window_checkpoints_used": True,
            "sampled_baseline_recomputed": False,
            "heldout_residual_added_to_full_bank": False,
            "training_executed": False,
            "optimizer_created": False,
            "checkpoint_written": False,
            "production_codebook_replaced": False,
            "metrics_can_accept_product_quality": False,
            "items": all_item_reports,
            "next_action": (
                "if_full_train_materially_closes_coverage_gap_then_redesign_codebook_retention;_"
                "otherwise_reject_current_cross_utterance_512_256_codevector_capacity_before_training"
            ),
        }
        _atomic_json(output_dir / "full_train_residual_retrieval_coverage_v2_report.json", report)
        return report

    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_HELDOUT_ITEMS)
    parser.add_argument("--max-train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--utterance-index", type=int, default=None)
    parser.add_argument("--max-new-windows", type=int, default=None)
    args = parser.parse_args()
    result = run_resumable_full_train_coverage(
        args.root,
        heldout_items=args.heldout_items,
        max_train_items=args.max_train_items,
        chunk_size=args.chunk_size,
        utterance_index=args.utterance_index,
        max_new_windows=args.max_new_windows,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
