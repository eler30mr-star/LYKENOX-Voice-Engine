"""Full-TRAIN residual retrieval coverage audit for the owned LYKENOX codebook line.

The retained residual codebook v1 is a deterministic compression: it keeps at most 128 real
TRAIN residual vectors per conditioning bucket. The preceding synthesis-coverage audit showed that
searching every retained compatible codeword improves over PRESELECT_K, but still leaves substantial
held-out coverage gaps. This diagnostic isolates whether those gaps come from the 128-per-bucket
compression or from the 512/256 residual-codevector representation itself.

It builds a diagnostic-only retrieval bank containing *every* owned TRAIN residual analysis vector,
with the same 512-sample sqrt-Hann / 256-hop representation and the same conditioning buckets. It
then evaluates held-out VAL windows in the exact same frozen-renderer local waveform domain and with
the same non-negative least-squares gain used by V4/V5/coverage-v1. Candidate compatibility uses the
same oracle rule: voicing state, periodicity +/-1 bin, F0 +/-40 Hz, with the same fallbacks.

The full retrieval bank is written under evaluation only and does not replace residual_codebook_v1,
authorize production use, train a model, create an optimizer/checkpoint, or admit held-out residuals
into TRAIN. Metrics can reject but cannot accept product quality under LYX-POL-001. CPU only.
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

from scripts.diagnostic_residual_codebook_oracle_v1 import _candidate_indices
from scripts.diagnostic_residual_codebook_synthesis_coverage_v1 import (
    DEFAULT_CHUNK_SIZE,
    _coverage_summary,
    _score_candidates,
)
from scripts.diagnostic_residual_codebook_oracle_v4_synthesis_domain import (
    LOCAL_CONV_FFT_SIZE,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    CODEVECTOR_SAMPLES,
    POLICY_ID,
    RESIDUAL_CODEBOOK_VERSION,
    _bucket_key,
    _conditioning_bucket,
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
    SAMPLE_RATE,
    one_sided_real_cepstrum_to_minimum_phase_fir,
)


DIAGNOSTIC_VERSION = "owned-full-train-residual-retrieval-synthesis-coverage-v1"
FULL_BANK_VERSION = "owned-full-train-residual-retrieval-bank-v1"
DEFAULT_HELDOUT_ITEMS = 3
DEFAULT_TRAIN_ITEMS = 1_000_000


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _build_full_train_bank(root: Path, *, max_train_items: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if max_train_items < 1:
        raise ValueError("max_train_items must be positive")
    train = collect_owned_vocoder_utterances(root, split="train", max_items=max_train_items)
    bucket_chunks: dict[tuple[str, int, int], list[torch.Tensor]] = {}
    source_utterances: list[dict[str, object]] = []
    candidate_window_count = 0

    with torch.no_grad():
        for utterance in train:
            frame_count = int(utterance.mel_frames)
            residual, _, extension_frames = extract_owned_real_residual(
                utterance.waveform.cpu(),
                frame_count=frame_count,
            )
            vectors = residual_analysis_vectors(residual)
            if int(vectors.shape[0]) != frame_count + 1:
                raise RuntimeError("full TRAIN residual vector geometry changed")

            by_bucket: dict[tuple[str, int, int], list[int]] = {}
            for frame_index in range(frame_count + 1):
                conditioning_index = min(frame_index, frame_count - 1)
                bucket = _conditioning_bucket(
                    f0_hz=float(utterance.f0_hz[conditioning_index]),
                    voiced=float(utterance.voiced[conditioning_index]),
                    periodicity=float(utterance.periodicity[conditioning_index]),
                )
                by_bucket.setdefault(bucket, []).append(frame_index)
                candidate_window_count += 1

            for bucket, frame_indices in by_bucket.items():
                index_tensor = torch.tensor(frame_indices, dtype=torch.long)
                bucket_chunks.setdefault(bucket, []).append(
                    vectors[index_tensor].detach().cpu().to(torch.float32).contiguous()
                )

            source_utterances.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "conditioning_frames": frame_count,
                    "candidate_windows": frame_count + 1,
                    "pitch_cache_version": utterance.pitch_cache_version,
                    "conditioning_contract_version": utterance.conditioning_contract_version,
                    "terminal_transfer_extension_frames": extension_frames,
                }
            )

    if not bucket_chunks:
        raise RuntimeError("no owned TRAIN residual vectors available for full retrieval bank")

    all_vectors: list[torch.Tensor] = []
    bucket_records: list[dict[str, object]] = []
    offset = 0
    for bucket in sorted(bucket_chunks, key=_bucket_key):
        stacked = torch.cat(bucket_chunks[bucket], dim=0).to(torch.float32).contiguous()
        if int(stacked.shape[1]) != CODEVECTOR_SAMPLES:
            raise RuntimeError("full TRAIN retrieval bank codevector geometry changed")
        count = int(stacked.shape[0])
        state, f0_bin, periodicity_bin = bucket
        bucket_records.append(
            {
                "key": _bucket_key(bucket),
                "voicing_state": state,
                "f0_bin_hz": f0_bin,
                "periodicity_bin": periodicity_bin,
                "start_index": offset,
                "count": count,
            }
        )
        all_vectors.append(stacked)
        offset += count

    bank = torch.cat(all_vectors, dim=0).contiguous()
    if int(bank.shape[0]) != candidate_window_count:
        raise RuntimeError("full TRAIN retrieval bank lost candidate windows")
    if not bool(torch.isfinite(bank).all()):
        raise RuntimeError("full TRAIN retrieval bank contains non-finite values")

    metadata: dict[str, Any] = {
        "status": "built_diagnostic_full_owned_train_retrieval_bank",
        "bank_version": FULL_BANK_VERSION,
        "policy_id": POLICY_ID,
        "source_split": "train",
        "artifact_role": "diagnostic_full_train_retrieval_bank_not_production_codebook",
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "heldout_data_in_bank": False,
        "third_party_voice_data_used": False,
        "renderer_version": RENDERER_VERSION,
        "sample_rate": SAMPLE_RATE,
        "codevector_samples": CODEVECTOR_SAMPLES,
        "candidate_window_count": candidate_window_count,
        "retained_window_count": int(bank.shape[0]),
        "bucket_count": len(bucket_records),
        "max_per_bucket": None,
        "buckets": bucket_records,
        "source_utterance_count": len(source_utterances),
        "source_utterances": source_utterances,
    }
    return bank, metadata


def _best_from_allowed(
    target: torch.Tensor,
    bank: torch.Tensor,
    allowed: torch.Tensor,
    *,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
    chunk_size: int,
) -> dict[str, object]:
    if int(allowed.numel()) < 1:
        raise RuntimeError("coverage comparison has no compatible candidates")
    best: dict[str, object] | None = None
    for start in range(0, int(allowed.numel()), chunk_size):
        indices = allowed[start : start + chunk_size]
        gains, cosine, normalized_mse = _score_candidates(
            target,
            bank[indices],
            vector_index=vector_index,
            filter_fft=filter_fft,
            frame_count=frame_count,
        )
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
        raise RuntimeError("coverage comparison failed to select a candidate")
    best["candidate_count"] = int(allowed.numel())
    return best


def run_full_train_coverage(
    root: Path,
    *,
    heldout_items: int = DEFAULT_HELDOUT_ITEMS,
    max_train_items: int = DEFAULT_TRAIN_ITEMS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if heldout_items < 1 or heldout_items > 3:
        raise ValueError("heldout_items must be between 1 and 3")
    if chunk_size < 1 or chunk_size > 1024:
        raise ValueError("chunk_size must be in [1, 1024]")

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

    calibration_dir = root / "models" / "lykenox_identity" / "calibration"
    sampled_bank, sampled_metadata = load_owned_residual_codebook(
        calibration_dir / "residual_codebook_v1.pt",
        calibration_dir / "residual_codebook_v1.json",
    )
    full_bank, full_metadata = _build_full_train_bank(root, max_train_items=max_train_items)

    bank_path = output_dir / "full_train_residual_retrieval_bank_v1.pt"
    temporary_bank = bank_path.with_suffix(bank_path.suffix + ".tmp")
    torch.save(full_bank, temporary_bank)
    os.replace(temporary_bank, bank_path)
    _atomic_json(output_dir / "full_train_residual_retrieval_bank_v1.json", full_metadata)

    heldout = collect_owned_vocoder_utterances(root, split="val", max_items=heldout_items)
    all_sampled_cosines: list[float] = []
    all_sampled_mse: list[float] = []
    all_full_cosines: list[float] = []
    all_full_mse: list[float] = []
    full_improvement_count = 0
    item_reports: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in heldout:
            frame_count = int(utterance.mel_frames)
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            target_residual, cepstrum, extension_frames = extract_owned_real_residual(
                reference,
                frame_count=frame_count,
            )
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("held-out full-TRAIN coverage cepstrum geometry changed")
            targets = residual_analysis_vectors(target_residual)
            if int(targets.shape[0]) != frame_count + 1:
                raise RuntimeError("held-out full-TRAIN coverage residual geometry changed")

            filters = one_sided_real_cepstrum_to_minimum_phase_fir(cepstrum, n_fft=N_FFT).to(torch.float32)
            filter_fft = torch.fft.rfft(filters, n=LOCAL_CONV_FFT_SIZE, dim=-1).contiguous()
            rows: list[dict[str, object]] = []

            for vector_index in range(frame_count + 1):
                conditioning_index = min(vector_index, frame_count - 1)
                f0_hz = float(utterance.f0_hz[conditioning_index])
                voiced = float(utterance.voiced[conditioning_index])
                periodicity = float(utterance.periodicity[conditioning_index])
                sampled_allowed = _candidate_indices(
                    sampled_metadata,
                    f0_hz=f0_hz,
                    voiced=voiced,
                    periodicity=periodicity,
                )
                full_allowed = _candidate_indices(
                    full_metadata,
                    f0_hz=f0_hz,
                    voiced=voiced,
                    periodicity=periodicity,
                )
                sampled = _best_from_allowed(
                    targets[vector_index],
                    sampled_bank,
                    sampled_allowed,
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                    chunk_size=chunk_size,
                )
                full = _best_from_allowed(
                    targets[vector_index],
                    full_bank,
                    full_allowed,
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                    chunk_size=chunk_size,
                )
                improvement = float(sampled["normalized_mse"]) - float(full["normalized_mse"])
                if improvement > 1.0e-9:
                    full_improvement_count += 1
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
                rows.append(row)
                all_sampled_cosines.append(float(sampled["cosine"]))
                all_sampled_mse.append(float(sampled["normalized_mse"]))
                all_full_cosines.append(float(full["cosine"]))
                all_full_mse.append(float(full["normalized_mse"]))

            csv_path = output_dir / f"{utterance.utterance_id}__full_train_coverage_windows.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            worst = sorted(
                rows,
                key=lambda item: (
                    float(item["full_train_best_filtered_response_cosine"]),
                    -float(item["full_train_best_normalized_mse"]),
                    int(item["vector_index"]),
                ),
            )[:20]
            item_reports.append(
                {
                    "utterance_id": utterance.utterance_id,
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
                    "csv": str(csv_path),
                }
            )

    total_windows = len(all_full_cosines)
    report: dict[str, object] = {
        "status": "ready_for_full_train_residual_retrieval_coverage_review",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "sampled_codebook_version": RESIDUAL_CODEBOOK_VERSION,
        "sampled_codebook_retained_count": int(sampled_bank.shape[0]),
        "sampled_codebook_max_per_bucket": sampled_metadata.get("max_per_bucket"),
        "sampled_codebook_original_candidate_window_count": sampled_metadata.get("candidate_window_count"),
        "full_train_retrieval_window_count": int(full_bank.shape[0]),
        "full_train_bucket_count": int(full_metadata["bucket_count"]),
        "heldout_window_count": total_windows,
        "full_train_improvement_count": full_improvement_count,
        "full_train_improvement_fraction": full_improvement_count / float(max(total_windows, 1)),
        "sampled_codebook_coverage": _coverage_summary(all_sampled_cosines, all_sampled_mse),
        "full_train_retrieval_coverage": _coverage_summary(all_full_cosines, all_full_mse),
        "full_bank_artifact": str(bank_path),
        "heldout_residual_added_to_full_bank": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "production_codebook_replaced": False,
        "metrics_can_accept_product_quality": False,
        "items": item_reports,
        "next_action": (
            "if_full_train_materially_closes_coverage_gap_then_redesign_codebook_retention;_"
            "otherwise_reject_current_cross_utterance_512_256_codevector_capacity_before_training"
        ),
    }
    _atomic_json(output_dir / "full_train_residual_retrieval_coverage_v1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_HELDOUT_ITEMS)
    parser.add_argument("--max-train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    print(
        json.dumps(
            run_full_train_coverage(
                args.root,
                heldout_items=args.heldout_items,
                max_train_items=args.max_train_items,
                chunk_size=args.chunk_size,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
