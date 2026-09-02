"""Held-out synthesis-domain coverage audit for the owned LYKENOX residual codebook.

V4 and V5 evaluate candidates in the frozen renderer waveform domain, but both first reduce the
compatible owned-train candidate set to PRESELECT_K items by residual cosine. V5 additionally adds
filtered-domain temporal continuity and was still reported perceptually gangoso. Before changing any
selection algorithm or authorizing training, this diagnostic measures whether the existing codebook
actually contains adequate local excitation coverage.

For every held-out analysis window it reports two ceilings using the same exact V4 local renderer
response and the same non-negative least-squares gain in the filtered waveform domain:

1. ``preselected``: best candidate among V4's residual-cosine PRESELECT_K set.
2. ``all_compatible``: best candidate among every compatible owned-train codeword returned by the
   existing voicing/F0/periodicity bucket rule, evaluated in deterministic CPU chunks.

This distinguishes codebook coverage failure from preselection failure. Held-out residual/cepstrum
are diagnostic oracle targets only and are never added to the codebook. Metrics can reject but cannot
accept product quality under LYX-POL-001. No audio modification, model, optimizer, training,
checkpoint, production change, third-party voice component, or remote service is used.
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

from scripts.diagnostic_residual_codebook_oracle_v1 import _candidate_indices
from scripts.diagnostic_residual_codebook_oracle_v4_synthesis_domain import (
    ENERGY_EPSILON,
    LOCAL_CONV_FFT_SIZE,
    PRESELECT_K,
    _local_filtered_vector_response,
    _preselect_residual_candidates,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    POLICY_ID,
    RESIDUAL_CODEBOOK_VERSION,
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


DIAGNOSTIC_VERSION = "owned-residual-codebook-synthesis-coverage-v1"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
DEFAULT_CHUNK_SIZE = 128
COSINE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - float(lower)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _coverage_summary(cosines: list[float], normalized_mse: list[float]) -> dict[str, object]:
    if not cosines or len(cosines) != len(normalized_mse):
        raise ValueError("coverage summary arrays must be non-empty and aligned")
    result: dict[str, object] = {
        "window_count": len(cosines),
        "cosine_min": min(cosines),
        "cosine_p05": _percentile(cosines, 0.05),
        "cosine_p10": _percentile(cosines, 0.10),
        "cosine_p25": _percentile(cosines, 0.25),
        "cosine_median": _percentile(cosines, 0.50),
        "cosine_p75": _percentile(cosines, 0.75),
        "cosine_p90": _percentile(cosines, 0.90),
        "cosine_p95": _percentile(cosines, 0.95),
        "cosine_mean": sum(cosines) / float(len(cosines)),
        "normalized_mse_p50": _percentile(normalized_mse, 0.50),
        "normalized_mse_p90": _percentile(normalized_mse, 0.90),
        "normalized_mse_p95": _percentile(normalized_mse, 0.95),
        "normalized_mse_max": max(normalized_mse),
    }
    for threshold in COSINE_THRESHOLDS:
        count = sum(1 for value in cosines if value < threshold)
        key = str(int(round(threshold * 100.0)))
        result[f"windows_below_cosine_{key}"] = count
        result[f"fraction_below_cosine_{key}"] = count / float(len(cosines))
    return result


def _score_candidates(
    target: torch.Tensor,
    candidates: torch.Tensor,
    *,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return non-negative LS gains, filtered-response cosine, and normalized MSE."""

    if candidates.ndim != 2:
        raise ValueError("candidates must have shape [K, samples]")
    all_vectors = torch.cat((target.unsqueeze(0).to(torch.float32), candidates.to(torch.float32)), dim=0)
    responses = _local_filtered_vector_response(
        all_vectors,
        vector_index=vector_index,
        filter_fft=filter_fft,
        frame_count=frame_count,
    )
    target_response = responses[0]
    candidate_responses = responses[1:]
    target_energy = target_response.square().sum().clamp_min(ENERGY_EPSILON)
    candidate_energy = candidate_responses.square().sum(dim=1).clamp_min(ENERGY_EPSILON)
    dots = torch.mv(candidate_responses, target_response)
    gains = (dots / candidate_energy).clamp_min(0.0)
    scaled = candidate_responses * gains.unsqueeze(1)
    error_energy = (scaled - target_response.unsqueeze(0)).square().sum(dim=1)
    normalized_mse = error_energy / target_energy
    cosine = dots / torch.sqrt(candidate_energy * target_energy)
    return gains.contiguous(), cosine.contiguous(), normalized_mse.contiguous()


def _best_preselected(
    target: torch.Tensor,
    codewords: torch.Tensor,
    allowed: torch.Tensor,
    *,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
) -> dict[str, object]:
    indices, candidates, _ = _preselect_residual_candidates(target, codewords, allowed)
    gains, cosine, normalized_mse = _score_candidates(
        target,
        candidates,
        vector_index=vector_index,
        filter_fft=filter_fft,
        frame_count=frame_count,
    )
    best = int(torch.argmin(normalized_mse))
    return {
        "index": int(indices[best]),
        "gain": float(gains[best]),
        "cosine": float(cosine[best]),
        "normalized_mse": float(normalized_mse[best]),
        "candidate_count": int(indices.numel()),
    }


def _best_all_compatible(
    target: torch.Tensor,
    codewords: torch.Tensor,
    allowed: torch.Tensor,
    *,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
    chunk_size: int,
) -> dict[str, object]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    best: dict[str, object] | None = None
    for start in range(0, int(allowed.numel()), chunk_size):
        indices = allowed[start : start + chunk_size]
        candidates = codewords[indices]
        gains, cosine, normalized_mse = _score_candidates(
            target,
            candidates,
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
        raise RuntimeError("no compatible candidates available for exhaustive coverage audit")
    best["candidate_count"] = int(allowed.numel())
    return best


def run_synthesis_coverage_audit(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    tensor_path: Path | None = None,
    index_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("held-out coverage audit must not run on train")
    if max_items < 1 or max_items > 3:
        raise ValueError("max_items must be between 1 and 3")
    if chunk_size < 1 or chunk_size > 1024:
        raise ValueError("chunk_size must be in [1, 1024]")

    root = Path(root).resolve()
    calibration_dir = root / "models" / "lykenox_identity" / "calibration"
    tensor_path = Path(tensor_path).resolve() if tensor_path is not None else calibration_dir / "residual_codebook_v1.pt"
    index_path = Path(index_path).resolve() if index_path is not None else calibration_dir / "residual_codebook_v1.json"
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "evaluation" / "vocoder_minimum_phase_residual_codebook_synthesis_coverage_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    codewords, metadata = load_owned_residual_codebook(tensor_path, index_path)
    if metadata.get("source_split") != "train":
        raise RuntimeError("coverage audit codebook is contaminated by non-train identity data")

    utterances = collect_owned_vocoder_utterances(root, split=split, max_items=max_items)
    item_reports: list[dict[str, object]] = []
    all_pre_cosines: list[float] = []
    all_pre_mse: list[float] = []
    all_full_cosines: list[float] = []
    all_full_mse: list[float] = []
    total_preselector_misses = 0

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            target_residual, cepstrum, extension_frames = extract_owned_real_residual(reference, frame_count=frame_count)
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("held-out coverage cepstrum geometry changed")
            target_vectors = residual_analysis_vectors(target_residual)
            if int(target_vectors.shape[0]) != frame_count + 1:
                raise RuntimeError("held-out coverage residual analysis geometry changed")

            filters = one_sided_real_cepstrum_to_minimum_phase_fir(cepstrum, n_fft=N_FFT).to(torch.float32)
            filter_fft = torch.fft.rfft(filters, n=LOCAL_CONV_FFT_SIZE, dim=-1).contiguous()
            windows: list[dict[str, object]] = []

            for vector_index in range(frame_count + 1):
                conditioning_index = min(vector_index, frame_count - 1)
                f0_hz = float(utterance.f0_hz[conditioning_index])
                voiced = float(utterance.voiced[conditioning_index])
                periodicity = float(utterance.periodicity[conditioning_index])
                allowed = _candidate_indices(
                    metadata,
                    f0_hz=f0_hz,
                    voiced=voiced,
                    periodicity=periodicity,
                )
                pre = _best_preselected(
                    target_vectors[vector_index],
                    codewords,
                    allowed,
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                )
                full = _best_all_compatible(
                    target_vectors[vector_index],
                    codewords,
                    allowed,
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                    chunk_size=chunk_size,
                )
                preselector_missed_best = int(pre["index"]) != int(full["index"])
                if preselector_missed_best:
                    total_preselector_misses += 1
                row = {
                    "utterance_id": utterance.utterance_id,
                    "vector_index": vector_index,
                    "conditioning_index": conditioning_index,
                    "f0_hz": f0_hz,
                    "voiced": voiced,
                    "periodicity": periodicity,
                    "compatible_candidate_count": int(allowed.numel()),
                    "preselected_candidate_count": int(pre["candidate_count"]),
                    "preselected_best_index": int(pre["index"]),
                    "preselected_best_gain": float(pre["gain"]),
                    "preselected_best_filtered_response_cosine": float(pre["cosine"]),
                    "preselected_best_normalized_mse": float(pre["normalized_mse"]),
                    "all_compatible_best_index": int(full["index"]),
                    "all_compatible_best_gain": float(full["gain"]),
                    "all_compatible_best_filtered_response_cosine": float(full["cosine"]),
                    "all_compatible_best_normalized_mse": float(full["normalized_mse"]),
                    "preselector_missed_all_compatible_best": preselector_missed_best,
                    "normalized_mse_improvement_all_vs_preselected": float(pre["normalized_mse"]) - float(full["normalized_mse"]),
                }
                windows.append(row)
                all_pre_cosines.append(float(pre["cosine"]))
                all_pre_mse.append(float(pre["normalized_mse"]))
                all_full_cosines.append(float(full["cosine"]))
                all_full_mse.append(float(full["normalized_mse"]))

            worst = sorted(
                windows,
                key=lambda item: (
                    float(item["all_compatible_best_filtered_response_cosine"]),
                    -float(item["all_compatible_best_normalized_mse"]),
                    int(item["vector_index"]),
                ),
            )[:20]
            item_reports.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "terminal_transfer_extension_frames": extension_frames,
                    "window_count": len(windows),
                    "preselected_coverage": _coverage_summary(
                        [float(item["preselected_best_filtered_response_cosine"]) for item in windows],
                        [float(item["preselected_best_normalized_mse"]) for item in windows],
                    ),
                    "all_compatible_coverage": _coverage_summary(
                        [float(item["all_compatible_best_filtered_response_cosine"]) for item in windows],
                        [float(item["all_compatible_best_normalized_mse"]) for item in windows],
                    ),
                    "preselector_miss_count": sum(1 for item in windows if bool(item["preselector_missed_all_compatible_best"])),
                    "worst_20_all_compatible_windows": worst,
                    "windows": windows,
                }
            )

            csv_path = output_dir / f"{utterance.utterance_id}__synthesis_coverage_windows.csv"
            if windows:
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(windows[0].keys()))
                    writer.writeheader()
                    writer.writerows(windows)

    total_windows = len(all_full_cosines)
    report: dict[str, object] = {
        "status": "ready_for_residual_codebook_synthesis_coverage_review",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "codebook_version": RESIDUAL_CODEBOOK_VERSION,
        "codebook_source_split": metadata.get("source_split"),
        "codebook_tensor_sha256": metadata.get("tensor_sha256"),
        "retained_codeword_count": int(codewords.shape[0]),
        "heldout_split": split,
        "heldout_residual_used_only_as_oracle_measurement_target": True,
        "heldout_residual_added_to_codebook": False,
        "v4_preselect_k": PRESELECT_K,
        "all_compatible_evaluated": True,
        "all_compatible_chunk_size": chunk_size,
        "selection_domain": "exact_local_frozen_renderer_waveform_contribution",
        "gain_rule": "non_negative_least_squares_in_exact_local_renderer_response_domain",
        "metrics_can_accept_product_quality": False,
        "metrics_can_reject_representation_or_preselector": True,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "production_renderer_modified": False,
        "third_party_voice_component_used": False,
        "remote_inference_used": False,
        "total_window_count": total_windows,
        "preselected_coverage": _coverage_summary(all_pre_cosines, all_pre_mse),
        "all_compatible_coverage": _coverage_summary(all_full_cosines, all_full_mse),
        "preselector_miss_count": total_preselector_misses,
        "preselector_miss_fraction": total_preselector_misses / float(max(total_windows, 1)),
        "items": item_reports,
        "next_action": "review_all_compatible_filtered_response_coverage_before_any_new_selector_or_training",
    }
    _atomic_json(output_dir / "residual_codebook_synthesis_coverage_v1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_ITEMS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    result = run_synthesis_coverage_audit(
        args.root,
        split="val",
        max_items=args.heldout_items,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
