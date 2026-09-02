"""Resumable FULL-TRAIN residual oracle listening gate for LYKENOX.

This is the final audible capacity test for the current CELP-style residual-codebook line. Earlier
coverage audits established that the retained 6,234-codeword artifact is strongly under-covered and
that exhaustive retrieval across all 119,897 owned TRAIN residual vectors materially improves the
held-out synthesis-domain ceiling. Signal-aware retention at the same 24,404-vector budget did not
materially improve over hash1024, so another retention metric sweep is not useful before listening.

For one held-out VAL utterance at a time this diagnostic searches every compatible vector in the
already-built FULL-TRAIN diagnostic bank using the exact local frozen-renderer waveform response and
the same non-negative least-squares gain used by the validated synthesis-domain coverage audits. Each
completed window checkpoints the selected FULL-TRAIN index and gain. On completion those selected
TRAIN-only residual vectors are overlap-added, passed through the unchanged minimum-phase renderer,
and written as a complete WAV beside the clean identity-roundtrip ceiling and reference.

Held-out residual/cepstrum are oracle targets only. No held-out residual enters TRAIN. No model,
optimizer, training, learned checkpoint, production codebook replacement, post-hoc gain, EQ, denoise,
duration modification, third-party voice component, or remote inference is used. Metrics cannot
accept product quality; the complete held-out WAV must be judged by listening under LYX-POL-001.
CPU only.
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

from scripts.diagnostic_full_train_residual_retrieval_coverage_v2_resumable import (
    _best_full_train_against_target_response,
    _load_or_build_full_bank,
)
from scripts.diagnostic_residual_codebook_oracle_v1 import _candidate_indices, _safe_name
from scripts.diagnostic_residual_codebook_oracle_v4_synthesis_domain import (
    LOCAL_CONV_FFT_SIZE,
    _local_filtered_vector_response,
    _rms,
    _write_float_wav,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    POLICY_ID,
    residual_analysis_vectors,
    residual_synthesis_from_analysis_vectors,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    one_sided_real_cepstrum_to_minimum_phase_fir,
    render_time_varying_minimum_phase,
)


DIAGNOSTIC_VERSION = "owned-full-train-residual-oracle-listening-v1"
DEFAULT_HELDOUT_ITEMS = 3
DEFAULT_CHUNK_SIZE = 512
DEFAULT_TRAIN_ITEMS = 1_000_000
CHECKPOINT_VERSION = "owned-full-train-residual-oracle-listening-window-checkpoint-v1"

ROW_FIELDS = [
    "utterance_id",
    "vector_index",
    "conditioning_index",
    "f0_hz",
    "voiced",
    "periodicity",
    "compatible_candidate_count",
    "selected_full_train_index",
    "selected_gain",
    "best_filtered_response_cosine",
    "best_normalized_mse",
]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _load_checkpoint(path: Path, *, utterance_id: str) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise RuntimeError("full-TRAIN listening checkpoint schema mismatch")
        for raw in reader:
            if raw["utterance_id"] != utterance_id:
                raise RuntimeError("full-TRAIN listening checkpoint utterance mismatch")
            index = int(raw["vector_index"])
            rows[index] = {
                "utterance_id": raw["utterance_id"],
                "vector_index": index,
                "conditioning_index": int(raw["conditioning_index"]),
                "f0_hz": float(raw["f0_hz"]),
                "voiced": float(raw["voiced"]),
                "periodicity": float(raw["periodicity"]),
                "compatible_candidate_count": int(raw["compatible_candidate_count"]),
                "selected_full_train_index": int(raw["selected_full_train_index"]),
                "selected_gain": float(raw["selected_gain"]),
                "best_filtered_response_cosine": float(raw["best_filtered_response_cosine"]),
                "best_normalized_mse": float(raw["best_normalized_mse"]),
            }
    return rows


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


def _finalize_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_full_train_listening_oracle(
    root: Path,
    *,
    heldout_items: int = DEFAULT_HELDOUT_ITEMS,
    utterance_index: int = 2,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_train_items: int = DEFAULT_TRAIN_ITEMS,
    max_new_windows: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if heldout_items < 1 or heldout_items > 3:
        raise ValueError("heldout_items must be between 1 and 3")
    if not 0 <= utterance_index < heldout_items:
        raise ValueError("utterance_index must select one requested held-out item")
    if chunk_size < 1 or chunk_size > 2048:
        raise ValueError("chunk_size must be in [1, 2048]")
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
        / "vocoder_minimum_phase_full_train_residual_oracle_listening_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    full_bank_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_full_train_residual_retrieval_coverage_v1"
    )
    full_bank, full_metadata, reused_existing_bank = _load_or_build_full_bank(
        root,
        full_bank_dir,
        max_train_items=max_train_items,
    )
    if full_metadata.get("source_split") != "train":
        raise RuntimeError("FULL-TRAIN listening bank is not TRAIN-only")
    if full_metadata.get("heldout_data_in_bank") is not False:
        raise RuntimeError("FULL-TRAIN listening bank does not prove held-out exclusion")

    heldout = collect_owned_vocoder_utterances(root, split="val", max_items=heldout_items)
    utterance = heldout[utterance_index]
    frame_count = int(utterance.mel_frames)
    expected_windows = frame_count + 1
    expected_samples = frame_count * HOP_LENGTH
    reference = utterance.waveform.cpu().to(torch.float32).contiguous()
    target_residual, cepstrum, extension_frames = extract_owned_real_residual(
        reference,
        frame_count=frame_count,
    )
    if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
        raise RuntimeError("FULL-TRAIN listening cepstrum geometry changed")
    target_vectors = residual_analysis_vectors(target_residual)
    if int(target_vectors.shape[0]) != expected_windows:
        raise RuntimeError("FULL-TRAIN listening residual analysis geometry changed")

    filters = one_sided_real_cepstrum_to_minimum_phase_fir(
        cepstrum,
        n_fft=N_FFT,
    ).to(torch.float32)
    filter_fft = torch.fft.rfft(filters, n=LOCAL_CONV_FFT_SIZE, dim=-1).contiguous()

    stem = _safe_name(utterance.utterance_id)
    checkpoint_path = output_dir / f"{stem}__full_train_oracle_windows.partial.csv"
    completed = _load_checkpoint(checkpoint_path, utterance_id=utterance.utterance_id)
    new_windows_done = 0

    with torch.no_grad():
        for vector_index in range(expected_windows):
            if vector_index in completed:
                continue
            if max_new_windows is not None and new_windows_done >= max_new_windows:
                break

            conditioning_index = min(vector_index, frame_count - 1)
            f0_hz = float(utterance.f0_hz[conditioning_index])
            voiced = float(utterance.voiced[conditioning_index])
            periodicity = float(utterance.periodicity[conditioning_index])
            allowed = _candidate_indices(
                full_metadata,
                f0_hz=f0_hz,
                voiced=voiced,
                periodicity=periodicity,
            )
            target_response = _local_filtered_vector_response(
                target_vectors[vector_index].unsqueeze(0),
                vector_index=vector_index,
                filter_fft=filter_fft,
                frame_count=frame_count,
            ).squeeze(0)
            best = _best_full_train_against_target_response(
                target_response=target_response,
                bank=full_bank,
                allowed=allowed,
                vector_index=vector_index,
                filter_fft=filter_fft,
                frame_count=frame_count,
                chunk_size=chunk_size,
            )
            row = {
                "utterance_id": utterance.utterance_id,
                "vector_index": vector_index,
                "conditioning_index": conditioning_index,
                "f0_hz": f0_hz,
                "voiced": voiced,
                "periodicity": periodicity,
                "compatible_candidate_count": int(best["candidate_count"]),
                "selected_full_train_index": int(best["index"]),
                "selected_gain": float(best["gain"]),
                "best_filtered_response_cosine": float(best["cosine"]),
                "best_normalized_mse": float(best["normalized_mse"]),
            }
            _append_checkpoint(checkpoint_path, row)
            completed[vector_index] = row
            new_windows_done += 1
            print(
                json.dumps(
                    {
                        "status": "checkpointed_full_train_listening_window",
                        "utterance_index": utterance_index,
                        "utterance_id": utterance.utterance_id,
                        "vector_index": vector_index,
                        "completed_windows": len(completed),
                        "expected_windows": expected_windows,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if len(completed) != expected_windows:
        progress: dict[str, object] = {
            "status": "full_train_residual_oracle_listening_in_progress",
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "policy_id": POLICY_ID,
            "device": "cpu",
            "renderer_version": RENDERER_VERSION,
            "reused_existing_full_train_bank": reused_existing_bank,
            "full_train_retrieval_window_count": int(full_bank.shape[0]),
            "utterance_index": utterance_index,
            "utterance_id": utterance.utterance_id,
            "completed_windows": len(completed),
            "expected_windows": expected_windows,
            "new_windows_completed_this_run": new_windows_done,
            "checkpoint": str(checkpoint_path),
            "training_executed": False,
            "optimizer_created": False,
            "checkpoint_written": False,
            "production_codebook_replaced": False,
        }
        _atomic_json(output_dir / "full_train_residual_oracle_listening_v1_progress.json", progress)
        return progress

    ordered = [completed[index] for index in range(expected_windows)]
    final_csv = output_dir / f"{stem}__full_train_oracle_windows.csv"
    _finalize_csv(final_csv, ordered)

    selected_vectors = torch.stack(
        [
            full_bank[int(row["selected_full_train_index"])] * float(row["selected_gain"])
            for row in ordered
        ],
        dim=0,
    ).to(torch.float32).contiguous()
    selected_residual = residual_synthesis_from_analysis_vectors(
        selected_vectors,
        output_samples=expected_samples,
    )
    prediction = render_time_varying_minimum_phase(
        selected_residual.unsqueeze(0),
        cepstrum.unsqueeze(0),
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    ).squeeze(0)
    identity_roundtrip = render_time_varying_minimum_phase(
        target_residual.unsqueeze(0),
        cepstrum.unsqueeze(0),
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    ).squeeze(0)
    if prediction.shape != reference.shape or selected_residual.shape != reference.shape:
        raise RuntimeError("FULL-TRAIN listening output length mismatch")
    if not bool(
        torch.isfinite(prediction).all()
        and torch.isfinite(selected_residual).all()
        and torch.isfinite(identity_roundtrip).all()
    ):
        raise RuntimeError("FULL-TRAIN listening produced non-finite audio")

    prediction_path = output_dir / f"{stem}__full_train_residual_oracle.wav"
    residual_path = output_dir / f"{stem}__selected_full_train_residual.wav"
    ceiling_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
    reference_path = output_dir / f"{stem}__reference.wav"
    _write_float_wav(prediction_path, prediction)
    _write_float_wav(residual_path, selected_residual)
    _write_float_wav(ceiling_path, identity_roundtrip)
    _write_float_wav(reference_path, reference)

    cosines = [float(row["best_filtered_response_cosine"]) for row in ordered]
    nmse = [float(row["best_normalized_mse"]) for row in ordered]
    gains = [float(row["selected_gain"]) for row in ordered]
    report: dict[str, object] = {
        "status": "ready_for_full_train_residual_oracle_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "full_train_bank_source_split": full_metadata.get("source_split"),
        "full_train_retrieval_window_count": int(full_bank.shape[0]),
        "heldout_split": "val",
        "utterance_index": utterance_index,
        "utterance_id": utterance.utterance_id,
        "seconds": expected_samples / float(SAMPLE_RATE),
        "oracle_search_windows": expected_windows,
        "selection_domain": "exact_local_frozen_renderer_waveform_contribution",
        "gain_rule": "non_negative_least_squares_in_exact_local_renderer_response_domain",
        "mean_filtered_response_cosine": sum(cosines) / float(len(cosines)),
        "minimum_filtered_response_cosine": min(cosines),
        "mean_normalized_mse": sum(nmse) / float(len(nmse)),
        "mean_oracle_gain": sum(gains) / float(len(gains)),
        "unique_full_train_codewords_used": len(
            {int(row["selected_full_train_index"]) for row in ordered}
        ),
        "target_residual_rms": _rms(target_residual),
        "selected_full_train_residual_rms": _rms(selected_residual),
        "prediction_rms": _rms(prediction),
        "reference_rms": _rms(reference),
        "full_train_residual_oracle": str(prediction_path),
        "selected_full_train_residual": str(residual_path),
        "identity_roundtrip_ceiling": str(ceiling_path),
        "reference": str(reference_path),
        "heldout_residual_used_only_as_oracle_target": True,
        "heldout_residual_added_to_train": False,
        "oracle_indices_or_gains_valid_for_product_inference": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "production_codebook_replaced": False,
        "production_renderer_modified": False,
        "posthoc_output_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "next_action": (
            "listen_to_full_train_oracle_vs_identity_roundtrip_ceiling;_"
            "if_still_gangoso_close_current_celp_codebook_line;_"
            "if_materially_better_keep_full_train_coverage_as_required_capacity_evidence"
        ),
    }
    _atomic_json(output_dir / "full_train_residual_oracle_listening_v1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_HELDOUT_ITEMS)
    parser.add_argument("--utterance-index", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--max-new-windows", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_full_train_listening_oracle(
                args.root,
                heldout_items=args.heldout_items,
                utterance_index=args.utterance_index,
                chunk_size=args.chunk_size,
                max_train_items=args.max_train_items,
                max_new_windows=args.max_new_windows,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
