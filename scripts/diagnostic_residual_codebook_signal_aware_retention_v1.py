"""TRAIN-only signal-aware residual-codebook retention audit.

The hash-cap sweep showed that raising the existing deterministic hash retention from 128 to 1024
codewords per conditioning bucket improves held-out synthesis-domain coverage, but cap1024 recovers
only part of the gap to exhaustive full-TRAIN retrieval.  This diagnostic therefore changes *which*
TRAIN residual vectors are retained while keeping the same maximum 1024-per-bucket budget.

Retention is selected exclusively from owned TRAIN residual vectors.  Within each existing
voicing/F0/periodicity conditioning bucket, every 512-sample residual codevector receives a fixed,
non-learned DSP descriptor:

* zero-crossing rate;
* normalized spectral centroid;
* upper-half spectral-energy fraction;
* lag-1 normalized autocorrelation;
* temporal energy centroid;
* signed half-window balance (phase/polarity-sensitive).

Each descriptor dimension is split into three deterministic within-bucket rank quantiles.  The six
ternary coordinates define up to 729 signal-shape strata.  Selection proceeds round-robin across
occupied strata so every represented signal region receives one vector before any stratum receives a
second; a deterministic source-index hash is used only to order ties/extra members within a stratum.
No held-out target participates in retention.

The resulting diagnostic bank is compared at the same 1024-per-bucket budget against the completed
hash1024 baseline and the full-TRAIN ceiling using the same compatibility rule, exact local frozen
minimum-phase renderer response, and non-negative least-squares synthesis-domain gain.  The audit is
resumable per held-out window.  It does not train a model, create an optimizer/checkpoint, replace the
production codebook, modify the renderer, or admit held-out residuals into TRAIN.  Metrics can reject
but cannot accept product quality under LYX-POL-001.  CPU only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from scripts.diagnostic_residual_codebook_oracle_v1 import _candidate_indices
from scripts.diagnostic_residual_codebook_oracle_v4_synthesis_domain import (
    LOCAL_CONV_FFT_SIZE,
    _local_filtered_vector_response,
)
from scripts.diagnostic_residual_codebook_synthesis_coverage_v1 import _coverage_summary
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    POLICY_ID,
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


DIAGNOSTIC_VERSION = "owned-residual-codebook-signal-aware-retention-v1"
SIGNAL_BANK_VERSION = "owned-residual-codebook-signal-aware-retention-bank-v1"
RETENTION_CAP = 1024
DESCRIPTOR_QUANTILE_BINS = 3
DESCRIPTOR_FEATURES = (
    "zero_crossing_rate",
    "normalized_spectral_centroid",
    "upper_half_spectral_energy_fraction",
    "lag1_normalized_autocorrelation",
    "temporal_energy_centroid",
    "signed_half_window_balance",
)
DEFAULT_HELDOUT_ITEMS = 3
DEFAULT_TRAIN_ITEMS = 1_000_000
DEFAULT_CHUNK_SIZE = 512
CHECKPOINT_VERSION = "owned-signal-aware-retention-window-checkpoint-v1"

ROW_FIELDS = [
    "utterance_id",
    "vector_index",
    "conditioning_index",
    "f0_hz",
    "voiced",
    "periodicity",
    "hash1024_compatible_candidate_count",
    "signal1024_compatible_candidate_count",
    "hash1024_best_filtered_response_cosine",
    "hash1024_best_normalized_mse",
    "signal1024_best_filtered_response_cosine",
    "signal1024_best_normalized_mse",
    "full_train_best_filtered_response_cosine",
    "full_train_best_normalized_mse",
    "normalized_mse_improvement_signal_vs_hash1024",
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
        raise RuntimeError("signal-aware retention bank tensor has invalid geometry")
    value = value.detach().cpu().to(torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("signal-aware retention bank contains non-finite values")
    return value


def _source_tie_score(source_index: int) -> int:
    payload = f"{SIGNAL_BANK_VERSION}|source_index={int(source_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def _descriptor_features(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2 or int(vectors.shape[1]) < 4:
        raise ValueError("descriptor vectors must have shape [N, samples]")
    vectors = vectors.detach().cpu().to(torch.float32).contiguous()
    count, samples = int(vectors.shape[0]), int(vectors.shape[1])
    if count < 1:
        raise ValueError("descriptor requires at least one vector")

    energy = vectors.square().sum(dim=1).clamp_min(1.0e-12)
    signs = vectors >= 0.0
    zero_crossing = (signs[:, 1:] != signs[:, :-1]).to(torch.float32).mean(dim=1)

    spectrum = torch.fft.rfft(vectors, dim=1)
    power = spectrum.real.square() + spectrum.imag.square()
    spectral_energy = power.sum(dim=1).clamp_min(1.0e-12)
    frequencies = torch.linspace(0.0, 1.0, int(power.shape[1]), dtype=torch.float32)
    centroid = (power * frequencies.unsqueeze(0)).sum(dim=1) / spectral_energy
    high_start = max(1, int(power.shape[1]) // 2)
    high_fraction = power[:, high_start:].sum(dim=1) / spectral_energy

    left = vectors[:, :-1]
    right = vectors[:, 1:]
    lag1_denominator = torch.sqrt(
        left.square().sum(dim=1).clamp_min(1.0e-12)
        * right.square().sum(dim=1).clamp_min(1.0e-12)
    )
    lag1 = (left * right).sum(dim=1) / lag1_denominator

    time = torch.linspace(0.0, 1.0, samples, dtype=torch.float32)
    temporal_centroid = (vectors.square() * time.unsqueeze(0)).sum(dim=1) / energy

    half = samples // 2
    signed_balance = (
        vectors[:, half:].sum(dim=1) - vectors[:, :half].sum(dim=1)
    ) / torch.sqrt(energy * float(samples))

    features = torch.stack(
        (
            zero_crossing,
            centroid,
            high_fraction,
            lag1,
            temporal_centroid,
            signed_balance,
        ),
        dim=1,
    ).to(torch.float32).contiguous()
    if features.shape != (count, len(DESCRIPTOR_FEATURES)):
        raise RuntimeError("signal descriptor geometry changed")
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("signal descriptor contains non-finite values")
    return features


def _rank_quantile_cells(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or int(features.shape[1]) != len(DESCRIPTOR_FEATURES):
        raise ValueError("features have invalid signal-descriptor geometry")
    count = int(features.shape[0])
    if count < 1:
        raise ValueError("features must be non-empty")
    bins = torch.zeros(count, len(DESCRIPTOR_FEATURES), dtype=torch.long)
    for feature_index in range(len(DESCRIPTOR_FEATURES)):
        values = features[:, feature_index]
        order = sorted(range(count), key=lambda index: (float(values[index]), index))
        for rank, local_index in enumerate(order):
            quantile = min(
                DESCRIPTOR_QUANTILE_BINS - 1,
                (rank * DESCRIPTOR_QUANTILE_BINS) // count,
            )
            bins[local_index, feature_index] = int(quantile)

    cell = torch.zeros(count, dtype=torch.long)
    for feature_index in range(len(DESCRIPTOR_FEATURES)):
        cell = cell * DESCRIPTOR_QUANTILE_BINS + bins[:, feature_index]
    return cell.contiguous()


def _select_signal_aware_indices(
    vectors: torch.Tensor,
    *,
    source_start_index: int,
    cap: int = RETENTION_CAP,
) -> torch.Tensor:
    if cap < 1:
        raise ValueError("retention cap must be positive")
    count = int(vectors.shape[0])
    if count <= cap:
        return torch.arange(count, dtype=torch.long)

    cells = _rank_quantile_cells(_descriptor_features(vectors))
    groups: dict[int, list[int]] = {}
    for local_index in range(count):
        groups.setdefault(int(cells[local_index]), []).append(local_index)
    for cell_id, members in groups.items():
        members.sort(
            key=lambda local_index: (
                _source_tie_score(source_start_index + local_index),
                source_start_index + local_index,
            )
        )
        groups[cell_id] = members

    selected: list[int] = []
    ordered_cells = sorted(groups)
    depth = 0
    while len(selected) < cap:
        added = False
        for cell_id in ordered_cells:
            members = groups[cell_id]
            if depth >= len(members):
                continue
            selected.append(members[depth])
            added = True
            if len(selected) >= cap:
                break
        if not added:
            break
        depth += 1

    if len(selected) != cap:
        raise RuntimeError("signal-aware round-robin retention did not fill requested cap")
    return torch.tensor(selected, dtype=torch.long)


def _build_signal_aware_bank(
    full_bank: torch.Tensor,
    full_metadata: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    buckets = full_metadata.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise RuntimeError("full TRAIN metadata has no buckets")

    retained_chunks: list[torch.Tensor] = []
    retained_buckets: list[dict[str, object]] = []
    offset = 0
    original_total = 0
    for raw in buckets:
        item = dict(raw)
        source_start = int(item["start_index"])
        source_count = int(item["count"])
        if source_count < 1:
            continue
        source_end = source_start + source_count
        source_vectors = full_bank[source_start:source_end]
        if int(source_vectors.shape[0]) != source_count:
            raise RuntimeError("full TRAIN bucket slice geometry mismatch")
        selected_local = _select_signal_aware_indices(
            source_vectors,
            source_start_index=source_start,
            cap=min(RETENTION_CAP, source_count),
        )
        selected = source_vectors[selected_local].to(torch.float32).contiguous()
        retained_count = int(selected.shape[0])
        retained_chunks.append(selected)
        retained_buckets.append(
            {
                "key": item.get("key"),
                "voicing_state": item["voicing_state"],
                "f0_bin_hz": int(item["f0_bin_hz"]),
                "periodicity_bin": int(item["periodicity_bin"]),
                "start_index": offset,
                "count": retained_count,
                "source_full_train_start_index": source_start,
                "source_full_train_count": source_count,
                "occupied_descriptor_strata": int(
                    torch.unique(_rank_quantile_cells(_descriptor_features(source_vectors))).numel()
                ),
            }
        )
        offset += retained_count
        original_total += source_count

    if not retained_chunks:
        raise RuntimeError("signal-aware retention produced no vectors")
    bank = torch.cat(retained_chunks, dim=0).to(torch.float32).contiguous()
    if not bool(torch.isfinite(bank).all()):
        raise RuntimeError("signal-aware retained bank contains non-finite values")

    metadata: dict[str, Any] = {
        "status": "built_diagnostic_train_only_signal_aware_retention_bank",
        "bank_version": SIGNAL_BANK_VERSION,
        "policy_id": POLICY_ID,
        "source_split": "train",
        "artifact_role": "diagnostic_signal_aware_retention_not_production_codebook",
        "renderer_version": RENDERER_VERSION,
        "retention_cap_per_bucket": RETENTION_CAP,
        "descriptor_features": list(DESCRIPTOR_FEATURES),
        "descriptor_quantile_bins": DESCRIPTOR_QUANTILE_BINS,
        "maximum_descriptor_strata": DESCRIPTOR_QUANTILE_BINS ** len(DESCRIPTOR_FEATURES),
        "selection_rule": "rank_quantile_signal_strata_round_robin_then_source_index_hash_tiebreak",
        "full_train_source_window_count": original_total,
        "retained_window_count": int(bank.shape[0]),
        "bucket_count": len(retained_buckets),
        "buckets": retained_buckets,
        "heldout_used_to_select_codewords": False,
        "heldout_data_in_bank": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "production_codebook_replaced": False,
        "third_party_voice_data_used": False,
    }
    return bank, metadata


def _load_or_build_signal_bank(
    root: Path,
    output_dir: Path,
    *,
    max_train_items: int,
) -> tuple[torch.Tensor, dict[str, Any], bool]:
    tensor_path = output_dir / "residual_codebook_signal_aware_retention_1024_v1.pt"
    metadata_path = output_dir / "residual_codebook_signal_aware_retention_1024_v1.json"
    if tensor_path.exists() and metadata_path.exists():
        metadata = _load_json(metadata_path)
        if metadata.get("bank_version") != SIGNAL_BANK_VERSION:
            raise RuntimeError("existing signal-aware bank version mismatch")
        if metadata.get("policy_id") != POLICY_ID or metadata.get("source_split") != "train":
            raise RuntimeError("existing signal-aware bank provenance mismatch")
        if metadata.get("heldout_used_to_select_codewords") is not False:
            raise RuntimeError("existing signal-aware bank does not prove TRAIN-only selection")
        bank = _load_tensor(tensor_path)
        if int(metadata.get("retained_window_count", -1)) != int(bank.shape[0]):
            raise RuntimeError("existing signal-aware bank count mismatch")
        return bank, metadata, True

    full_output = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_full_train_residual_retrieval_coverage_v1"
    )
    full_bank, full_metadata, _ = _load_or_build_full_bank(
        root,
        full_output,
        max_train_items=max_train_items,
    )
    bank, metadata = _build_signal_aware_bank(full_bank, full_metadata)

    sweep_report_path = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_residual_codebook_hash_retention_sweep_v1"
        / "residual_codebook_hash_retention_sweep_v1_report.json"
    )
    if sweep_report_path.exists():
        sweep_report = _load_json(sweep_report_path)
        expected = int(sweep_report.get("nested_bank_retained_count", -1))
        if expected > 0 and expected != int(bank.shape[0]):
            raise RuntimeError(
                "signal-aware equal-budget comparison failed: retained count differs from hash1024"
            )
        metadata["equal_budget_hash1024_retained_count"] = expected
        metadata["equal_budget_match"] = expected == int(bank.shape[0])

    temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    torch.save(bank, temporary)
    os.replace(temporary, tensor_path)
    _atomic_json(metadata_path, metadata)
    return bank, metadata, False


def _load_hash_sweep_baseline(root: Path, utterance_id: str) -> dict[int, dict[str, float | int]]:
    path = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_residual_codebook_hash_retention_sweep_v1"
        / f"{utterance_id}__retention_sweep_windows.csv"
    )
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: dict[int, dict[str, float | int]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows[int(raw["vector_index"])] = {
                "hash_candidate_count": int(raw["cap1024_compatible_candidate_count"]),
                "hash_cosine": float(raw["cap1024_best_filtered_response_cosine"]),
                "hash_mse": float(raw["cap1024_best_normalized_mse"]),
                "full_cosine": float(raw["full_train_best_filtered_response_cosine"]),
                "full_mse": float(raw["full_train_best_normalized_mse"]),
            }
    if not rows:
        raise RuntimeError("hash1024 baseline CSV contains no rows")
    return rows


def _load_checkpoint(path: Path, *, utterance_id: str) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    result: dict[int, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise RuntimeError("signal-aware checkpoint schema mismatch")
        for raw in reader:
            if raw["utterance_id"] != utterance_id:
                raise RuntimeError("signal-aware checkpoint utterance mismatch")
            row: dict[str, object] = {
                "utterance_id": raw["utterance_id"],
                "vector_index": int(raw["vector_index"]),
                "conditioning_index": int(raw["conditioning_index"]),
                "f0_hz": float(raw["f0_hz"]),
                "voiced": float(raw["voiced"]),
                "periodicity": float(raw["periodicity"]),
                "hash1024_compatible_candidate_count": int(raw["hash1024_compatible_candidate_count"]),
                "signal1024_compatible_candidate_count": int(raw["signal1024_compatible_candidate_count"]),
                "hash1024_best_filtered_response_cosine": float(raw["hash1024_best_filtered_response_cosine"]),
                "hash1024_best_normalized_mse": float(raw["hash1024_best_normalized_mse"]),
                "signal1024_best_filtered_response_cosine": float(raw["signal1024_best_filtered_response_cosine"]),
                "signal1024_best_normalized_mse": float(raw["signal1024_best_normalized_mse"]),
                "full_train_best_filtered_response_cosine": float(raw["full_train_best_filtered_response_cosine"]),
                "full_train_best_normalized_mse": float(raw["full_train_best_normalized_mse"]),
                "normalized_mse_improvement_signal_vs_hash1024": float(raw["normalized_mse_improvement_signal_vs_hash1024"]),
            }
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


def _write_final_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _coverage(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    return _coverage_summary(
        [float(row[f"{prefix}_best_filtered_response_cosine"]) for row in rows],
        [float(row[f"{prefix}_best_normalized_mse"]) for row in rows],
    )


def _equal_budget_comparison(rows: list[dict[str, object]]) -> dict[str, float]:
    hash_total = sum(float(row["hash1024_best_normalized_mse"]) for row in rows)
    signal_total = sum(float(row["signal1024_best_normalized_mse"]) for row in rows)
    full_total = sum(float(row["full_train_best_normalized_mse"]) for row in rows)
    denominator = hash_total - full_total
    recovered = hash_total - signal_total
    return {
        "total_nmse_hash1024": hash_total,
        "total_nmse_signal1024": signal_total,
        "total_nmse_full_train": full_total,
        "fraction_of_hash1024_to_full_train_nmse_gap_recovered": (
            recovered / denominator if denominator > 1.0e-12 else 0.0
        ),
        "fraction_windows_improved_vs_hash1024": sum(
            1
            for row in rows
            if float(row["signal1024_best_normalized_mse"])
            < float(row["hash1024_best_normalized_mse"]) - 1.0e-9
        )
        / float(len(rows)),
        "fraction_windows_worse_than_hash1024": sum(
            1
            for row in rows
            if float(row["signal1024_best_normalized_mse"])
            > float(row["hash1024_best_normalized_mse"]) + 1.0e-9
        )
        / float(len(rows)),
    }


def run_signal_aware_retention_audit(
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
        / "vocoder_minimum_phase_residual_codebook_signal_aware_retention_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bank, metadata, reused_bank = _load_or_build_signal_bank(
        root,
        output_dir,
        max_train_items=max_train_items,
    )
    if metadata.get("heldout_used_to_select_codewords") is not False:
        raise RuntimeError("signal-aware retention is not proven TRAIN-only")

    heldout = collect_owned_vocoder_utterances(root, split="val", max_items=heldout_items)
    selected_items = [utterance_index] if utterance_index is not None else list(range(len(heldout)))
    new_windows_done = 0
    incomplete: list[dict[str, object]] = []

    with torch.no_grad():
        for item_index in selected_items:
            utterance = heldout[item_index]
            frame_count = int(utterance.mel_frames)
            expected_windows = frame_count + 1
            baseline = _load_hash_sweep_baseline(root, utterance.utterance_id)
            if len(baseline) != expected_windows:
                raise RuntimeError("signal-aware baseline geometry mismatch")

            checkpoint = output_dir / f"{utterance.utterance_id}__signal_aware_retention_windows.partial.csv"
            completed = _load_checkpoint(checkpoint, utterance_id=utterance.utterance_id)

            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            target_residual, cepstrum, _ = extract_owned_real_residual(reference, frame_count=frame_count)
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("signal-aware held-out cepstrum geometry changed")
            targets = residual_analysis_vectors(target_residual)
            if int(targets.shape[0]) != expected_windows:
                raise RuntimeError("signal-aware held-out residual geometry changed")

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
                allowed = _candidate_indices(
                    metadata,
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
                signal = _best_full_train_against_target_response(
                    target_response=target_response,
                    bank=bank,
                    allowed=allowed,
                    vector_index=vector_index,
                    filter_fft=filter_fft,
                    frame_count=frame_count,
                    chunk_size=chunk_size,
                )
                base = baseline[vector_index]
                row = {
                    "utterance_id": utterance.utterance_id,
                    "vector_index": vector_index,
                    "conditioning_index": conditioning_index,
                    "f0_hz": f0_hz,
                    "voiced": voiced,
                    "periodicity": periodicity,
                    "hash1024_compatible_candidate_count": int(base["hash_candidate_count"]),
                    "signal1024_compatible_candidate_count": int(signal["candidate_count"]),
                    "hash1024_best_filtered_response_cosine": float(base["hash_cosine"]),
                    "hash1024_best_normalized_mse": float(base["hash_mse"]),
                    "signal1024_best_filtered_response_cosine": float(signal["cosine"]),
                    "signal1024_best_normalized_mse": float(signal["normalized_mse"]),
                    "full_train_best_filtered_response_cosine": float(base["full_cosine"]),
                    "full_train_best_normalized_mse": float(base["full_mse"]),
                    "normalized_mse_improvement_signal_vs_hash1024": (
                        float(base["hash_mse"]) - float(signal["normalized_mse"])
                    ),
                }
                _append_checkpoint(checkpoint, row)
                completed[vector_index] = row
                new_windows_done += 1
                print(
                    json.dumps(
                        {
                            "status": "checkpointed_signal_aware_retention_window",
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
                _write_final_csv(
                    output_dir / f"{utterance.utterance_id}__signal_aware_retention_windows.csv",
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
        "status": "signal_aware_retention_audit_in_progress" if incomplete else "selected_items_complete",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "reused_existing_signal_bank": reused_bank,
        "signal_aware_retained_count": int(bank.shape[0]),
        "signal_aware_bucket_count": metadata.get("bucket_count"),
        "retention_cap_per_bucket": RETENTION_CAP,
        "new_windows_completed_this_run": new_windows_done,
        "incomplete_items": incomplete,
        "heldout_used_to_select_codewords": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "production_codebook_replaced": False,
    }
    _atomic_json(output_dir / "residual_codebook_signal_aware_retention_v1_progress.json", progress)

    all_rows: list[dict[str, object]] = []
    complete = True
    for utterance in heldout:
        final_csv = output_dir / f"{utterance.utterance_id}__signal_aware_retention_windows.csv"
        if not final_csv.exists():
            complete = False
            break
        loaded = _load_checkpoint(final_csv, utterance_id=utterance.utterance_id)
        all_rows.extend(loaded.values())

    if not complete:
        return progress

    all_rows.sort(key=lambda row: (str(row["utterance_id"]), int(row["vector_index"])))
    report: dict[str, object] = {
        "status": "ready_for_signal_aware_retention_review",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "heldout_window_count": len(all_rows),
        "signal_aware_retained_count": int(bank.shape[0]),
        "equal_budget_hash1024_retained_count": metadata.get("equal_budget_hash1024_retained_count"),
        "equal_budget_match": metadata.get("equal_budget_match"),
        "retention_cap_per_bucket": RETENTION_CAP,
        "descriptor_features": list(DESCRIPTOR_FEATURES),
        "descriptor_quantile_bins": DESCRIPTOR_QUANTILE_BINS,
        "coverage": {
            "hash1024": _coverage(all_rows, "hash1024"),
            "signal1024": _coverage(all_rows, "signal1024"),
            "full_train": _coverage(all_rows, "full_train"),
        },
        "equal_budget_comparison": _equal_budget_comparison(all_rows),
        "retention_rule": metadata.get("selection_rule"),
        "heldout_used_to_select_codewords": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "production_codebook_replaced": False,
        "metrics_can_accept_product_quality": False,
        "next_action": (
            "if_signal1024_materially_beats_equal_budget_hash1024_then_keep_train_only_signal_aware_"
            "retention_line_and_build_listening_oracle;_otherwise_do_not_increase_hash_cap_blindly_and_"
            "reassess_retention_descriptor_or_required_bank_capacity"
        ),
    }
    _atomic_json(output_dir / "residual_codebook_signal_aware_retention_v1_report.json", report)
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
            run_signal_aware_retention_audit(
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
