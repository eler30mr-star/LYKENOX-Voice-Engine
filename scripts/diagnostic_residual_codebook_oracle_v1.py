"""Held-out CELP-style oracle capacity test for the owned LYKENOX residual codebook.

This diagnostic answers one question only: does a codebook made exclusively from owned ``train``
real residual windows contain enough excitation detail to approach the clean Step-3f ceiling on
complete held-out ``val`` utterances?

For each held-out residual analysis vector, the diagnostic searches compatible train codewords and
uses the least-squares non-negative scalar gain that minimizes residual-domain squared error.  This
is analysis-by-synthesis oracle information: the target held-out residual is available only because
this is a diagnostic.  The selected indices/gains are therefore NOT a product inference mechanism,
are never added back into the codebook, and cannot authorize production quality.

No learned model, optimizer, checkpoint, external voice component, remote service, EQ, denoise,
gain normalization, or duration modification is used.  CPU only under LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    CODEVECTOR_SAMPLES,
    DEFAULT_MAX_PER_BUCKET,
    POLICY_ID,
    RESIDUAL_CODEBOOK_VERSION,
    _conditioning_bucket,
    build_owned_residual_codebook,
    load_owned_residual_codebook,
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
    render_time_varying_minimum_phase,
)


DIAGNOSTIC_VERSION = "owned-residual-codebook-heldout-oracle-v1"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
MAX_ORACLE_GAIN = 4.0
F0_SEARCH_RADIUS_HZ = 40
PERIODICITY_BIN_RADIUS = 1


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _write_float_wav(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = waveform.detach().cpu().to(torch.float32).contiguous().numpy()
    sf.write(str(path), values, SAMPLE_RATE, subtype="FLOAT")


def _bucket_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    buckets = metadata.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise RuntimeError("residual codebook has no bucket index")
    return [dict(item) for item in buckets]


def _candidate_indices(
    metadata: dict[str, Any],
    *,
    f0_hz: float,
    voiced: float,
    periodicity: float,
) -> torch.Tensor:
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
        if target_state != "unvoiced" and abs(int(item["f0_bin_hz"]) - target_f0_bin) > F0_SEARCH_RADIUS_HZ:
            return False
        return True

    selected = [item for item in buckets if compatible(item)]
    if not selected:
        selected = [item for item in buckets if item["voicing_state"] == target_state]
    if not selected:
        selected = buckets

    ranges: list[torch.Tensor] = []
    for item in selected:
        start = int(item["start_index"])
        count = int(item["count"])
        if count > 0:
            ranges.append(torch.arange(start, start + count, dtype=torch.long))
    if not ranges:
        raise RuntimeError("no codebook candidates available for oracle search")
    return torch.cat(ranges, dim=0)


def _oracle_select_codevector(
    target: torch.Tensor,
    codewords: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> tuple[torch.Tensor, int, float, float]:
    candidates = codewords[candidate_indices]
    target = target.to(torch.float32).contiguous()
    dots = torch.mv(candidates, target)
    norms = candidates.square().sum(dim=1).clamp_min(1.0e-12)
    gains = (dots / norms).clamp(0.0, MAX_ORACLE_GAIN)
    target_energy = target.square().sum()
    errors = target_energy - 2.0 * gains * dots + gains.square() * norms
    local_index = int(torch.argmin(errors))
    global_index = int(candidate_indices[local_index])
    gain = float(gains[local_index])
    mse = float(errors[local_index].clamp_min(0.0) / float(CODEVECTOR_SAMPLES))
    selected = candidates[local_index] * gains[local_index]
    return selected.contiguous(), global_index, gain, mse


def run_residual_codebook_oracle(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    tensor_path: Path | None = None,
    index_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("held-out residual codebook oracle must not run on train")
    if max_items < 1 or max_items > 3:
        raise ValueError("max_items must be between 1 and 3")

    root = Path(root).resolve()
    calibration_dir = root / "models" / "lykenox_identity" / "calibration"
    tensor_path = Path(tensor_path).resolve() if tensor_path is not None else calibration_dir / "residual_codebook_v1.pt"
    index_path = Path(index_path).resolve() if index_path is not None else calibration_dir / "residual_codebook_v1.json"
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "evaluation" / "vocoder_minimum_phase_residual_codebook_oracle_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    codewords, metadata = load_owned_residual_codebook(tensor_path, index_path)
    if metadata.get("source_split") != "train":
        raise RuntimeError("oracle codebook is contaminated by non-train identity data")

    utterances = collect_owned_vocoder_utterances(root, split=split, max_items=max_items)
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            expected_samples = frame_count * HOP_LENGTH
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            target_residual, cepstrum, extension_frames = extract_owned_real_residual(
                reference,
                frame_count=frame_count,
            )
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("held-out oracle cepstrum geometry changed")
            target_vectors = residual_analysis_vectors(target_residual)
            if int(target_vectors.shape[0]) != frame_count + 1:
                raise RuntimeError("held-out residual analysis geometry changed")

            selected_vectors: list[torch.Tensor] = []
            selected_indices: list[int] = []
            gains: list[float] = []
            residual_mse: list[float] = []
            candidate_counts: list[int] = []

            for frame_index in range(frame_count + 1):
                conditioning_index = min(frame_index, frame_count - 1)
                candidate_indices = _candidate_indices(
                    metadata,
                    f0_hz=float(utterance.f0_hz[conditioning_index]),
                    voiced=float(utterance.voiced[conditioning_index]),
                    periodicity=float(utterance.periodicity[conditioning_index]),
                )
                selected, code_index, gain, mse = _oracle_select_codevector(
                    target_vectors[frame_index],
                    codewords,
                    candidate_indices,
                )
                selected_vectors.append(selected)
                selected_indices.append(code_index)
                gains.append(gain)
                residual_mse.append(mse)
                candidate_counts.append(int(candidate_indices.numel()))

            codebook_residual = residual_synthesis_from_analysis_vectors(
                torch.stack(selected_vectors, dim=0),
                output_samples=expected_samples,
            )
            prediction = render_time_varying_minimum_phase(
                codebook_residual.unsqueeze(0),
                cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            if prediction.shape != reference.shape or codebook_residual.shape != reference.shape:
                raise RuntimeError("codebook oracle output length mismatch")
            if not bool(torch.isfinite(prediction).all() and torch.isfinite(codebook_residual).all()):
                raise RuntimeError("codebook oracle produced non-finite audio")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__residual_codebook_oracle.wav"
            residual_path = output_dir / f"{stem}__selected_codebook_residual.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(residual_path, codebook_residual)
            _write_float_wav(reference_path, reference)

            real_residual_ceiling = (
                root
                / "models"
                / "lykenox_identity"
                / "evaluation"
                / "vocoder_minimum_phase_oracle_real_residual_v1"
                / f"{stem}__real_residual_resynthesis.wav"
            )
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "terminal_transfer_extension_frames": extension_frames,
                    "oracle_search_windows": frame_count + 1,
                    "unique_codewords_used": len(set(selected_indices)),
                    "mean_candidate_count": sum(candidate_counts) / float(len(candidate_counts)),
                    "mean_oracle_gain": sum(gains) / float(len(gains)),
                    "min_oracle_gain": min(gains),
                    "max_oracle_gain": max(gains),
                    "mean_residual_window_mse": sum(residual_mse) / float(len(residual_mse)),
                    "prediction_samples": int(prediction.numel()),
                    "reference_samples": int(reference.numel()),
                    "exact_output_length": int(prediction.numel()) == expected_samples,
                    "residual_codebook_oracle": str(prediction_path),
                    "selected_codebook_residual": str(residual_path),
                    "reference": str(reference_path),
                    "real_residual_resynthesis_ceiling": str(real_residual_ceiling),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_residual_codebook_oracle_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "codebook_version": RESIDUAL_CODEBOOK_VERSION,
        "codebook_source_split": metadata.get("source_split"),
        "codebook_tensor_sha256": metadata.get("tensor_sha256"),
        "heldout_split": split,
        "heldout_residual_used_only_as_oracle_search_target": True,
        "heldout_residual_added_to_codebook": False,
        "oracle_indices_or_gains_valid_for_product_inference": False,
        "analysis_by_synthesis_oracle_only": True,
        "model_used": False,
        "model_instantiated": False,
        "training_executed": False,
        "optimizer_created": False,
        "parameter_update_executed": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "third_party_voice_component_used": False,
        "remote_inference_used": False,
        "production_renderer_modified": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_to_codebook_oracle_vs_reference_and_real_residual_ceiling_before_training_any_selector",
    }
    _atomic_json(output_dir / "residual_codebook_oracle_report.json", report)
    return report


def run_build_and_oracle(
    root: Path,
    *,
    max_train_items: int = 1_000_000,
    max_per_bucket: int = DEFAULT_MAX_PER_BUCKET,
    heldout_items: int = DEFAULT_ITEMS,
) -> dict[str, object]:
    root = Path(root).resolve()
    build = build_owned_residual_codebook(
        root,
        split="train",
        max_items=max_train_items,
        max_per_bucket=max_per_bucket,
    )
    oracle = run_residual_codebook_oracle(
        root,
        split="val",
        max_items=heldout_items,
    )
    return {
        "status": oracle["status"],
        "policy_id": POLICY_ID,
        "device": "cpu",
        "codebook_build_status": build["status"],
        "retained_codeword_count": build["retained_codeword_count"],
        "bucket_count": build["bucket_count"],
        "heldout_item_count": len(oracle["items"]),
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "selector_training_authorized_by_this_run": False,
        "next_action": oracle["next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-train-items", type=int, default=1_000_000)
    parser.add_argument("--max-per-bucket", type=int, default=DEFAULT_MAX_PER_BUCKET)
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_ITEMS)
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    if args.oracle_only:
        result = run_residual_codebook_oracle(
            args.root,
            split="val",
            max_items=args.heldout_items,
        )
    else:
        result = run_build_and_oracle(
            args.root,
            max_train_items=args.max_train_items,
            max_per_bucket=args.max_per_bucket,
            heldout_items=args.heldout_items,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
