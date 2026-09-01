"""Level-valid held-out CELP-style oracle for the owned LYKENOX residual codebook.

V1 used a least-squares codevector gain capped at 4.0.  That arbitrary ceiling can leave the
selected residual far below the held-out target level and makes human listening misleading.

V2 tests codebook *shape* capacity separately from excitation gain capacity.  For each held-out
analysis vector it searches only compatible codewords from the owned-train codebook, chooses the
codeword with the highest absolute normalized correlation, applies the signed target-energy/codeword-
energy gain, overlap-adds the selected residual, and passes it through the unchanged production
minimum-phase filter.  The held-out residual is used only to provide oracle index/sign/gain targets.
Those oracle parameters are explicitly invalid for product inference.

This is not post-hoc output normalization: gain is an excitation/codebook parameter applied before
the synthesis filter, exactly at the representation being tested.  No learned model, optimizer,
checkpoint, EQ, denoise, output gain normalization, external voice component, or remote service is
used.  CPU only under LYX-POL-001.
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

from scripts.diagnostic_residual_codebook_oracle_v1 import _candidate_indices, _safe_name
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    CODEVECTOR_SAMPLES,
    POLICY_ID,
    RESIDUAL_CODEBOOK_VERSION,
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


DIAGNOSTIC_VERSION = "owned-residual-codebook-heldout-oracle-v2-level-valid"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
ENERGY_EPSILON = 1.0e-12


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_float_wav(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        waveform.detach().cpu().to(torch.float32).contiguous().numpy(),
        SAMPLE_RATE,
        subtype="FLOAT",
    )


def _rms(waveform: torch.Tensor) -> float:
    return float(torch.sqrt(waveform.to(torch.float64).square().mean().clamp_min(1.0e-30)))


def _oracle_select_level_valid_codevector(
    target: torch.Tensor,
    codewords: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> tuple[torch.Tensor, int, float, float, float]:
    """Select shape by |cosine| and match target energy with a signed excitation gain."""

    candidates = codewords[candidate_indices].to(torch.float32)
    target = target.to(torch.float32).contiguous()
    target_energy = target.square().sum().clamp_min(ENERGY_EPSILON)
    candidate_energy = candidates.square().sum(dim=1).clamp_min(ENERGY_EPSILON)
    dots = torch.mv(candidates, target)
    cosine = dots / torch.sqrt(candidate_energy * target_energy)
    local_index = int(torch.argmax(cosine.abs()))
    global_index = int(candidate_indices[local_index])

    selected_candidate = candidates[local_index]
    sign = 1.0 if float(cosine[local_index]) >= 0.0 else -1.0
    energy_gain = torch.sqrt(target_energy / candidate_energy[local_index])
    signed_gain = float(energy_gain) * sign
    selected = selected_candidate * signed_gain
    mse = float((selected - target).square().mean())
    similarity = float(cosine[local_index].abs())
    return selected.contiguous(), global_index, signed_gain, similarity, mse


def run_residual_codebook_oracle_v2(
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
        else root / "models" / "lykenox_identity" / "evaluation" / "vocoder_minimum_phase_residual_codebook_oracle_v2"
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
            similarities: list[float] = []
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
                selected, code_index, gain, similarity, mse = _oracle_select_level_valid_codevector(
                    target_vectors[frame_index],
                    codewords,
                    candidate_indices,
                )
                selected_vectors.append(selected)
                selected_indices.append(code_index)
                gains.append(gain)
                similarities.append(similarity)
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
            prediction_path = output_dir / f"{stem}__residual_codebook_oracle_v2.wav"
            residual_path = output_dir / f"{stem}__selected_codebook_residual_v2.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(residual_path, codebook_residual)
            _write_float_wav(reference_path, reference)

            prediction_rms = _rms(prediction)
            reference_rms = _rms(reference)
            target_residual_rms = _rms(target_residual)
            selected_residual_rms = _rms(codebook_residual)
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
                    "mean_abs_oracle_gain": sum(abs(value) for value in gains) / float(len(gains)),
                    "min_oracle_gain": min(gains),
                    "max_oracle_gain": max(gains),
                    "mean_absolute_cosine_similarity": sum(similarities) / float(len(similarities)),
                    "mean_residual_window_mse": sum(residual_mse) / float(len(residual_mse)),
                    "target_residual_rms": target_residual_rms,
                    "selected_codebook_residual_rms": selected_residual_rms,
                    "selected_to_target_residual_rms_ratio": selected_residual_rms / max(target_residual_rms, 1.0e-12),
                    "prediction_rms": prediction_rms,
                    "reference_rms": reference_rms,
                    "prediction_to_reference_rms_ratio": prediction_rms / max(reference_rms, 1.0e-12),
                    "prediction_samples": int(prediction.numel()),
                    "reference_samples": int(reference.numel()),
                    "exact_output_length": int(prediction.numel()) == expected_samples,
                    "residual_codebook_oracle_v2": str(prediction_path),
                    "selected_codebook_residual_v2": str(residual_path),
                    "reference": str(reference_path),
                    "real_residual_resynthesis_ceiling": str(real_residual_ceiling),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_level_valid_residual_codebook_oracle_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "codebook_version": RESIDUAL_CODEBOOK_VERSION,
        "codebook_source_split": metadata.get("source_split"),
        "codebook_tensor_sha256": metadata.get("tensor_sha256"),
        "heldout_split": split,
        "selection_rule": "max_absolute_normalized_correlation_with_signed_target_energy_gain",
        "v1_arbitrary_max_gain_removed": True,
        "gain_applied_at_excitation_before_filter": True,
        "posthoc_output_gain_normalization_used": False,
        "heldout_residual_used_only_as_oracle_search_and_gain_target": True,
        "heldout_residual_added_to_codebook": False,
        "oracle_indices_signs_or_gains_valid_for_product_inference": False,
        "analysis_by_synthesis_oracle_only": True,
        "model_used": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "third_party_voice_component_used": False,
        "remote_inference_used": False,
        "production_renderer_modified": False,
        "predicted_duration_modified": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_to_level_valid_codebook_oracle_vs_reference_and_real_residual_ceiling_before_training_any_selector",
    }
    _atomic_json(output_dir / "residual_codebook_oracle_v2_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_ITEMS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_residual_codebook_oracle_v2(
                args.root,
                split="val",
                max_items=args.heldout_items,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
