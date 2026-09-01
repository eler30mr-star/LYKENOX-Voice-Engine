"""Synthesis-domain held-out oracle for the owned LYKENOX residual codebook.

V2 and V3 selected codewords from residual-domain similarity/continuity and remained perceptually
``gangoso``.  The clean identity roundtrip proved that the 512/256 sqrt-Hann representation, OLA,
and frozen minimum-phase renderer are not the defect when the correct residual trajectory is used.

V4 therefore scores the codebook where quality actually matters: after the frozen renderer.  For
each held-out residual analysis vector, a broad owned-train candidate set is preselected only for
CPU tractability.  The target vector and every candidate are then converted into their exact local
waveform contribution through the same time-varying minimum-phase renderer.  A non-negative oracle
excitation gain is solved in that synthesis domain and the candidate with minimum filtered-response
MSE is selected.  The local response implementation is mathematically equivalent to inserting one
analysis vector into the normal sqrt-Hann OLA excitation and rendering it through the production
renderer, but evaluates only the output blocks affected by that vector.

The held-out residual/cepstrum are diagnostic oracle targets only and are never added to the train
codebook.  Selected indices/gains are invalid for product inference.  No model, optimizer, training,
checkpoint, production change, post-hoc output normalization, third-party voice component, or
remote service is used. CPU only under LYX-POL-001.
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
    _sqrt_hann,
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
    one_sided_real_cepstrum_to_minimum_phase_fir,
    render_time_varying_minimum_phase,
)


DIAGNOSTIC_VERSION = "owned-residual-codebook-heldout-oracle-v4-synthesis-domain"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
PRESELECT_K = 64
ENERGY_EPSILON = 1.0e-12
LOCAL_CONV_FFT_SIZE = 2048


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


def _preselect_residual_candidates(
    target: torch.Tensor,
    codewords: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    top_k: int = PRESELECT_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broad CPU preselection by signed residual cosine; final choice is synthesis-domain only."""

    candidates = codewords[candidate_indices].to(torch.float32)
    target = target.to(torch.float32).contiguous()
    target_energy = target.square().sum().clamp_min(ENERGY_EPSILON)
    candidate_energy = candidates.square().sum(dim=1).clamp_min(ENERGY_EPSILON)
    cosine = torch.mv(candidates, target) / torch.sqrt(candidate_energy * target_energy)
    count = min(int(top_k), int(candidates.shape[0]))
    if count < 1:
        raise RuntimeError("no codebook candidates available for synthesis-domain oracle")
    _, local = torch.topk(cosine, k=count, largest=True, sorted=True)
    return (
        candidate_indices[local].to(torch.long).contiguous(),
        candidates[local].contiguous(),
        cosine[local].contiguous(),
    )


def _local_filtered_vector_response(
    vectors: torch.Tensor,
    *,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
) -> torch.Tensor:
    """Return exact local renderer output contributed by each analysis vector in ``vectors``.

    ``vectors`` are the sqrt-Hann analysis vectors stored by the codebook.  The normal synthesis
    path multiplies them by sqrt-Hann once more before OLA.  Vector ``j`` is inserted at padded OLA
    offset ``j*hop`` and the final crop removes one hop, so its excitation contribution starts at
    global sample ``(j-1)*hop``.  Convolution with each frame FIR plus the renderer's exact frame
    interpolation is evaluated only on affected output blocks.
    """

    if vectors.ndim != 2 or int(vectors.shape[1]) != CODEVECTOR_SAMPLES:
        raise ValueError("vectors must have shape [K, codevector_samples]")
    if vector_index < 0 or vector_index > frame_count:
        raise ValueError("vector_index outside frame_count+1 analysis geometry")
    if filter_fft.ndim != 2 or int(filter_fft.shape[0]) != frame_count:
        raise ValueError("filter_fft frame geometry mismatch")

    vectors = vectors.to(torch.float32).contiguous()
    synthesis_window = _sqrt_hann(dtype=vectors.dtype)
    excitation_piece = vectors * synthesis_window.unsqueeze(0)

    total_samples = frame_count * HOP_LENGTH
    original_start = (vector_index - 1) * HOP_LENGTH
    left_clip = max(0, -original_start)
    right_clip = max(0, original_start + CODEVECTOR_SAMPLES - total_samples)
    right_boundary = CODEVECTOR_SAMPLES - right_clip if right_clip else CODEVECTOR_SAMPLES
    excitation_piece = excitation_piece[:, left_clip:right_boundary].contiguous()
    global_start = max(0, original_start)
    if int(excitation_piece.shape[1]) < 1:
        return torch.zeros(vectors.shape[0], 0, dtype=vectors.dtype)

    convolution_length = int(excitation_piece.shape[1]) + N_FFT - 1
    if convolution_length > LOCAL_CONV_FFT_SIZE:
        raise RuntimeError("local convolution FFT size is too small")
    excitation_fft = torch.fft.rfft(
        excitation_piece,
        n=LOCAL_CONV_FFT_SIZE,
        dim=-1,
    )

    support_end = min(total_samples, global_start + convolution_length)
    first_block = max(0, global_start // HOP_LENGTH)
    last_block = min(frame_count - 1, (support_end - 1) // HOP_LENGTH)
    if last_block < first_block:
        return torch.zeros(vectors.shape[0], 0, dtype=vectors.dtype)

    alpha = torch.linspace(0.0, 1.0, HOP_LENGTH, dtype=vectors.dtype)
    blocks: list[torch.Tensor] = []
    cached_convolutions: dict[int, torch.Tensor] = {}

    def convolution_for_filter(frame_index: int) -> torch.Tensor:
        cached = cached_convolutions.get(frame_index)
        if cached is None:
            cached = torch.fft.irfft(
                excitation_fft * filter_fft[frame_index].unsqueeze(0),
                n=LOCAL_CONV_FFT_SIZE,
                dim=-1,
            )[:, :convolution_length].contiguous()
            cached_convolutions[frame_index] = cached
        return cached

    for block_index in range(first_block, last_block + 1):
        block_start = block_index * HOP_LENGTH
        block_end = block_start + HOP_LENGTH
        block = torch.zeros(vectors.shape[0], HOP_LENGTH, dtype=vectors.dtype)
        global_left = max(block_start, global_start)
        global_right = min(block_end, global_start + convolution_length, total_samples)
        if global_right > global_left:
            source_left = global_left - global_start
            source_right = global_right - global_start
            dest_left = global_left - block_start
            dest_right = global_right - block_start
            current = convolution_for_filter(block_index)[:, source_left:source_right]
            if block_index == 0:
                values = current
            else:
                previous = convolution_for_filter(block_index - 1)[:, source_left:source_right]
                local_alpha = alpha[dest_left:dest_right].unsqueeze(0)
                values = previous + (current - previous) * local_alpha
            block[:, dest_left:dest_right] = values
        blocks.append(block)

    return torch.cat(blocks, dim=-1).contiguous()


def _select_synthesis_domain_codevector(
    target: torch.Tensor,
    codewords: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
) -> tuple[torch.Tensor, int, float, float, float, int]:
    pre_indices, pre_candidates, pre_cosine = _preselect_residual_candidates(
        target,
        codewords,
        candidate_indices,
    )
    all_vectors = torch.cat((target.unsqueeze(0).to(torch.float32), pre_candidates), dim=0)
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
    difference = candidate_responses * gains.unsqueeze(1) - target_response.unsqueeze(0)
    mse = difference.square().mean(dim=1)
    local_index = int(torch.argmin(mse))

    response_cosine = dots[local_index] / torch.sqrt(
        candidate_energy[local_index] * target_energy
    )
    gain = float(gains[local_index])
    return (
        (pre_candidates[local_index] * gains[local_index]).contiguous(),
        int(pre_indices[local_index]),
        gain,
        float(mse[local_index]),
        float(response_cosine),
        int(pre_candidates.shape[0]),
    )


def run_synthesis_domain_oracle(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    tensor_path: Path | None = None,
    index_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("held-out synthesis-domain oracle must not run on train")
    if max_items < 1 or max_items > 3:
        raise ValueError("max_items must be between 1 and 3")

    root = Path(root).resolve()
    calibration_dir = root / "models" / "lykenox_identity" / "calibration"
    tensor_path = Path(tensor_path).resolve() if tensor_path is not None else calibration_dir / "residual_codebook_v1.pt"
    index_path = Path(index_path).resolve() if index_path is not None else calibration_dir / "residual_codebook_v1.json"
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "evaluation" / "vocoder_minimum_phase_residual_codebook_oracle_v4_synthesis_domain"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    codewords, metadata = load_owned_residual_codebook(tensor_path, index_path)
    if metadata.get("source_split") != "train":
        raise RuntimeError("synthesis-domain oracle codebook is contaminated by non-train identity data")

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

            filters = one_sided_real_cepstrum_to_minimum_phase_fir(
                cepstrum,
                n_fft=N_FFT,
            ).to(torch.float32)
            filter_fft = torch.fft.rfft(filters, n=LOCAL_CONV_FFT_SIZE, dim=-1).contiguous()

            selected_vectors: list[torch.Tensor] = []
            selected_indices: list[int] = []
            gains: list[float] = []
            response_mse: list[float] = []
            response_cosines: list[float] = []
            candidate_counts: list[int] = []

            for vector_index in range(frame_count + 1):
                conditioning_index = min(vector_index, frame_count - 1)
                allowed = _candidate_indices(
                    metadata,
                    f0_hz=float(utterance.f0_hz[conditioning_index]),
                    voiced=float(utterance.voiced[conditioning_index]),
                    periodicity=float(utterance.periodicity[conditioning_index]),
                )
                selected, code_index, gain, mse, response_cosine, preselected_count = (
                    _select_synthesis_domain_codevector(
                        target_vectors[vector_index],
                        codewords,
                        allowed,
                        vector_index=vector_index,
                        filter_fft=filter_fft,
                        frame_count=frame_count,
                    )
                )
                selected_vectors.append(selected)
                selected_indices.append(code_index)
                gains.append(gain)
                response_mse.append(mse)
                response_cosines.append(response_cosine)
                candidate_counts.append(preselected_count)

            selected_tensor = torch.stack(selected_vectors, dim=0).contiguous()
            codebook_residual = residual_synthesis_from_analysis_vectors(
                selected_tensor,
                output_samples=expected_samples,
            )
            prediction = render_time_varying_minimum_phase(
                codebook_residual.unsqueeze(0),
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
            if prediction.shape != reference.shape or codebook_residual.shape != reference.shape:
                raise RuntimeError("synthesis-domain oracle output length mismatch")
            if not bool(
                torch.isfinite(prediction).all()
                and torch.isfinite(codebook_residual).all()
                and torch.isfinite(identity_roundtrip).all()
            ):
                raise RuntimeError("synthesis-domain oracle produced non-finite audio")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__residual_codebook_oracle_v4_synthesis_domain.wav"
            residual_path = output_dir / f"{stem}__selected_codebook_residual_v4_synthesis_domain.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(residual_path, codebook_residual)
            _write_float_wav(identity_path, identity_roundtrip)
            _write_float_wav(reference_path, reference)

            prediction_rms = _rms(prediction)
            reference_rms = _rms(reference)
            target_residual_rms = _rms(target_residual)
            selected_residual_rms = _rms(codebook_residual)
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "terminal_transfer_extension_frames": extension_frames,
                    "oracle_search_windows": frame_count + 1,
                    "preselect_k": PRESELECT_K,
                    "selection_domain": "exact_local_frozen_renderer_waveform_contribution",
                    "gain_domain": "exact_local_frozen_renderer_waveform_contribution",
                    "gain_non_negative": True,
                    "unique_codewords_used": len(set(selected_indices)),
                    "mean_oracle_gain": sum(gains) / float(len(gains)),
                    "min_oracle_gain": min(gains),
                    "max_oracle_gain": max(gains),
                    "mean_filtered_response_cosine": sum(response_cosines) / float(len(response_cosines)),
                    "minimum_filtered_response_cosine": min(response_cosines),
                    "mean_filtered_response_mse": sum(response_mse) / float(len(response_mse)),
                    "mean_preselected_candidate_count": sum(candidate_counts) / float(len(candidate_counts)),
                    "target_residual_rms": target_residual_rms,
                    "selected_codebook_residual_rms": selected_residual_rms,
                    "selected_to_target_residual_rms_ratio": selected_residual_rms / max(target_residual_rms, 1.0e-12),
                    "prediction_rms": prediction_rms,
                    "reference_rms": reference_rms,
                    "prediction_to_reference_rms_ratio": prediction_rms / max(reference_rms, 1.0e-12),
                    "residual_codebook_oracle_v4_synthesis_domain": str(prediction_path),
                    "selected_codebook_residual_v4_synthesis_domain": str(residual_path),
                    "identity_roundtrip_ceiling": str(identity_path),
                    "reference": str(reference_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_synthesis_domain_residual_codebook_oracle_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "codebook_version": RESIDUAL_CODEBOOK_VERSION,
        "codebook_source_split": metadata.get("source_split"),
        "codebook_tensor_sha256": metadata.get("tensor_sha256"),
        "heldout_split": split,
        "selection_rule": "broad_residual_preselection_then_exact_local_renderer_response_mse",
        "gain_rule": "non_negative_least_squares_in_exact_local_renderer_response_domain",
        "local_response_exactly_matches_single_vector_renderer_contribution_required": True,
        "heldout_residual_used_only_as_oracle_target": True,
        "heldout_residual_added_to_codebook": False,
        "oracle_indices_or_gains_valid_for_product_inference": False,
        "model_used": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "third_party_voice_component_used": False,
        "remote_inference_used": False,
        "production_renderer_modified": False,
        "predicted_duration_modified": False,
        "posthoc_output_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_to_synthesis_domain_codebook_oracle_vs_identity_roundtrip_ceiling_before_any_selector_training",
    }
    _atomic_json(output_dir / "residual_codebook_oracle_v4_synthesis_domain_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_ITEMS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_synthesis_domain_oracle(
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
