"""Synthesis-domain coherent held-out oracle for the owned LYKENOX residual codebook.

V4 moved final codeword evaluation into the correct domain: each broad residual-domain candidate
is passed through the exact local frozen minimum-phase renderer response, with a non-negative
least-squares excitation gain solved in that filtered waveform domain.  V4 still selected each
window independently.

V5 keeps V4's validated preselection, local filtered-response implementation, and non-negative
filtered-domain gain unchanged, but replaces the greedy per-window argmin with a deterministic
bounded beam search.  Each path accumulates the same V4 local filtered-response MSE plus a filtered-
domain continuity cost between consecutive selected candidates over the 256-sample OLA region shared
by their underlying 512-sample / 256-hop excitation windows.

The held-out residual/cepstrum remain diagnostic oracle targets only and are never added to the
owned-train codebook.  Selected indices/gains/path decisions are invalid for product inference.
No learned model, optimizer, checkpoint, production renderer change, post-hoc output normalization,
third-party voice component, or remote service is used. CPU only under LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostic_residual_codebook_oracle_v1 import _candidate_indices, _safe_name
from scripts.diagnostic_residual_codebook_oracle_v4_synthesis_domain import (
    ENERGY_EPSILON,
    LOCAL_CONV_FFT_SIZE,
    PRESELECT_K,
    _local_filtered_vector_response,
    _preselect_residual_candidates,
    _rms,
    _write_float_wav,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
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
    one_sided_real_cepstrum_to_minimum_phase_fir,
    render_time_varying_minimum_phase,
)


DIAGNOSTIC_VERSION = "owned-residual-codebook-heldout-oracle-v5-synthesis-domain-coherent"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
DEFAULT_BEAM_SIZE = 8
DEFAULT_CONTINUITY_WEIGHT = 1.0


@dataclass(frozen=True)
class CandidateSet:
    indices: torch.Tensor
    scaled_vectors: torch.Tensor
    scaled_responses: torch.Tensor
    local_mse: torch.Tensor
    response_cosine: torch.Tensor
    gains: torch.Tensor
    global_response_start: int


@dataclass(frozen=True)
class BeamPath:
    cumulative_cost: float
    states: tuple[int, ...]
    last_state: int


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _local_response_global_start(vector_index: int) -> int:
    """Global waveform sample represented by element zero of V4's local response tensor."""

    if vector_index < 0:
        raise ValueError("vector_index must be non-negative")
    # V4 inserts analysis vector j at cropped excitation start (j-1)*hop and returns complete
    # renderer blocks beginning at max(0, floor(global_start/hop)).  Because starts are hop-aligned,
    # this is exactly max(0, (j-1)*hop).
    return max(0, (vector_index - 1) * HOP_LENGTH)


def _build_synthesis_domain_candidates(
    target: torch.Tensor,
    codewords: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    vector_index: int,
    filter_fft: torch.Tensor,
    frame_count: int,
) -> CandidateSet:
    """Return V4-equivalent candidates/gains without performing its final greedy argmin."""

    pre_indices, pre_candidates, _ = _preselect_residual_candidates(
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

    # Keep V4's non-negative least-squares gain exactly unchanged.
    gains = (dots / candidate_energy).clamp_min(0.0)
    scaled_responses = candidate_responses * gains.unsqueeze(1)
    difference = scaled_responses - target_response.unsqueeze(0)
    local_mse = difference.square().mean(dim=1)
    response_cosine = dots / torch.sqrt(candidate_energy * target_energy)
    scaled_vectors = pre_candidates * gains.unsqueeze(1)

    return CandidateSet(
        indices=pre_indices.to(torch.long).contiguous(),
        scaled_vectors=scaled_vectors.contiguous(),
        scaled_responses=scaled_responses.contiguous(),
        local_mse=local_mse.contiguous(),
        response_cosine=response_cosine.contiguous(),
        gains=gains.contiguous(),
        global_response_start=_local_response_global_start(vector_index),
    )


def _filtered_overlap_discontinuity(
    previous: CandidateSet,
    current: CandidateSet,
    *,
    current_vector_index: int,
) -> torch.Tensor:
    """Pairwise filtered-domain MSE on the 256-sample OLA region shared by adjacent windows."""

    if current_vector_index < 1:
        raise ValueError("current_vector_index must be >= 1 for a transition")

    total_samples_previous_end = previous.global_response_start + int(previous.scaled_responses.shape[1])
    total_samples_current_end = current.global_response_start + int(current.scaled_responses.shape[1])

    # Consecutive analysis vectors i-1 and i overlap in excitation on [(i-1)H, iH).  Evaluate the
    # already-filtered local responses over these exact global waveform coordinates.
    overlap_start = (current_vector_index - 1) * HOP_LENGTH
    overlap_end = current_vector_index * HOP_LENGTH
    overlap_start = max(
        overlap_start,
        previous.global_response_start,
        current.global_response_start,
    )
    overlap_end = min(
        overlap_end,
        total_samples_previous_end,
        total_samples_current_end,
    )
    if overlap_end <= overlap_start:
        raise RuntimeError("adjacent filtered candidate responses have no valid OLA overlap")

    previous_left = overlap_start - previous.global_response_start
    previous_right = overlap_end - previous.global_response_start
    current_left = overlap_start - current.global_response_start
    current_right = overlap_end - current.global_response_start
    previous_overlap = previous.scaled_responses[:, previous_left:previous_right]
    current_overlap = current.scaled_responses[:, current_left:current_right]
    if int(previous_overlap.shape[1]) != int(current_overlap.shape[1]):
        raise RuntimeError("filtered overlap geometry mismatch")
    if int(previous_overlap.shape[1]) != HOP_LENGTH:
        raise RuntimeError("filtered continuity region must be exactly one 256-sample hop")

    difference = previous_overlap[:, None, :] - current_overlap[None, :, :]
    return difference.square().mean(dim=-1).contiguous()


def _beam_select_sequence(
    candidate_sets: list[CandidateSet],
    *,
    beam_size: int = DEFAULT_BEAM_SIZE,
    continuity_weight: float = DEFAULT_CONTINUITY_WEIGHT,
) -> tuple[list[int], list[float]]:
    """Return chosen local candidate states and per-transition continuity costs."""

    if not candidate_sets:
        raise ValueError("candidate_sets must be non-empty")
    if beam_size < 1:
        raise ValueError("beam_size must be positive")
    if continuity_weight < 0.0:
        raise ValueError("continuity_weight must be non-negative")

    first = candidate_sets[0]
    initial_order = sorted(
        range(int(first.local_mse.numel())),
        key=lambda state: (float(first.local_mse[state]), state),
    )[:beam_size]
    beam = [
        BeamPath(
            cumulative_cost=float(first.local_mse[state]),
            states=(state,),
            last_state=state,
        )
        for state in initial_order
    ]

    transition_matrices: list[torch.Tensor] = []
    for vector_index in range(1, len(candidate_sets)):
        previous = candidate_sets[vector_index - 1]
        current = candidate_sets[vector_index]
        transition = _filtered_overlap_discontinuity(
            previous,
            current,
            current_vector_index=vector_index,
        )
        transition_matrices.append(transition)

        expanded: list[BeamPath] = []
        for path in beam:
            previous_state = path.last_state
            for current_state in range(int(current.local_mse.numel())):
                total = (
                    path.cumulative_cost
                    + float(current.local_mse[current_state])
                    + continuity_weight * float(transition[previous_state, current_state])
                )
                expanded.append(
                    BeamPath(
                        cumulative_cost=total,
                        states=path.states + (current_state,),
                        last_state=current_state,
                    )
                )
        expanded.sort(key=lambda item: (item.cumulative_cost, item.states))
        beam = expanded[:beam_size]
        if not beam:
            raise RuntimeError("beam search lost all candidate paths")

    best = min(beam, key=lambda item: (item.cumulative_cost, item.states))
    selected_states = list(best.states)
    if len(selected_states) != len(candidate_sets):
        raise RuntimeError("beam backtrace length mismatch")

    selected_transition_costs: list[float] = [0.0]
    for vector_index in range(1, len(candidate_sets)):
        transition = transition_matrices[vector_index - 1]
        selected_transition_costs.append(
            float(transition[selected_states[vector_index - 1], selected_states[vector_index]])
        )
    return selected_states, selected_transition_costs


def run_synthesis_domain_coherent_oracle(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    beam_size: int = DEFAULT_BEAM_SIZE,
    continuity_weight: float = DEFAULT_CONTINUITY_WEIGHT,
    tensor_path: Path | None = None,
    index_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("held-out synthesis-domain coherent oracle must not run on train")
    if max_items < 1 or max_items > 3:
        raise ValueError("max_items must be between 1 and 3")
    if beam_size < 1 or beam_size > 64:
        raise ValueError("beam_size must be in [1, 64]")
    if continuity_weight < 0.0:
        raise ValueError("continuity_weight must be non-negative")

    root = Path(root).resolve()
    calibration_dir = root / "models" / "lykenox_identity" / "calibration"
    tensor_path = Path(tensor_path).resolve() if tensor_path is not None else calibration_dir / "residual_codebook_v1.pt"
    index_path = Path(index_path).resolve() if index_path is not None else calibration_dir / "residual_codebook_v1.json"
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_residual_codebook_oracle_v5_synthesis_domain_coherent"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    codewords, metadata = load_owned_residual_codebook(tensor_path, index_path)
    if metadata.get("source_split") != "train":
        raise RuntimeError("synthesis-domain coherent oracle codebook is contaminated by non-train identity data")

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

            candidate_sets: list[CandidateSet] = []
            for vector_index in range(frame_count + 1):
                conditioning_index = min(vector_index, frame_count - 1)
                allowed = _candidate_indices(
                    metadata,
                    f0_hz=float(utterance.f0_hz[conditioning_index]),
                    voiced=float(utterance.voiced[conditioning_index]),
                    periodicity=float(utterance.periodicity[conditioning_index]),
                )
                candidate_sets.append(
                    _build_synthesis_domain_candidates(
                        target_vectors[vector_index],
                        codewords,
                        allowed,
                        vector_index=vector_index,
                        filter_fft=filter_fft,
                        frame_count=frame_count,
                    )
                )

            states, transition_costs = _beam_select_sequence(
                candidate_sets,
                beam_size=beam_size,
                continuity_weight=continuity_weight,
            )

            selected_vectors: list[torch.Tensor] = []
            selected_indices: list[int] = []
            gains: list[float] = []
            response_mse: list[float] = []
            response_cosines: list[float] = []
            candidate_counts: list[int] = []
            for candidate_set, state in zip(candidate_sets, states):
                selected_vectors.append(candidate_set.scaled_vectors[state])
                selected_indices.append(int(candidate_set.indices[state]))
                gains.append(float(candidate_set.gains[state]))
                response_mse.append(float(candidate_set.local_mse[state]))
                response_cosines.append(float(candidate_set.response_cosine[state]))
                candidate_counts.append(int(candidate_set.indices.numel()))

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
                raise RuntimeError("synthesis-domain coherent oracle output length mismatch")
            if not bool(
                torch.isfinite(prediction).all()
                and torch.isfinite(codebook_residual).all()
                and torch.isfinite(identity_roundtrip).all()
            ):
                raise RuntimeError("synthesis-domain coherent oracle produced non-finite audio")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__residual_codebook_oracle_v5_synthesis_domain_coherent.wav"
            residual_path = output_dir / f"{stem}__selected_codebook_residual_v5_synthesis_domain_coherent.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            v4_path = (
                root
                / "models"
                / "lykenox_identity"
                / "evaluation"
                / "vocoder_minimum_phase_residual_codebook_oracle_v4_synthesis_domain"
                / f"{stem}__residual_codebook_oracle_v4_synthesis_domain.wav"
            )
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
                    "beam_size": beam_size,
                    "continuity_weight": continuity_weight,
                    "selection_domain": "exact_local_frozen_renderer_waveform_contribution_plus_filtered_overlap_continuity",
                    "gain_domain": "exact_local_frozen_renderer_waveform_contribution",
                    "gain_non_negative": True,
                    "continuity_domain": "filtered_waveform_256_sample_adjacent_ola_overlap",
                    "unique_codewords_used": len(set(selected_indices)),
                    "mean_oracle_gain": sum(gains) / float(len(gains)),
                    "min_oracle_gain": min(gains),
                    "max_oracle_gain": max(gains),
                    "mean_filtered_response_cosine": sum(response_cosines) / float(len(response_cosines)),
                    "minimum_filtered_response_cosine": min(response_cosines),
                    "mean_filtered_response_mse": sum(response_mse) / float(len(response_mse)),
                    "mean_filtered_overlap_discontinuity_mse": sum(transition_costs) / float(len(transition_costs)),
                    "max_filtered_overlap_discontinuity_mse": max(transition_costs),
                    "mean_preselected_candidate_count": sum(candidate_counts) / float(len(candidate_counts)),
                    "target_residual_rms": target_residual_rms,
                    "selected_codebook_residual_rms": selected_residual_rms,
                    "selected_to_target_residual_rms_ratio": selected_residual_rms / max(target_residual_rms, 1.0e-12),
                    "prediction_rms": prediction_rms,
                    "reference_rms": reference_rms,
                    "prediction_to_reference_rms_ratio": prediction_rms / max(reference_rms, 1.0e-12),
                    "residual_codebook_oracle_v5_synthesis_domain_coherent": str(prediction_path),
                    "selected_codebook_residual_v5_synthesis_domain_coherent": str(residual_path),
                    "identity_roundtrip_ceiling": str(identity_path),
                    "reference": str(reference_path),
                    "v4_synthesis_domain_comparison": str(v4_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_synthesis_domain_coherent_residual_codebook_oracle_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "codebook_version": RESIDUAL_CODEBOOK_VERSION,
        "codebook_source_split": metadata.get("source_split"),
        "codebook_tensor_sha256": metadata.get("tensor_sha256"),
        "heldout_split": split,
        "v4_source_modified": False,
        "v4_preselection_reused_unchanged": True,
        "v4_local_filtered_response_reused_unchanged": True,
        "gain_rule": "v4_non_negative_least_squares_in_exact_local_renderer_response_domain",
        "sequence_rule": "bounded_deterministic_beam_search_local_filtered_mse_plus_filtered_ola_overlap_mse",
        "beam_size": beam_size,
        "continuity_weight": continuity_weight,
        "continuity_overlap_samples": HOP_LENGTH,
        "heldout_residual_used_only_as_oracle_target": True,
        "heldout_residual_added_to_codebook": False,
        "oracle_indices_gains_or_path_valid_for_product_inference": False,
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
        "next_action": "listen_to_v5_vs_v4_and_identity_roundtrip_ceiling_and_report_before_any_further_iteration",
    }
    _atomic_json(output_dir / "residual_codebook_oracle_v5_synthesis_domain_coherent_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_ITEMS)
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--continuity-weight", type=float, default=DEFAULT_CONTINUITY_WEIGHT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_synthesis_domain_coherent_oracle(
                args.root,
                split="val",
                max_items=args.heldout_items,
                beam_size=args.beam_size,
                continuity_weight=args.continuity_weight,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
