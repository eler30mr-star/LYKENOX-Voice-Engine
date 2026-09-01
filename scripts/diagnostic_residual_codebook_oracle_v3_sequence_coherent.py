"""Sequence-coherent held-out oracle for the owned LYKENOX residual codebook.

V2 proved that level alone is not enough: independently replacing every real-residual analysis
window with the strongest absolute-correlation train codeword produced intelligible but gangoso
speech.  The positive identity roundtrip then proved that the 512/256 sqrt-Hann representation and
frozen minimum-phase filter remain clean when the exact residual trajectory is preserved.

V3 therefore isolates the remaining variable: *sequence coherence*.  It keeps only positive-polarity
codeword matches, assigns each candidate a positive target-energy excitation gain, and performs a
Viterbi-style oracle search over complete held-out utterances.  The path cost combines target-window
shape error with overlap continuity between neighboring selected codewords after gain application.

The held-out residual is available only as a diagnostic oracle target.  It is never added to the
owned-train codebook.  Selected indices/gains/path decisions are invalid for product inference.
No learned model, optimizer, checkpoint, production renderer change, post-hoc output normalization,
third-party voice component, or remote service is used. CPU only under LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import math
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
    render_time_varying_minimum_phase,
)


DIAGNOSTIC_VERSION = "owned-residual-codebook-heldout-oracle-v3-sequence-coherent"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
TOP_K = 12
EMISSION_WEIGHT = 1.0
CONTINUITY_WEIGHT = 1.0
ENERGY_EPSILON = 1.0e-12
OVERLAP_WINDOW_FLOOR = 0.20


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


def _top_positive_candidates(
    target: torch.Tensor,
    codewords: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    top_k: int = TOP_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (indices, scaled_vectors, cosine, gains) using positive-polarity matches only."""

    candidates = codewords[candidate_indices].to(torch.float32)
    target = target.to(torch.float32).contiguous()
    target_energy = target.square().sum().clamp_min(ENERGY_EPSILON)
    candidate_energy = candidates.square().sum(dim=1).clamp_min(ENERGY_EPSILON)
    dots = torch.mv(candidates, target)
    cosine = dots / torch.sqrt(candidate_energy * target_energy)

    # V2 used abs(cosine) and could invert individual windows by 180 degrees.  V3 forbids
    # per-window polarity inversion.  If every compatible candidate is negative, fall back to the
    # least-negative candidates with zero-clamped cosine ranking rather than introducing a sign flip.
    positive_score = cosine.clamp_min(0.0)
    count = min(int(top_k), int(candidates.shape[0]))
    if count < 1:
        raise RuntimeError("no codebook candidates available")
    _, local = torch.topk(positive_score, k=count, largest=True, sorted=True)
    selected_indices = candidate_indices[local]
    selected_candidates = candidates[local]
    selected_energy = candidate_energy[local]
    selected_cosine = cosine[local]
    gains = torch.sqrt(target_energy / selected_energy)
    scaled = selected_candidates * gains.unsqueeze(1)
    return (
        selected_indices.to(torch.long).contiguous(),
        scaled.contiguous(),
        selected_cosine.contiguous(),
        gains.contiguous(),
    )


def _overlap_unwindowed(vectors: torch.Tensor, *, right: bool) -> torch.Tensor:
    """Approximate the underlying residual samples in the 256-sample neighboring overlap."""

    if vectors.ndim != 2 or int(vectors.shape[1]) != CODEVECTOR_SAMPLES:
        raise ValueError("vectors must have shape [K, codevector_samples]")
    window = _sqrt_hann(dtype=vectors.dtype)
    if right:
        values = vectors[:, HOP_LENGTH:]
        weights = window[HOP_LENGTH:]
    else:
        values = vectors[:, :HOP_LENGTH]
        weights = window[:HOP_LENGTH]
    mask = weights >= OVERLAP_WINDOW_FLOOR
    if int(mask.sum()) < HOP_LENGTH // 2:
        raise RuntimeError("overlap comparison mask is unexpectedly small")
    return (values[:, mask] / weights[mask].unsqueeze(0)).contiguous()


def _transition_cost(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Pairwise normalized overlap mismatch for all previous/current candidate combinations."""

    previous_overlap = _overlap_unwindowed(previous, right=True)
    current_overlap = _overlap_unwindowed(current, right=False)
    difference = previous_overlap[:, None, :] - current_overlap[None, :, :]
    mse = difference.square().mean(dim=-1)
    previous_power = previous_overlap.square().mean(dim=-1)[:, None]
    current_power = current_overlap.square().mean(dim=-1)[None, :]
    scale = 0.5 * (previous_power + current_power) + ENERGY_EPSILON
    return (mse / scale).contiguous()


def _sequence_select(
    target_vectors: torch.Tensor,
    codewords: torch.Tensor,
    metadata: dict[str, Any],
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
) -> tuple[torch.Tensor, list[int], list[float], list[float], list[float]]:
    """Viterbi search over top-K positive-polarity candidates for a complete utterance."""

    frame_total = int(target_vectors.shape[0])
    conditioning_frames = int(f0_hz.numel())
    candidate_sets: list[torch.Tensor] = []
    scaled_sets: list[torch.Tensor] = []
    cosine_sets: list[torch.Tensor] = []
    gain_sets: list[torch.Tensor] = []

    for frame_index in range(frame_total):
        conditioning_index = min(frame_index, conditioning_frames - 1)
        allowed = _candidate_indices(
            metadata,
            f0_hz=float(f0_hz[conditioning_index]),
            voiced=float(voiced[conditioning_index]),
            periodicity=float(periodicity[conditioning_index]),
        )
        indices, scaled, cosine, gains = _top_positive_candidates(
            target_vectors[frame_index],
            codewords,
            allowed,
        )
        candidate_sets.append(indices)
        scaled_sets.append(scaled)
        cosine_sets.append(cosine)
        gain_sets.append(gains)

    first_emission = (1.0 - cosine_sets[0].clamp(-1.0, 1.0)) * EMISSION_WEIGHT
    cumulative = first_emission
    backpointers: list[torch.Tensor] = []
    transition_selected_costs: list[torch.Tensor] = []

    for frame_index in range(1, frame_total):
        emission = (1.0 - cosine_sets[frame_index].clamp(-1.0, 1.0)) * EMISSION_WEIGHT
        transition = _transition_cost(scaled_sets[frame_index - 1], scaled_sets[frame_index])
        total = cumulative[:, None] + CONTINUITY_WEIGHT * transition + emission[None, :]
        best_cost, best_previous = torch.min(total, dim=0)
        cumulative = best_cost
        backpointers.append(best_previous.to(torch.long))
        transition_selected_costs.append(transition)

    state = int(torch.argmin(cumulative))
    states = [state]
    for pointer in reversed(backpointers):
        state = int(pointer[state])
        states.append(state)
    states.reverse()
    if len(states) != frame_total:
        raise RuntimeError("sequence backtrace length mismatch")

    selected_vectors: list[torch.Tensor] = []
    selected_indices: list[int] = []
    selected_gains: list[float] = []
    selected_cosines: list[float] = []
    selected_transition_costs: list[float] = [0.0]
    for frame_index, local_state in enumerate(states):
        selected_vectors.append(scaled_sets[frame_index][local_state])
        selected_indices.append(int(candidate_sets[frame_index][local_state]))
        selected_gains.append(float(gain_sets[frame_index][local_state]))
        selected_cosines.append(float(cosine_sets[frame_index][local_state]))
        if frame_index > 0:
            transition = transition_selected_costs[frame_index - 1]
            previous_state = states[frame_index - 1]
            selected_transition_costs.append(float(transition[previous_state, local_state]))

    return (
        torch.stack(selected_vectors, dim=0).contiguous(),
        selected_indices,
        selected_gains,
        selected_cosines,
        selected_transition_costs,
    )


def run_sequence_coherent_oracle(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    tensor_path: Path | None = None,
    index_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("held-out sequence-coherent oracle must not run on train")
    if max_items < 1 or max_items > 3:
        raise ValueError("max_items must be between 1 and 3")

    root = Path(root).resolve()
    calibration_dir = root / "models" / "lykenox_identity" / "calibration"
    tensor_path = Path(tensor_path).resolve() if tensor_path is not None else calibration_dir / "residual_codebook_v1.pt"
    index_path = Path(index_path).resolve() if index_path is not None else calibration_dir / "residual_codebook_v1.json"
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "evaluation" / "vocoder_minimum_phase_residual_codebook_oracle_v3_sequence_coherent"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    codewords, metadata = load_owned_residual_codebook(tensor_path, index_path)
    if metadata.get("source_split") != "train":
        raise RuntimeError("sequence oracle codebook is contaminated by non-train identity data")

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

            selected_vectors, selected_indices, gains, cosines, transitions = _sequence_select(
                target_vectors,
                codewords,
                metadata,
                utterance.f0_hz,
                utterance.voiced,
                utterance.periodicity,
            )
            codebook_residual = residual_synthesis_from_analysis_vectors(
                selected_vectors,
                output_samples=expected_samples,
            )
            prediction = render_time_varying_minimum_phase(
                codebook_residual.unsqueeze(0),
                cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            if prediction.shape != reference.shape or codebook_residual.shape != reference.shape:
                raise RuntimeError("sequence-coherent oracle output length mismatch")
            if not bool(torch.isfinite(prediction).all() and torch.isfinite(codebook_residual).all()):
                raise RuntimeError("sequence-coherent oracle produced non-finite audio")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__residual_codebook_oracle_v3_sequence_coherent.wav"
            residual_path = output_dir / f"{stem}__selected_codebook_residual_v3_sequence_coherent.wav"
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
                    "top_k_per_window": TOP_K,
                    "emission_weight": EMISSION_WEIGHT,
                    "continuity_weight": CONTINUITY_WEIGHT,
                    "per_window_polarity_inversion_allowed": False,
                    "unique_codewords_used": len(set(selected_indices)),
                    "mean_oracle_gain": sum(gains) / float(len(gains)),
                    "min_oracle_gain": min(gains),
                    "max_oracle_gain": max(gains),
                    "mean_positive_cosine_similarity": sum(cosines) / float(len(cosines)),
                    "minimum_selected_cosine_similarity": min(cosines),
                    "mean_selected_overlap_transition_cost": sum(transitions) / float(len(transitions)),
                    "max_selected_overlap_transition_cost": max(transitions),
                    "target_residual_rms": target_residual_rms,
                    "selected_codebook_residual_rms": selected_residual_rms,
                    "selected_to_target_residual_rms_ratio": selected_residual_rms / max(target_residual_rms, 1.0e-12),
                    "prediction_rms": prediction_rms,
                    "reference_rms": reference_rms,
                    "prediction_to_reference_rms_ratio": prediction_rms / max(reference_rms, 1.0e-12),
                    "residual_codebook_oracle_v3_sequence_coherent": str(prediction_path),
                    "selected_codebook_residual_v3_sequence_coherent": str(residual_path),
                    "reference": str(reference_path),
                    "real_residual_resynthesis_ceiling": str(real_residual_ceiling),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_sequence_coherent_residual_codebook_oracle_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "codebook_version": RESIDUAL_CODEBOOK_VERSION,
        "codebook_source_split": metadata.get("source_split"),
        "codebook_tensor_sha256": metadata.get("tensor_sha256"),
        "heldout_split": split,
        "selection_rule": "viterbi_topk_positive_cosine_plus_gain_scaled_overlap_continuity",
        "per_window_polarity_inversion_allowed": False,
        "gain_rule": "positive_target_energy_over_codeword_energy",
        "gain_applied_at_excitation_before_filter": True,
        "posthoc_output_gain_normalization_used": False,
        "heldout_residual_used_only_as_oracle_path_target": True,
        "heldout_residual_added_to_codebook": False,
        "oracle_indices_gains_or_path_valid_for_product_inference": False,
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
        "next_action": "listen_to_sequence_coherent_codebook_oracle_before_any_selector_training_or_production_change",
    }
    _atomic_json(output_dir / "residual_codebook_oracle_v3_sequence_coherent_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_ITEMS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_sequence_coherent_oracle(
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
