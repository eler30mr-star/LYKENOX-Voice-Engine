"""Phase-exclusive held-out renderer for the trained pitch-synchronous source.

The pitch-synchronous source is the strongest learned source so far, but held-out listening exposed
an end-of-word/phrase whistle.  The V2 phase-continuous renderer still mixed two independent residual
sources sample-wise:

    residual = V2 * (1 - gate) + pitch_sync * gate

where gate followed voiced/periodicity.  During voiced-to-unvoiced transitions this makes two unrelated
excitation phases coexist, which can create beating, tonal chirps, and apparent timbre thickening.

This renderer removes that structural defect without retraining or post-hoc enhancement:

1. pitch-sync has exclusive authority wherever a complete conditioning-derived F0 cycle exists;
2. V2 has exclusive authority outside pitch-sync coverage;
3. no sample-wise source mixture is used;
4. each authority transition is replaced by one short cubic-Hermite bridge whose width is derived
   from the adjacent physical F0 period.  The bridge matches value and first derivative at both ends,
   so the handoff is C1-continuous without simultaneously sounding both sources.

The trained pitch-sync checkpoint, V2 checkpoint, oracle envelope and fixed minimum-phase renderer are
unchanged. No optimizer, training, gain normalization, EQ, denoise, codebook, third-party model/weight,
or remote service is used. Policy: LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import _ola_vectors
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_time_varying_minimum_phase,
)
from lykenox_voice_engine.training.speech_vocoder_pitch_synchronous_cycle_source_train_v1 import (
    POLICY_ID,
    conditioning_cycle_boundaries,
)
from scripts.render_pitch_synchronous_cycle_source_v2_phase_continuous import (
    _load_cycle_model,
    _load_v2_model,
    _rms,
    synthesize_phase_continuous_cycles,
)


EVALUATION_VERSION = "owned-pitch-synchronous-cycle-source-phase-exclusive-handoff-v3"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _write(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().to(torch.float32).numpy(), SAMPLE_RATE, subtype="FLOAT")


def _coverage_runs(coverage: torch.Tensor) -> list[tuple[int, int]]:
    """Return [start, end) runs where pitch-sync has complete-cycle authority."""
    if coverage.ndim != 1:
        raise ValueError("coverage must be one-dimensional")
    active = coverage > 0.5
    if not bool(active.any()):
        return []
    padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.bool),
            active.cpu(),
            torch.zeros(1, dtype=torch.bool),
        )
    )
    delta = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = torch.nonzero(delta == 1, as_tuple=False).flatten().tolist()
    ends = torch.nonzero(delta == -1, as_tuple=False).flatten().tolist()
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _nearest_cycle_period(
    edge: int,
    boundaries: list[tuple[int, int, int, float]],
) -> int:
    if not boundaries:
        return HOP_LENGTH
    best_period = HOP_LENGTH
    best_distance = None
    for left, right, _, _ in boundaries:
        distance = min(abs(int(edge) - int(left)), abs(int(edge) - int(right)))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_period = max(2, int(right) - int(left))
    return best_period


def _bridge_half_width(period_samples: int) -> int:
    """Use one quarter-cycle total handoff support, split around the authority edge."""
    # Total bridge length is approximately period/4, so each side is period/8.  The floor of four
    # samples prevents a degenerate derivative bridge at very high F0; 32 prevents a long envelope
    # deformation at very low F0.  These are geometry guards, not perceptual tuning parameters.
    return max(4, min(32, int(round(float(period_samples) / 8.0))))


def _hermite_segment(
    y0: torch.Tensor,
    slope0: torch.Tensor,
    y1: torch.Tensor,
    slope1: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    """Cubic Hermite segment including both endpoints."""
    if samples < 2:
        raise ValueError("Hermite bridge needs at least two samples")
    t = torch.linspace(0.0, 1.0, samples, dtype=torch.float32)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    span = float(samples - 1)
    return h00 * y0 + h10 * (slope0 * span) + h01 * y1 + h11 * (slope1 * span)


def _apply_c1_bridge_inplace(
    residual: torch.Tensor,
    *,
    edge: int,
    half_width: int,
) -> tuple[int, int] | None:
    """Replace a local authority switch by a value/slope-continuous bridge.

    ``residual`` must already contain exactly one authoritative source on each side of ``edge``.
    No weighted sum of the two source waveforms is computed here.
    """
    if residual.ndim != 1:
        raise ValueError("residual must be one-dimensional")
    total = int(residual.numel())
    left = max(1, int(edge) - int(half_width))
    right = min(total - 2, int(edge) + int(half_width))
    if right - left < 3:
        return None
    y0 = residual[left].clone()
    y1 = residual[right].clone()
    slope0 = (residual[left] - residual[left - 1]).clone()
    slope1 = (residual[right + 1] - residual[right]).clone()
    residual[left : right + 1] = _hermite_segment(
        y0,
        slope0,
        y1,
        slope1,
        right - left + 1,
    )
    return left, right + 1


def synthesize_phase_exclusive_handoff(
    pitch_sync_waveform: torch.Tensor,
    coverage: torch.Tensor,
    v2_residual: torch.Tensor,
    boundaries: list[tuple[int, int, int, float]],
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Select one source at a time and stitch authority changes with C1 bridges."""
    if pitch_sync_waveform.shape != v2_residual.shape or coverage.shape != v2_residual.shape:
        raise ValueError("source/coverage geometry mismatch")
    authority = coverage > 0.5
    residual = torch.where(authority, pitch_sync_waveform, v2_residual).to(torch.float32).contiguous()
    transitions: list[dict[str, int]] = []
    for start, end in _coverage_runs(coverage):
        for edge in (start, end):
            if edge <= 1 or edge >= int(residual.numel()) - 2:
                continue
            period = _nearest_cycle_period(edge, boundaries)
            half_width = _bridge_half_width(period)
            bridge = _apply_c1_bridge_inplace(residual, edge=edge, half_width=half_width)
            if bridge is not None:
                transitions.append(
                    {
                        "edge_sample": int(edge),
                        "period_samples": int(period),
                        "bridge_start_sample": int(bridge[0]),
                        "bridge_end_sample": int(bridge[1]),
                    }
                )
    if not bool(torch.isfinite(residual).all()):
        raise RuntimeError("phase-exclusive handoff produced non-finite residual")
    return residual, transitions


def render_heldout_phase_exclusive_handoff(
    root: Path,
    *,
    heldout_items: int = 3,
    checkpoint: Path | None = None,
    v2_checkpoint: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    if heldout_items < 1 or heldout_items > 3:
        raise ValueError("heldout_items must be in [1,3]")
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root / "models" / "lykenox_identity" / "training" / "pitch_synchronous_cycle_source_v1" / "best.pt"
    )
    v2_checkpoint = (
        Path(v2_checkpoint).resolve()
        if v2_checkpoint is not None
        else root / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    if not v2_checkpoint.exists():
        raise FileNotFoundError(str(v2_checkpoint))

    cycle_model, cycle_payload = _load_cycle_model(checkpoint)
    v2_model = _load_v2_model(v2_checkpoint)
    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_pitch_synchronous_cycle_source_v3_phase_exclusive_handoff"
    )
    utterances = collect_owned_vocoder_utterances(root, "val", max_items=heldout_items)
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            expected_samples = frame_count * HOP_LENGTH
            target_residual, oracle_cepstrum, _ = extract_owned_real_residual(
                utterance.waveform.cpu(), frame_count=frame_count
            )
            v2_vectors = v2_model.generate(
                utterance.mel.unsqueeze(0).cpu(),
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
            )
            v2_residual = _ola_vectors(v2_vectors, output_samples=expected_samples).squeeze(0)

            boundaries = conditioning_cycle_boundaries(utterance)
            pitch_sync_waveform = torch.zeros(expected_samples, dtype=torch.float32)
            coverage = torch.zeros(expected_samples, dtype=torch.float32)
            if boundaries:
                frame_indices = torch.tensor([item[2] for item in boundaries], dtype=torch.long)
                predicted_cycles, _ = cycle_model(
                    utterance.mel.unsqueeze(0).cpu(),
                    utterance.f0_hz.unsqueeze(0).cpu(),
                    utterance.voiced.unsqueeze(0).cpu(),
                    utterance.periodicity.unsqueeze(0).cpu(),
                    frame_indices,
                )
                pitch_sync_waveform, coverage = synthesize_phase_continuous_cycles(
                    predicted_cycles,
                    boundaries,
                    output_samples=expected_samples,
                )

            exclusive_residual, transitions = synthesize_phase_exclusive_handoff(
                pitch_sync_waveform,
                coverage,
                v2_residual,
                boundaries,
            )
            prediction = render_time_varying_minimum_phase(
                exclusive_residual.unsqueeze(0),
                oracle_cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            v2_prediction = render_time_varying_minimum_phase(
                v2_residual.unsqueeze(0),
                oracle_cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            identity = render_time_varying_minimum_phase(
                target_residual.unsqueeze(0),
                oracle_cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            for value in (prediction, v2_prediction, identity):
                if value.shape != reference.shape:
                    raise RuntimeError("phase-exclusive heldout output length mismatch")

            stem = utterance.utterance_id
            prediction_path = output_dir / f"{stem}__phase_exclusive_handoff_source_v3.wav"
            v2_path = output_dir / f"{stem}__v2_baseline_source.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            residual_path = output_dir / f"{stem}__phase_exclusive_handoff_residual.wav"
            _write(prediction_path, prediction)
            _write(v2_path, v2_prediction)
            _write(identity_path, identity)
            _write(reference_path, reference)
            _write(residual_path, exclusive_residual)
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "cycle_count": len(boundaries),
                    "authority_run_count": len(_coverage_runs(coverage)),
                    "handoff_count": len(transitions),
                    "pitch_sync_authority_fraction": float((coverage > 0.5).to(torch.float32).mean()),
                    "prediction_rms": _rms(prediction),
                    "v2_baseline_rms": _rms(v2_prediction),
                    "reference_rms": _rms(reference),
                    "prediction_to_reference_rms_ratio": _rms(prediction) / max(_rms(reference), 1.0e-12),
                    "phase_exclusive_handoff_source_v3": str(prediction_path),
                    "v2_baseline_source": str(v2_path),
                    "identity_roundtrip_ceiling": str(identity_path),
                    "reference": str(reference_path),
                    "phase_exclusive_handoff_residual": str(residual_path),
                    "handoffs": transitions,
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_phase_exclusive_handoff_source_v3_listening",
        "evaluation_version": EVALUATION_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_training_update": int(cycle_payload.get("update", -1)),
        "v2_checkpoint": str(v2_checkpoint),
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "raw_v2_pitch_sync_samplewise_mix_used": False,
        "source_authority": "pitch_sync_inside_complete_cycles_v2_elsewhere",
        "handoff": "period_derived_cubic_hermite_c1_bridge",
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "codebook_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_for_end_of_word_chirp_and_voice_naturalness_vs_v2_and_identity_ceiling",
    }
    _atomic_json(output_dir / "phase_exclusive_handoff_source_v3_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--v2-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            render_heldout_phase_exclusive_handoff(
                args.root,
                heldout_items=args.heldout_items,
                checkpoint=args.checkpoint,
                v2_checkpoint=args.v2_checkpoint,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
