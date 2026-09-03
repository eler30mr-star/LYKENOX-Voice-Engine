"""No-training temporal-phase-increment gate for the accepted candidate magnitude.

Human listening established that candidate STFT magnitude plus target residual phase sounds clean,
while deterministic Griffin-Lim recovery is intelligible but still mildly robotic. This diagnostic
keeps candidate magnitude fixed and isolates whether the missing information is the temporal phase
increment sequence of the real residual rather than absolute per-bin phase.

It emits three controlled candidate-magnitude hybrids:
1) target temporal phase increments + candidate initial per-bin phase;
2) target temporal phase increments + zero initial per-bin phase;
3) candidate temporal phase increments + target initial per-bin phase.

No training, optimizer, renderer modification, post-hoc gain normalization, EQ, denoise, duration
modification, third-party model/service, or new checkpoint is used. Policy: LYX-POL-001.
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

from scripts.diagnostic_residual_phase_magnitude_forensic_v1 import (
    DEFAULT_SCAN_ITEMS,
    DEFAULT_UTTERANCE_IDS,
    _load_candidate,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import extract_pitch_conditioning_v2
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
from lykenox_voice_engine.training.speech_vocoder_residual_statistics_source_train_v1 import (
    DEFAULT_SEED,
    _utterance_seed,
    synthesize_residual_from_statistics,
)


DIAGNOSTIC_VERSION = "owned-candidate-magnitude-temporal-phase-increment-v1"
POLICY_ID = "LYX-POL-001"
OUTPUT_DIR_NAME = "vocoder_candidate_magnitude_temporal_phase_increment_v1"
EPSILON = 1.0e-8


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        waveform.detach().cpu().to(torch.float32).contiguous().numpy(),
        SAMPLE_RATE,
        subtype="FLOAT",
    )


def _stft(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError("residual must be mono [samples]")
    window = torch.hann_window(N_FFT, dtype=value.dtype, device=value.device)
    return torch.stft(
        value,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )


def _istft(spectrum: torch.Tensor, *, length: int, dtype: torch.dtype) -> torch.Tensor:
    window = torch.hann_window(N_FFT, dtype=dtype, device=spectrum.device)
    waveform = torch.istft(
        spectrum,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        length=length,
    )
    if int(waveform.numel()) != length:
        raise RuntimeError("temporal phase increment reconstruction violated exact length")
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("temporal phase increment reconstruction contains non-finite values")
    return waveform.to(torch.float32).contiguous()


def _temporal_phase_increments(spectrum: torch.Tensor) -> torch.Tensor:
    """Wrapped phase advance between consecutive STFT frames, shape [bins, frames-1]."""
    if spectrum.ndim != 2 or not spectrum.is_complex():
        raise ValueError("spectrum must be complex [bins, frames]")
    if int(spectrum.shape[-1]) < 2:
        raise ValueError("spectrum must have at least two frames")
    return torch.angle(spectrum[:, 1:] * torch.conj(spectrum[:, :-1])).contiguous()


def _phase_from_temporal_increments(
    increments: torch.Tensor,
    initial_phase: torch.Tensor,
) -> torch.Tensor:
    if increments.ndim != 2:
        raise ValueError("increments must have shape [bins, frames-1]")
    if initial_phase.shape != increments.shape[:1]:
        raise ValueError("initial_phase must have shape [bins]")
    phase = torch.cat(
        (
            initial_phase.unsqueeze(-1),
            initial_phase.unsqueeze(-1) + torch.cumsum(increments, dim=-1),
        ),
        dim=-1,
    )
    # Keep numerical values bounded without changing the represented complex phase.
    return torch.atan2(torch.sin(phase), torch.cos(phase)).contiguous()


def _hybrid_from_magnitude_and_temporal_phase(
    magnitude: torch.Tensor,
    increments: torch.Tensor,
    initial_phase: torch.Tensor,
    *,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    phase = _phase_from_temporal_increments(increments, initial_phase)
    if phase.shape != magnitude.shape:
        raise RuntimeError("magnitude/phase geometry mismatch")
    spectrum = torch.polar(magnitude.to(torch.float32).clamp_min(0.0), phase.to(torch.float32))
    return _istft(spectrum, length=length, dtype=dtype)


def _candidate_mag_target_phase(
    target_spec: torch.Tensor,
    candidate_spec: torch.Tensor,
    *,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if target_spec.shape != candidate_spec.shape:
        raise RuntimeError("target/candidate STFT geometry mismatch")
    spectrum = torch.polar(candidate_spec.abs(), torch.angle(target_spec))
    return _istft(spectrum, length=length, dtype=dtype)


def _temporal_increment_alignment_score(
    target_spec: torch.Tensor,
    candidate_spec: torch.Tensor,
) -> float:
    target_inc = _temporal_phase_increments(target_spec)
    candidate_inc = _temporal_phase_increments(candidate_spec)
    delta = target_inc - candidate_inc
    weights = (
        target_spec[:, 1:].abs().square()
        + target_spec[:, :-1].abs().square()
    ).mul(0.5)
    score = (torch.cos(delta) * weights).sum() / weights.sum().clamp_min(EPSILON)
    return float(score)


def _temporal_increment_circular_mae(
    target_spec: torch.Tensor,
    candidate_spec: torch.Tensor,
) -> float:
    target_inc = _temporal_phase_increments(target_spec)
    candidate_inc = _temporal_phase_increments(candidate_spec)
    delta = torch.atan2(torch.sin(target_inc - candidate_inc), torch.cos(target_inc - candidate_inc))
    weights = (
        target_spec[:, 1:].abs().square()
        + target_spec[:, :-1].abs().square()
    ).mul(0.5)
    return float((delta.abs() * weights).sum() / weights.sum().clamp_min(EPSILON))


def run_candidate_magnitude_temporal_phase_increment(
    root: Path,
    *,
    utterance_ids: tuple[str, ...] = DEFAULT_UTTERANCE_IDS,
    scan_items: int = DEFAULT_SCAN_ITEMS,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root
        / "models"
        / "lykenox_identity"
        / "training"
        / "residual_statistics_source_v1"
        / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"candidate checkpoint missing: {checkpoint}")
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / OUTPUT_DIR_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    wanted = tuple(dict.fromkeys(utterance_ids))
    if not wanted:
        raise ValueError("at least one utterance id is required")
    candidate = _load_candidate(checkpoint)
    utterances = collect_owned_vocoder_utterances(
        root,
        split="val",
        max_items=max(scan_items, len(wanted)),
    )
    by_id = {utterance.utterance_id: utterance for utterance in utterances}
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise RuntimeError("requested held-out utterances not found: " + ", ".join(missing))

    items: list[dict[str, object]] = []
    with torch.no_grad():
        for utterance_id in wanted:
            utterance = by_id[utterance_id]
            frames = int(utterance.mel_frames)
            expected_samples = frames * HOP_LENGTH
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            if int(reference.numel()) != expected_samples:
                raise RuntimeError("held-out waveform length contract changed")

            target_residual, oracle_cepstrum, _ = extract_owned_real_residual(
                reference,
                frame_count=frames,
            )
            conditioning = extract_pitch_conditioning_v2(
                reference,
                frame_count=frames,
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                frame_length=int(PITCH_CONFIG["frame_length"]),
                min_f0_hz=float(PITCH_CONFIG["min_f0_hz"]),
                max_f0_hz=float(PITCH_CONFIG["max_f0_hz"]),
                anchor_periodicity_threshold=float(PITCH_CONFIG["voiced_periodicity_threshold"]),
                anchor_rms_fraction=float(PITCH_CONFIG["voiced_rms_fraction"]),
            )
            source_cepstrum, log_rms, source_periodicity = candidate(
                utterance.mel.unsqueeze(0).cpu(),
                conditioning.f0_track_hz.unsqueeze(0).cpu(),
                conditioning.energy_confidence.unsqueeze(0).cpu(),
                conditioning.periodic_strength.unsqueeze(0).cpu(),
            )
            candidate_residual = synthesize_residual_from_statistics(
                source_cepstrum,
                log_rms,
                source_periodicity,
                conditioning.f0_track_hz.unsqueeze(0).cpu(),
                seed=_utterance_seed(utterance_id, DEFAULT_SEED + 1800000),
            )
            if candidate_residual.ndim == 2:
                candidate_residual = candidate_residual[0]
            candidate_residual = candidate_residual.to(torch.float32).contiguous()
            if candidate_residual.shape != target_residual.shape:
                raise RuntimeError("candidate residual length differs from target residual")

            target_spec = _stft(target_residual)
            candidate_spec = _stft(candidate_residual)
            if target_spec.shape != candidate_spec.shape:
                raise RuntimeError("target/candidate STFT geometry mismatch")
            target_inc = _temporal_phase_increments(target_spec)
            candidate_inc = _temporal_phase_increments(candidate_spec)
            candidate_mag = candidate_spec.abs()
            length = int(target_residual.numel())

            target_dphase_candidate_anchor = _hybrid_from_magnitude_and_temporal_phase(
                candidate_mag,
                target_inc,
                torch.angle(candidate_spec[:, 0]),
                length=length,
                dtype=target_residual.dtype,
            )
            target_dphase_zero_anchor = _hybrid_from_magnitude_and_temporal_phase(
                candidate_mag,
                target_inc,
                torch.zeros_like(torch.angle(target_spec[:, 0])),
                length=length,
                dtype=target_residual.dtype,
            )
            candidate_dphase_target_anchor = _hybrid_from_magnitude_and_temporal_phase(
                candidate_mag,
                candidate_inc,
                torch.angle(target_spec[:, 0]),
                length=length,
                dtype=target_residual.dtype,
            )
            candidate_target_phase = _candidate_mag_target_phase(
                target_spec,
                candidate_spec,
                length=length,
                dtype=target_residual.dtype,
            )

            residuals = {
                "target_real_residual": target_residual,
                "candidate_statistics_residual": candidate_residual,
                "candidate_mag_target_phase_ceiling": candidate_target_phase,
                "candidate_mag_target_dphase_candidate_anchor": target_dphase_candidate_anchor,
                "candidate_mag_target_dphase_zero_anchor": target_dphase_zero_anchor,
                "candidate_mag_candidate_dphase_target_anchor": candidate_dphase_target_anchor,
            }
            renders = {
                key: render_time_varying_minimum_phase(
                    residual.unsqueeze(0),
                    oracle_cepstrum.unsqueeze(0),
                    hop_length=HOP_LENGTH,
                    n_fft=N_FFT,
                ).squeeze(0)
                for key, residual in residuals.items()
            }
            for render in renders.values():
                if render.shape != reference.shape:
                    raise RuntimeError("temporal-phase render/reference shape mismatch")
                if not bool(torch.isfinite(render).all()):
                    raise RuntimeError("temporal-phase render contains non-finite values")

            paths: dict[str, str] = {}
            reference_path = output_dir / f"{utterance_id}__reference.wav"
            _write(reference_path, reference)
            paths["reference"] = str(reference_path)
            labels = {
                "target_real_residual": "identity_roundtrip_ceiling",
                "candidate_statistics_residual": "candidate_statistics_render",
                "candidate_mag_target_phase_ceiling": "candidate_mag_target_phase_ceiling",
                "candidate_mag_target_dphase_candidate_anchor": "candidate_mag_target_dphase_candidate_anchor_render",
                "candidate_mag_target_dphase_zero_anchor": "candidate_mag_target_dphase_zero_anchor_render",
                "candidate_mag_candidate_dphase_target_anchor": "candidate_mag_candidate_dphase_target_anchor_render",
            }
            for key, label in labels.items():
                path = output_dir / f"{utterance_id}__{label}.wav"
                _write(path, renders[key])
                paths[label] = str(path)

            items.append(
                {
                    "utterance_id": utterance_id,
                    "temporal_phase_increment_alignment_score_target_vs_candidate": (
                        _temporal_increment_alignment_score(target_spec, candidate_spec)
                    ),
                    "temporal_phase_increment_circular_mae_radians": (
                        _temporal_increment_circular_mae(target_spec, candidate_spec)
                    ),
                    "paths": paths,
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_temporal_phase_increment_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "split": "val",
        "utterance_ids": list(wanted),
        "checkpoint": str(checkpoint),
        "training_executed": False,
        "optimizer_created": False,
        "renderer_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "predicted_duration_modified": False,
        "third_party_model_used": False,
        "metrics_can_accept_product_quality": False,
        "known_listening_evidence": (
            "speech_0021 candidate magnitude plus target full residual phase sounds correct; "
            "candidate-magnitude Griffin-Lim is intelligible but mildly robotic"
        ),
        "listening_interpretation": {
            "target_dphase_candidate_anchor_good_and_zero_anchor_good": (
                "temporal_phase_increments_are_sufficient; absolute_phase_anchor_is_not_required"
            ),
            "target_dphase_candidate_anchor_good_zero_anchor_bad": (
                "temporal_phase_increments_plus_initial_per_bin_phase_anchor_are_required"
            ),
            "both_target_dphase_hybrids_bad_but_full_target_phase_ceiling_good": (
                "temporal_phase_increments_alone_are_insufficient; within_frame_frequency_phase_structure_or_group_delay_is_required"
            ),
            "candidate_dphase_target_anchor_bad": (
                "candidate_temporal_phase_evolution_remains_invalid_even_with_target_initial_phase"
            ),
        },
        "items": items,
        "next_action": (
            "listen_first_to_candidate_mag_target_dphase_candidate_anchor_render and "
            "candidate_mag_target_dphase_zero_anchor_render against candidate_mag_target_phase_ceiling; "
            "do_not_train_or_modify_renderer"
        ),
    }
    _atomic_json(output_dir / "candidate_magnitude_temporal_phase_increment_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--utterance-id", action="append", dest="utterance_ids", default=None)
    parser.add_argument("--scan-items", type=int, default=DEFAULT_SCAN_ITEMS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    requested = tuple(args.utterance_ids) if args.utterance_ids else DEFAULT_UTTERANCE_IDS
    print(
        json.dumps(
            run_candidate_magnitude_temporal_phase_increment(
                args.root,
                utterance_ids=requested,
                scan_items=args.scan_items,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
