"""No-training phase-anchor/group-delay gate after temporal dphase isolation.

Listening established on speech_0021 that candidate STFT magnitude plus target temporal phase
increments and the candidate initial per-bin phase anchor sounds almost clean, while a zero anchor is
thinner/robotic/airier.  Therefore magnitude, the fixed renderer, and temporal phase increments are
held fixed here.  This diagnostic varies only the first-frame phase anchor across frequency.

It emits:
- candidate anchor baseline (known almost-good);
- full target anchor ceiling (same phase result as candidate magnitude + full target phase);
- target-low/candidate-high anchor with a smooth 3-4 kHz crossover;
- candidate-low/target-high anchor with the complementary crossover;
- candidate anchor corrected by a smooth low-dimensional target/candidate phase-offset field.

The band swaps localize whether the remaining wind/telephone texture is carried primarily by low- or
high-frequency cross-bin phase structure.  The smooth correction asks whether a low-dimensional
frequency-phase/group-delay field is enough, rather than requiring an unconstrained phase vector.

No training, optimizer, renderer modification, post-hoc gain normalization, EQ, denoise, duration
modification, third-party model/service, or new checkpoint is used. Policy: LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostic_residual_phase_magnitude_forensic_v1 import (
    DEFAULT_SCAN_ITEMS,
    DEFAULT_UTTERANCE_IDS,
    _load_candidate,
)
from scripts.diagnostic_candidate_magnitude_temporal_phase_increment_v1 import (
    _phase_from_temporal_increments,
    _stft,
    _istft,
    _temporal_phase_increments,
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


DIAGNOSTIC_VERSION = "owned-candidate-magnitude-phase-anchor-group-delay-v1"
POLICY_ID = "LYX-POL-001"
OUTPUT_DIR_NAME = "vocoder_candidate_magnitude_phase_anchor_group_delay_v1"
CROSSOVER_LOW_HZ = 3000.0
CROSSOVER_HIGH_HZ = 4000.0
SMOOTH_PHASE_OFFSET_BINS = 63
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


def _wrap_phase(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _circular_blend_phase(a: torch.Tensor, b: torch.Tensor, weight_b: torch.Tensor) -> torch.Tensor:
    """Blend phase a->b on the unit circle with scalar/bin weights in [0,1]."""
    if a.shape != b.shape or weight_b.shape != a.shape:
        raise ValueError("phase blend inputs must have matching [bins] geometry")
    wa = (1.0 - weight_b).clamp(0.0, 1.0)
    wb = weight_b.clamp(0.0, 1.0)
    real = wa * torch.cos(a) + wb * torch.cos(b)
    imag = wa * torch.sin(a) + wb * torch.sin(b)
    return torch.atan2(imag, real)


def _target_low_weight() -> torch.Tensor:
    frequencies = torch.fft.rfftfreq(N_FFT, d=1.0 / float(SAMPLE_RATE)).to(torch.float32)
    weight = torch.ones_like(frequencies)
    weight[frequencies >= CROSSOVER_HIGH_HZ] = 0.0
    transition = (frequencies > CROSSOVER_LOW_HZ) & (frequencies < CROSSOVER_HIGH_HZ)
    if bool(transition.any()):
        x = (frequencies[transition] - CROSSOVER_LOW_HZ) / (CROSSOVER_HIGH_HZ - CROSSOVER_LOW_HZ)
        weight[transition] = 0.5 * (1.0 + torch.cos(math.pi * x))
    return weight


def _smooth_phase_offset(target_anchor: torch.Tensor, candidate_anchor: torch.Tensor) -> torch.Tensor:
    """Circularly smooth target-candidate phase offset over frequency bins."""
    if target_anchor.shape != candidate_anchor.shape:
        raise ValueError("anchor shapes must match")
    if SMOOTH_PHASE_OFFSET_BINS < 3 or SMOOTH_PHASE_OFFSET_BINS % 2 == 0:
        raise RuntimeError("smooth phase-offset kernel must be odd and >=3")
    delta = _wrap_phase(target_anchor - candidate_anchor)
    real = torch.cos(delta).view(1, 1, -1)
    imag = torch.sin(delta).view(1, 1, -1)
    pad = SMOOTH_PHASE_OFFSET_BINS // 2
    real = F.pad(real, (pad, pad), mode="replicate")
    imag = F.pad(imag, (pad, pad), mode="replicate")
    kernel = torch.hann_window(
        SMOOTH_PHASE_OFFSET_BINS,
        periodic=False,
        dtype=real.dtype,
    ).view(1, 1, -1)
    kernel = kernel / kernel.sum().clamp_min(EPSILON)
    real_s = F.conv1d(real, kernel)[0, 0]
    imag_s = F.conv1d(imag, kernel)[0, 0]
    return torch.atan2(imag_s, real_s)


def _residual_from_anchor(
    candidate_magnitude: torch.Tensor,
    target_temporal_increments: torch.Tensor,
    anchor: torch.Tensor,
    *,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    phase = _phase_from_temporal_increments(target_temporal_increments, anchor)
    if phase.shape != candidate_magnitude.shape:
        raise RuntimeError("candidate magnitude / reconstructed phase geometry mismatch")
    spectrum = torch.polar(candidate_magnitude.to(torch.float32), phase.to(torch.float32))
    return _istft(spectrum, length=length, dtype=dtype)


def _anchor_alignment(target_anchor: torch.Tensor, candidate_anchor: torch.Tensor, weights: torch.Tensor) -> float:
    delta = _wrap_phase(target_anchor - candidate_anchor)
    return float((torch.cos(delta) * weights).sum() / weights.sum().clamp_min(EPSILON))


def run_candidate_magnitude_phase_anchor_group_delay(
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

    low_target_weight = _target_low_weight()
    high_target_weight = 1.0 - low_target_weight
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
            candidate_mag = candidate_spec.abs()
            target_anchor = torch.angle(target_spec[:, 0]).to(torch.float32)
            candidate_anchor = torch.angle(candidate_spec[:, 0]).to(torch.float32)
            if target_anchor.shape != low_target_weight.shape:
                raise RuntimeError("anchor/crossover frequency geometry mismatch")

            target_low_anchor = _circular_blend_phase(
                candidate_anchor,
                target_anchor,
                low_target_weight,
            )
            target_high_anchor = _circular_blend_phase(
                candidate_anchor,
                target_anchor,
                high_target_weight,
            )
            smooth_delta = _smooth_phase_offset(target_anchor, candidate_anchor)
            smooth_corrected_anchor = _wrap_phase(candidate_anchor + smooth_delta)

            anchors = {
                "candidate_anchor": candidate_anchor,
                "target_anchor": target_anchor,
                "target_low_candidate_high_anchor": target_low_anchor,
                "candidate_low_target_high_anchor": target_high_anchor,
                "smooth_group_delay_corrected_anchor": smooth_corrected_anchor,
            }
            residuals = {
                key: _residual_from_anchor(
                    candidate_mag,
                    target_inc,
                    anchor,
                    length=expected_samples,
                    dtype=target_residual.dtype,
                )
                for key, anchor in anchors.items()
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
            identity = render_time_varying_minimum_phase(
                target_residual.unsqueeze(0),
                oracle_cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            for value in (*renders.values(), identity):
                if value.shape != reference.shape:
                    raise RuntimeError("anchor forensic render/reference shape mismatch")
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError("anchor forensic render contains non-finite values")

            paths: dict[str, str] = {}
            reference_path = output_dir / f"{utterance_id}__reference.wav"
            identity_path = output_dir / f"{utterance_id}__identity_roundtrip_ceiling.wav"
            _write(reference_path, reference)
            _write(identity_path, identity)
            paths["reference"] = str(reference_path)
            paths["identity_roundtrip_ceiling"] = str(identity_path)

            labels = {
                "candidate_anchor": "candidate_mag_target_dphase_candidate_anchor_render",
                "target_anchor": "candidate_mag_target_dphase_target_anchor_ceiling",
                "target_low_candidate_high_anchor": "candidate_mag_target_dphase_target_low_anchor_render",
                "candidate_low_target_high_anchor": "candidate_mag_target_dphase_target_high_anchor_render",
                "smooth_group_delay_corrected_anchor": "candidate_mag_target_dphase_smooth_group_delay_anchor_render",
            }
            for key, label in labels.items():
                path = output_dir / f"{utterance_id}__{label}.wav"
                _write(path, renders[key])
                paths[label] = str(path)

            anchor_weights = target_spec[:, 0].abs().square().to(torch.float32)
            items.append(
                {
                    "utterance_id": utterance_id,
                    "candidate_anchor_alignment_score": _anchor_alignment(
                        target_anchor,
                        candidate_anchor,
                        anchor_weights,
                    ),
                    "smooth_corrected_anchor_alignment_score": _anchor_alignment(
                        target_anchor,
                        smooth_corrected_anchor,
                        anchor_weights,
                    ),
                    "paths": paths,
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_phase_anchor_group_delay_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "split": "val",
        "utterance_ids": list(wanted),
        "checkpoint": str(checkpoint),
        "crossover_low_hz": CROSSOVER_LOW_HZ,
        "crossover_high_hz": CROSSOVER_HIGH_HZ,
        "smooth_phase_offset_bins": SMOOTH_PHASE_OFFSET_BINS,
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
            "speech_0021 candidate magnitude + target temporal dphase + candidate anchor is almost very good "
            "with only slight airy/wind/telephone texture; zero anchor is thinner and more robotic"
        ),
        "listening_interpretation": {
            "target_high_anchor_cleans_remaining_artifact": (
                "remaining defect is primarily high-frequency anchor/group-delay structure"
            ),
            "target_low_anchor_cleans_remaining_artifact": (
                "remaining defect is primarily low-frequency anchor/group-delay structure"
            ),
            "smooth_group_delay_anchor_is_near_ceiling": (
                "a low-dimensional smooth frequency-phase/group-delay correction is sufficient"
            ),
            "only_full_target_anchor_is_clean": (
                "fine per-bin cross-frequency phase anchor structure remains required"
            ),
        },
        "items": items,
        "next_action": (
            "listen only to speech_0021 candidate-anchor baseline, target-low anchor, target-high anchor, "
            "smooth-group-delay anchor, and full-target-anchor ceiling; do not train or modify renderer"
        ),
    }
    _atomic_json(output_dir / "phase_anchor_group_delay_report.json", report)
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
            run_candidate_magnitude_phase_anchor_group_delay(
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
