"""Render complete held-out audio from the unified LYKENOX phase residual source V1.

The source is one jointly trained model.  No V2 fallback, pitch-sync checkpoint handoff, bridge,
codebook, stochastic innovation, external model/weight/service, gain normalization, EQ or denoise is
used.  The Step-3f oracle cepstrum is retained only to isolate source quality; the fixed minimum-phase
renderer is unchanged.  Metrics may reject but cannot accept product quality.
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

from lykenox_voice_engine.models.vocoder.network_minimum_phase_unified_phase_residual_source_v1 import (
    UNIFIED_PHASE_SOURCE_ARCHITECTURE,
    LykenoxUnifiedPhaseResidualSourceV1,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
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
from lykenox_voice_engine.training.speech_vocoder_unified_phase_residual_source_train_v1 import (
    CHECKPOINT_SCHEMA_VERSION,
    POLICY_ID,
    synthesize_unified_residual,
)


EVALUATION_VERSION = "owned-unified-phase-residual-source-heldout-v1"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _write(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().to(torch.float32).numpy(), SAMPLE_RATE, subtype="FLOAT")


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.to(torch.float64).square().mean().clamp_min(1.0e-30)))


def _load_model(checkpoint: Path) -> tuple[LykenoxUnifiedPhaseResidualSourceV1, dict[str, object]]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unified source checkpoint schema mismatch")
    if payload.get("architecture") != UNIFIED_PHASE_SOURCE_ARCHITECTURE:
        raise RuntimeError("unified source checkpoint architecture mismatch")
    model = LykenoxUnifiedPhaseResidualSourceV1().cpu()
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def render_heldout_unified_phase_source(
    root: Path,
    *,
    heldout_items: int = 3,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    if heldout_items < 1 or heldout_items > 3:
        raise ValueError("heldout_items must be in [1,3]")
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root / "models" / "lykenox_identity" / "training" / "unified_phase_residual_source_v1" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    model, payload = _load_model(checkpoint)
    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_unified_phase_residual_source_v1"
    )
    utterances = collect_owned_vocoder_utterances(root, "val", max_items=heldout_items)
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            target_residual, oracle_cepstrum, _ = extract_owned_real_residual(
                utterance.waveform.cpu(), frame_count=frame_count
            )
            harmonic_pairs, _, aperiodic_vectors, _ = model(
                utterance.mel.unsqueeze(0).cpu(),
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
            )
            residual, periodic_wave, aperiodic_wave, strength = synthesize_unified_residual(
                harmonic_pairs,
                aperiodic_vectors,
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
            )
            prediction = render_time_varying_minimum_phase(
                residual,
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
            residual = residual.squeeze(0)
            periodic_wave = periodic_wave.squeeze(0)
            aperiodic_wave = aperiodic_wave.squeeze(0)
            if prediction.shape != reference.shape or identity.shape != reference.shape or residual.shape != reference.shape:
                raise RuntimeError("unified heldout output length mismatch")

            stem = utterance.utterance_id
            prediction_path = output_dir / f"{stem}__unified_phase_residual_source_v1.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            residual_path = output_dir / f"{stem}__unified_residual.wav"
            periodic_path = output_dir / f"{stem}__periodic_coordinate.wav"
            aperiodic_path = output_dir / f"{stem}__aperiodic_coordinate.wav"
            _write(prediction_path, prediction)
            _write(identity_path, identity)
            _write(reference_path, reference)
            _write(residual_path, residual)
            _write(periodic_path, periodic_wave)
            _write(aperiodic_path, aperiodic_wave)
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "prediction_rms": _rms(prediction),
                    "reference_rms": _rms(reference),
                    "prediction_to_reference_rms_ratio": _rms(prediction) / max(_rms(reference), 1.0e-12),
                    "residual_to_target_rms_ratio": _rms(residual) / max(_rms(target_residual), 1.0e-12),
                    "mean_periodic_strength": float(strength.mean()),
                    "unified_phase_residual_source_v1": str(prediction_path),
                    "identity_roundtrip_ceiling": str(identity_path),
                    "reference": str(reference_path),
                    "unified_residual": str(residual_path),
                    "periodic_coordinate": str(periodic_path),
                    "aperiodic_coordinate": str(aperiodic_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_unified_phase_residual_source_v1_listening",
        "evaluation_version": EVALUATION_VERSION,
        "policy_id": POLICY_ID,
        "architecture": UNIFIED_PHASE_SOURCE_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "checkpoint": str(checkpoint),
        "checkpoint_training_update": int(payload.get("update", -1)),
        "single_model": True,
        "single_recurrent_state": True,
        "explicit_f0_phase": True,
        "periodic_aperiodic_coordinates_jointly_trained": True,
        "second_source_checkpoint_fallback_used": False,
        "source_handoff_or_bridge_used": False,
        "raw_source_crossfade_used": False,
        "codebook_used": False,
        "teacher_forcing_used": False,
        "stochastic_innovation_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_to_complete_unified_source_against_identity_ceiling_and_historical_pitch_sync_baseline",
    }
    _atomic_json(output_dir / "unified_phase_residual_source_v1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(render_heldout_unified_phase_source(
        args.root,
        heldout_items=args.heldout_items,
        checkpoint=args.checkpoint,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
