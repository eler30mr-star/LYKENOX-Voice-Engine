"""Held-out render for the controlled Continuous Source V2 conditioning-contract retrain.

The candidate and historical baseline use the exact same model architecture and fixed minimum-phase
renderer. The only intended training-path difference is pitch-conditioning V2 versus legacy
F0/voiced/periodicity. The historical V2 baseline checkpoint is rendered side by side from legacy
conditioning. The candidate checkpoint is rendered from f0_track_hz / energy_confidence /
periodic_strength. Identity roundtrip and reference are written beside both.

No gain normalization, EQ, denoise, external model/weight/service or duration modification is used.
Metrics may reject but cannot accept product quality. Policy: LYX-POL-001.
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

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v2 import (
    CONTINUOUS_SOURCE_ARCHITECTURE_V2,
    HOP_LENGTH,
    LykenoxContinuousResidualSourceV2,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import (
    PITCH_CONDITIONING_V2,
    extract_pitch_conditioning_v2,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import _ola_vectors
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v2 import (
    CHECKPOINT_SCHEMA_VERSION as BASE_V2_CHECKPOINT_SCHEMA_VERSION,
    POLICY_ID,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_v2_pitch_conditioning_v2 import (
    CHECKPOINT_SCHEMA_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_time_varying_minimum_phase,
)


EVALUATION_VERSION = "owned-continuous-source-v2-pitch-conditioning-v2-heldout-v1"
OUTPUT_DIR_NAME = "vocoder_minimum_phase_continuous_residual_source_v2_pitch_conditioning_v2"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().to(torch.float32).numpy(), SAMPLE_RATE, subtype="FLOAT")


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.to(torch.float64).square().mean().clamp_min(1.0e-30)))


def _load_model(path: Path, *, checkpoint_schema: str, conditioning_contract: str | None) -> tuple[LykenoxContinuousResidualSourceV2, dict[str, object]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("checkpoint_schema_version") != checkpoint_schema:
        raise RuntimeError(f"checkpoint schema mismatch: {path}")
    if payload.get("architecture") != CONTINUOUS_SOURCE_ARCHITECTURE_V2:
        raise RuntimeError(f"architecture mismatch: {path}")
    if conditioning_contract is not None and payload.get("conditioning_contract") != conditioning_contract:
        raise RuntimeError(f"conditioning contract mismatch: {path}")
    model = LykenoxContinuousResidualSourceV2().cpu()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def render_heldout_v2_pitch_conditioning_v2(
    root: Path,
    *,
    heldout_items: int = 3,
    checkpoint: Path | None = None,
    baseline_checkpoint: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    if heldout_items < 1 or heldout_items > 3:
        raise ValueError("heldout_items must be in [1,3]")
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2_pitch_conditioning_v2" / "best.pt"
    )
    baseline_checkpoint = (
        Path(baseline_checkpoint).resolve()
        if baseline_checkpoint is not None
        else root / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    if not baseline_checkpoint.exists():
        raise FileNotFoundError(str(baseline_checkpoint))

    candidate_model, candidate_payload = _load_model(
        checkpoint,
        checkpoint_schema=CHECKPOINT_SCHEMA_VERSION,
        conditioning_contract=PITCH_CONDITIONING_V2,
    )
    baseline_model, baseline_payload = _load_model(
        baseline_checkpoint,
        checkpoint_schema=BASE_V2_CHECKPOINT_SCHEMA_VERSION,
        conditioning_contract=None,
    )

    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    utterances = collect_owned_vocoder_utterances(root, "val", max_items=heldout_items)
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            expected_samples = frame_count * HOP_LENGTH
            target_residual, oracle_cepstrum, _ = extract_owned_real_residual(
                utterance.waveform.cpu(), frame_count=frame_count
            )
            conditioning = extract_pitch_conditioning_v2(
                utterance.waveform.cpu().to(torch.float32),
                frame_count=frame_count,
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                frame_length=int(PITCH_CONFIG["frame_length"]),
                min_f0_hz=float(PITCH_CONFIG["min_f0_hz"]),
                max_f0_hz=float(PITCH_CONFIG["max_f0_hz"]),
                anchor_periodicity_threshold=float(PITCH_CONFIG["voiced_periodicity_threshold"]),
                anchor_rms_fraction=float(PITCH_CONFIG["voiced_rms_fraction"]),
            )

            candidate_vectors, candidate_log_rms = candidate_model.forward_with_log_rms(
                utterance.mel.unsqueeze(0).cpu(),
                conditioning.f0_track_hz.unsqueeze(0).cpu(),
                conditioning.energy_confidence.unsqueeze(0).cpu(),
                conditioning.periodic_strength.unsqueeze(0).cpu(),
                teacher_vectors=None,
                teacher_forcing_ratio=0.0,
            )
            baseline_vectors, _ = baseline_model.forward_with_log_rms(
                utterance.mel.unsqueeze(0).cpu(),
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
                teacher_vectors=None,
                teacher_forcing_ratio=0.0,
            )

            candidate_residual = _ola_vectors(candidate_vectors, output_samples=expected_samples).squeeze(0)
            baseline_residual = _ola_vectors(baseline_vectors, output_samples=expected_samples).squeeze(0)
            candidate = render_time_varying_minimum_phase(
                candidate_residual.unsqueeze(0),
                oracle_cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            baseline = render_time_varying_minimum_phase(
                baseline_residual.unsqueeze(0),
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
            if not (candidate.shape == baseline.shape == identity.shape == reference.shape):
                raise RuntimeError("controlled V2 conditioning render geometry mismatch")

            stem = utterance.utterance_id
            candidate_path = output_dir / f"{stem}__v2_pitch_conditioning_v2.wav"
            baseline_path = output_dir / f"{stem}__v2_baseline_source.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write(candidate_path, candidate)
            _write(baseline_path, baseline)
            _write(identity_path, identity)
            _write(reference_path, reference)

            candidate_rms = _rms(candidate)
            baseline_rms = _rms(baseline)
            reference_rms = _rms(reference)
            items.append(
                {
                    "utterance_id": stem,
                    "candidate_rms_ratio": candidate_rms / max(reference_rms, 1.0e-12),
                    "baseline_rms_ratio": baseline_rms / max(reference_rms, 1.0e-12),
                    "mean_candidate_vector_log_rms": float(candidate_log_rms.mean()),
                    "v2_pitch_conditioning_v2": str(candidate_path),
                    "v2_baseline_source": str(baseline_path),
                    "identity_roundtrip_ceiling": str(identity_path),
                    "reference": str(reference_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_v2_pitch_conditioning_v2_controlled_listening",
        "evaluation_version": EVALUATION_VERSION,
        "policy_id": POLICY_ID,
        "architecture": CONTINUOUS_SOURCE_ARCHITECTURE_V2,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "renderer_version": RENDERER_VERSION,
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_update": int(candidate_payload.get("update", -1)),
        "historical_v2_baseline_checkpoint": str(baseline_checkpoint),
        "historical_v2_checkpoint_update": int(baseline_payload.get("update", -1)),
        "architecture_difference_between_candidate_and_baseline": False,
        "candidate_conditioning_slots": ["f0_track_hz", "energy_confidence", "periodic_strength"],
        "baseline_conditioning_slots": ["f0_hz", "voiced", "periodicity"],
        "source_generation_teacher_forcing_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_candidate_vs_historical_v2_baseline_vs_identity_ceiling_and_reject_if_transition_artifacts_do_not_materially_decrease",
    }
    _atomic_json(output_dir / "v2_pitch_conditioning_v2_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(render_heldout_v2_pitch_conditioning_v2(
        args.root,
        heldout_items=args.heldout_items,
        checkpoint=args.checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
