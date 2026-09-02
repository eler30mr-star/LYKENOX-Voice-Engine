"""Render complete held-out audio from the pitch-synchronous real residual cycle source.

V2 remains the known-good learned baseline for non-periodic/transitional residual. In voiced periodic
regions the fixed-frame V2 residual is replaced continuously by real residual cycles predicted in F0
phase coordinates. The gate comes only from owned voiced/periodicity conditioning. The fixed
minimum-phase renderer is unchanged and no global/post-hoc gain normalization, EQ, or denoise is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v2 import (
    CONTINUOUS_SOURCE_ARCHITECTURE_V2,
    LykenoxContinuousResidualSourceV2,
)
from lykenox_voice_engine.models.vocoder.network_pitch_synchronous_residual_cycle_source_v1 import (
    CYCLE_PHASE_BINS,
    PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE,
    LykenoxPitchSynchronousResidualCycleSourceV1,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import _ola_vectors
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v2 import (
    CHECKPOINT_SCHEMA_VERSION as V2_CHECKPOINT_SCHEMA_VERSION,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    fixed_linear_frame_to_sample,
    render_time_varying_minimum_phase,
)
from lykenox_voice_engine.training.speech_vocoder_pitch_synchronous_cycle_source_train_v1 import (
    CHECKPOINT_SCHEMA_VERSION,
    POLICY_ID,
    conditioning_cycle_boundaries,
)


EVALUATION_VERSION = "owned-pitch-synchronous-cycle-source-heldout-listening-v1"


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


def _load_cycle_model(checkpoint: Path) -> tuple[LykenoxPitchSynchronousResidualCycleSourceV1, dict[str, object]]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("pitch-synchronous checkpoint schema mismatch")
    if payload.get("architecture") != PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE:
        raise RuntimeError("pitch-synchronous checkpoint architecture mismatch")
    model = LykenoxPitchSynchronousResidualCycleSourceV1().cpu()
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _load_v2_model(checkpoint: Path) -> LykenoxContinuousResidualSourceV2:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("checkpoint_schema_version") != V2_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("V2 checkpoint schema mismatch")
    if payload.get("architecture") != CONTINUOUS_SOURCE_ARCHITECTURE_V2:
        raise RuntimeError("V2 checkpoint architecture mismatch")
    model = LykenoxContinuousResidualSourceV2().cpu()
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _decode_cycle(canonical: torch.Tensor, samples: int) -> torch.Tensor:
    if canonical.ndim != 1 or canonical.numel() != CYCLE_PHASE_BINS:
        raise ValueError("canonical cycle geometry changed")
    if samples < 2:
        raise ValueError("cycle period must contain at least two samples")
    return F.interpolate(
        canonical.view(1, 1, -1),
        size=int(samples),
        mode="linear",
        align_corners=True,
    )[0, 0].contiguous()


def render_heldout_pitch_synchronous_cycle_source(
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
        / "vocoder_minimum_phase_pitch_synchronous_cycle_source_v1"
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
            cycle_waveform = torch.zeros(expected_samples, dtype=torch.float32)
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
                if int(predicted_cycles.shape[0]) != len(boundaries):
                    raise RuntimeError("predicted cycle count differs from conditioning boundaries")
                for index, (left, right, _, _) in enumerate(boundaries):
                    decoded = _decode_cycle(predicted_cycles[index], right - left)
                    cycle_waveform[left:right] = decoded
                    coverage[left:right] = 1.0

            sample_voiced = fixed_linear_frame_to_sample(
                utterance.voiced.unsqueeze(0).to(torch.float32), hop_length=HOP_LENGTH
            ).squeeze(0)
            sample_periodicity = fixed_linear_frame_to_sample(
                utterance.periodicity.unsqueeze(0).to(torch.float32), hop_length=HOP_LENGTH
            ).squeeze(0)
            gate = (coverage * sample_voiced.clamp(0.0, 1.0) * sample_periodicity.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            hybrid_residual = v2_residual * (1.0 - gate) + cycle_waveform * gate

            prediction = render_time_varying_minimum_phase(
                hybrid_residual.unsqueeze(0),
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
            if prediction.shape != reference.shape or identity.shape != reference.shape:
                raise RuntimeError("pitch-synchronous heldout output length mismatch")

            stem = utterance.utterance_id
            prediction_path = output_dir / f"{stem}__pitch_synchronous_cycle_source.wav"
            residual_path = output_dir / f"{stem}__pitch_synchronous_hybrid_residual.wav"
            cycle_path = output_dir / f"{stem}__pitch_synchronous_voiced_cycles.wav"
            v2_path = output_dir / f"{stem}__v2_baseline_source.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write(prediction_path, prediction)
            _write(residual_path, hybrid_residual)
            _write(cycle_path, cycle_waveform)
            _write(v2_path, v2_prediction)
            _write(identity_path, identity)
            _write(reference_path, reference)
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "cycle_count": len(boundaries),
                    "voiced_cycle_gate_fraction": float((gate > 0.5).to(torch.float32).mean()),
                    "prediction_rms": _rms(prediction),
                    "v2_baseline_rms": _rms(v2_prediction),
                    "reference_rms": _rms(reference),
                    "prediction_to_reference_rms_ratio": _rms(prediction) / max(_rms(reference), 1.0e-12),
                    "pitch_synchronous_cycle_source": str(prediction_path),
                    "pitch_synchronous_hybrid_residual": str(residual_path),
                    "pitch_synchronous_voiced_cycles": str(cycle_path),
                    "v2_baseline_source": str(v2_path),
                    "identity_roundtrip_ceiling": str(identity_path),
                    "reference": str(reference_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_pitch_synchronous_cycle_source_heldout_listening",
        "evaluation_version": EVALUATION_VERSION,
        "policy_id": POLICY_ID,
        "architecture": PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "checkpoint": str(checkpoint),
        "checkpoint_training_update": int(cycle_payload.get("update", -1)),
        "v2_baseline_checkpoint": str(v2_checkpoint),
        "voiced_source_representation": "real_step3f_residual_cycles_in_f0_phase_coordinates",
        "unvoiced_transition_source": "owned_continuous_residual_source_v2_baseline",
        "codebook_used": False,
        "teacher_forcing_used": False,
        "stochastic_innovation_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "compare_pitch_synchronous_source_vs_v2_baseline_vs_identity_ceiling_by_complete_listening",
    }
    _atomic_json(output_dir / "pitch_synchronous_cycle_source_v1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--v2-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(render_heldout_pitch_synchronous_cycle_source(
        args.root,
        heldout_items=args.heldout_items,
        checkpoint=args.checkpoint,
        v2_checkpoint=args.v2_checkpoint,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
