"""Phase-continuous held-out renderer for the trained pitch-synchronous cycle source.

The V1 learned cycle source produced the strongest learned held-out voice so far, but listening
reported a residual robotic whistle/chirp.  V1 decoded each canonical cycle independently with
linear interpolation and assigned it directly into ``cycle_waveform[left:right]``.  That leaves
cycle-edge continuity completely unconstrained and repeats any edge jump at F0 rate.

This renderer fixes that synthesis defect without retraining or post-hoc enhancement:

1. canonical cycles are interpreted as periodic functions through their normalized Fourier series;
2. only harmonics representable by the requested physical period are synthesized (no resampling
   images above that period's Nyquist limit);
3. during cycle i, the periodic shape is continuously morphed from predicted cycle i toward cycle
   i+1.  Therefore the right edge approaches the same periodic phase that starts cycle i+1 instead
   of hard-splicing two unrelated decoded endpoints.

The trained V1 checkpoint, V2 unvoiced/transitional baseline, Step-3f oracle envelope, and fixed
minimum-phase renderer remain unchanged.  No optimizer, training, global gain normalization, EQ,
denoise, codebook, third-party model/weight, or remote service is used. Policy: LYX-POL-001.
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


EVALUATION_VERSION = "owned-pitch-synchronous-cycle-source-phase-continuous-heldout-v2"


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


def decode_cycle_v1_linear(canonical: torch.Tensor, samples: int) -> torch.Tensor:
    """Historical V1 decoder retained only for same-checkpoint audible comparison."""
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


def decode_cycle_periodic_bandlimited(canonical: torch.Tensor, samples: int) -> torch.Tensor:
    """Evaluate one canonical cycle as a periodic, band-limited Fourier series.

    ``norm='forward'`` makes the rFFT coefficients Fourier-series amplitudes independent of the
    requested output length.  Source/output Nyquist bins are deliberately excluded because a bin
    that is Nyquist for one discrete cycle length is generally not Nyquist for another; excluding
    that single edge bin avoids amplitude-doubling ambiguity while preserving every safely
    representable positive/negative harmonic pair.
    """
    if canonical.ndim != 1 or canonical.numel() != CYCLE_PHASE_BINS:
        raise ValueError("canonical cycle geometry changed")
    if samples < 2:
        raise ValueError("cycle period must contain at least two samples")
    spectrum = torch.fft.rfft(canonical.to(torch.float32), norm="forward")
    output_bins = int(samples) // 2 + 1
    resized = torch.zeros(output_bins, dtype=spectrum.dtype, device=spectrum.device)
    max_harmonic = min(CYCLE_PHASE_BINS // 2 - 1, (int(samples) - 1) // 2)
    resized[: max_harmonic + 1] = spectrum[: max_harmonic + 1]
    decoded = torch.fft.irfft(resized, n=int(samples), norm="forward")
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("periodic cycle decoder produced non-finite values")
    return decoded.to(torch.float32).contiguous()


def synthesize_phase_continuous_cycles(
    predicted_cycles: torch.Tensor,
    boundaries: list[tuple[int, int, int, float]],
    *,
    output_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize continuously evolving periodic cycles on the physical F0 grid."""
    if predicted_cycles.ndim != 2 or predicted_cycles.shape[-1] != CYCLE_PHASE_BINS:
        raise ValueError("predicted_cycles must be [cycles, phase_bins]")
    if int(predicted_cycles.shape[0]) != len(boundaries):
        raise ValueError("predicted cycle count and boundary count differ")
    waveform = torch.zeros(output_samples, dtype=torch.float32)
    coverage = torch.zeros(output_samples, dtype=torch.float32)
    if not boundaries:
        return waveform, coverage

    for index, (left, right, _, _) in enumerate(boundaries):
        samples = int(right - left)
        current = decode_cycle_periodic_bandlimited(predicted_cycles[index], samples)
        next_index = min(index + 1, int(predicted_cycles.shape[0]) - 1)
        following = decode_cycle_periodic_bandlimited(predicted_cycles[next_index], samples)
        phase = torch.arange(samples, dtype=torch.float32) / float(samples)
        decoded = current * (1.0 - phase) + following * phase
        waveform[left:right] = decoded
        coverage[left:right] = 1.0
    return waveform, coverage


def synthesize_v1_hard_splice_cycles(
    predicted_cycles: torch.Tensor,
    boundaries: list[tuple[int, int, int, float]],
    *,
    output_samples: int,
) -> torch.Tensor:
    waveform = torch.zeros(output_samples, dtype=torch.float32)
    for index, (left, right, _, _) in enumerate(boundaries):
        waveform[left:right] = decode_cycle_v1_linear(predicted_cycles[index], right - left)
    return waveform


def render_heldout_phase_continuous(
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
        / "vocoder_minimum_phase_pitch_synchronous_cycle_source_v2_phase_continuous"
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
            phase_cycle_waveform = torch.zeros(expected_samples, dtype=torch.float32)
            v1_cycle_waveform = torch.zeros(expected_samples, dtype=torch.float32)
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
                phase_cycle_waveform, coverage = synthesize_phase_continuous_cycles(
                    predicted_cycles,
                    boundaries,
                    output_samples=expected_samples,
                )
                v1_cycle_waveform = synthesize_v1_hard_splice_cycles(
                    predicted_cycles,
                    boundaries,
                    output_samples=expected_samples,
                )

            sample_voiced = fixed_linear_frame_to_sample(
                utterance.voiced.unsqueeze(0).to(torch.float32), hop_length=HOP_LENGTH
            ).squeeze(0)
            sample_periodicity = fixed_linear_frame_to_sample(
                utterance.periodicity.unsqueeze(0).to(torch.float32), hop_length=HOP_LENGTH
            ).squeeze(0)
            gate = (
                coverage
                * sample_voiced.clamp(0.0, 1.0)
                * sample_periodicity.clamp(0.0, 1.0)
            ).clamp(0.0, 1.0)

            phase_residual = v2_residual * (1.0 - gate) + phase_cycle_waveform * gate
            v1_residual = v2_residual * (1.0 - gate) + v1_cycle_waveform * gate
            prediction = render_time_varying_minimum_phase(
                phase_residual.unsqueeze(0), oracle_cepstrum.unsqueeze(0), hop_length=HOP_LENGTH, n_fft=N_FFT
            ).squeeze(0)
            v1_prediction = render_time_varying_minimum_phase(
                v1_residual.unsqueeze(0), oracle_cepstrum.unsqueeze(0), hop_length=HOP_LENGTH, n_fft=N_FFT
            ).squeeze(0)
            v2_prediction = render_time_varying_minimum_phase(
                v2_residual.unsqueeze(0), oracle_cepstrum.unsqueeze(0), hop_length=HOP_LENGTH, n_fft=N_FFT
            ).squeeze(0)
            identity = render_time_varying_minimum_phase(
                target_residual.unsqueeze(0), oracle_cepstrum.unsqueeze(0), hop_length=HOP_LENGTH, n_fft=N_FFT
            ).squeeze(0)
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            for value in (prediction, v1_prediction, v2_prediction, identity):
                if value.shape != reference.shape:
                    raise RuntimeError("phase-continuous heldout output length mismatch")

            stem = utterance.utterance_id
            phase_path = output_dir / f"{stem}__phase_continuous_cycle_source_v2.wav"
            v1_path = output_dir / f"{stem}__pitch_sync_v1_hard_splice_baseline.wav"
            v2_path = output_dir / f"{stem}__v2_baseline_source.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            residual_path = output_dir / f"{stem}__phase_continuous_hybrid_residual.wav"
            _write(phase_path, prediction)
            _write(v1_path, v1_prediction)
            _write(v2_path, v2_prediction)
            _write(identity_path, identity)
            _write(reference_path, reference)
            _write(residual_path, phase_residual)
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "cycle_count": len(boundaries),
                    "prediction_rms": _rms(prediction),
                    "v1_hard_splice_rms": _rms(v1_prediction),
                    "v2_baseline_rms": _rms(v2_prediction),
                    "reference_rms": _rms(reference),
                    "prediction_to_reference_rms_ratio": _rms(prediction) / max(_rms(reference), 1.0e-12),
                    "phase_continuous_cycle_source_v2": str(phase_path),
                    "pitch_sync_v1_hard_splice_baseline": str(v1_path),
                    "v2_baseline_source": str(v2_path),
                    "identity_roundtrip_ceiling": str(identity_path),
                    "reference": str(reference_path),
                    "phase_continuous_hybrid_residual": str(residual_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_phase_continuous_cycle_source_v2_listening",
        "evaluation_version": EVALUATION_VERSION,
        "policy_id": POLICY_ID,
        "source_checkpoint_architecture": PITCH_SYNCHRONOUS_SOURCE_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_training_update": int(cycle_payload.get("update", -1)),
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "cycle_decoder": "periodic_bandlimited_fourier_with_within_cycle_next_shape_morph",
        "hard_cycle_splice_removed": True,
        "linear_cycle_resampling_removed": True,
        "codebook_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_for_removal_of_residual_pitch_synchronous_whistle_without_regressing_voice_quality",
    }
    _atomic_json(output_dir / "phase_continuous_cycle_source_v2_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--v2-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(render_heldout_phase_continuous(
        args.root,
        heldout_items=args.heldout_items,
        checkpoint=args.checkpoint,
        v2_checkpoint=args.v2_checkpoint,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
