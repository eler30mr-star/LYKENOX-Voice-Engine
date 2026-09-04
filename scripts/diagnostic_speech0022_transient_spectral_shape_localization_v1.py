"""No-training temporal localization gate for speech_0022 spectral-shape artifact.

Listening established that under full target residual phase:
- candidate spectral shape + target frame level makes the low grinder-like artifact clearer;
- target spectral shape + candidate frame level sounds good and removes the chicken-like onset artifact.

Therefore spectral shape, not broadband frame level, is the primary remaining magnitude failure for
speech_0022. The user also identified a real chicken cry near the utterance onset and hypothesized
that the candidate attempts to imitate it and spreads a distorted grinder-like texture through later
speech. This diagnostic tests that hypothesis without training.

Candidate frame level and full target phase are frozen. Only spectral shape is swapped in time:
- target shape in the first 0.5, 1.0, 1.5, or 2.0 seconds, candidate shape elsewhere;
- candidate shape in the first 1.0 second, target shape afterwards;
- full candidate-shape baseline and full target-shape ceiling/control.

No optimizer, checkpoint write, renderer modification, denoise, EQ, duration change, third-party
model/service, or product-path normalization is used. AUDITION files use one common monitor gain.
Policy: LYX-POL-001.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostic_residual_phase_magnitude_forensic_v1 import _load_candidate
from scripts.diagnostic_target_phase_magnitude_level_shape_v1 import (
    _common_audition_gain,
    _compose_magnitude,
    _decompose_log_magnitude,
    _residual_from_magnitude_phase,
    _stft,
    _write,
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


DIAGNOSTIC_VERSION = "owned-speech0022-transient-spectral-shape-localization-v1"
POLICY_ID = "LYX-POL-001"
OUTPUT_DIR_NAME = "vocoder_speech0022_transient_spectral_shape_localization_v1"
TARGET_UTTERANCE_ID = "speech_0022_ba721f6129b9_seg_005"
PREFIX_SECONDS = (0.5, 1.0, 1.5, 2.0)
COMPLEMENT_SECONDS = 1.0
DEFAULT_SCAN_ITEMS = 256


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _mixed_shape(
    target_shape: torch.Tensor,
    candidate_shape: torch.Tensor,
    target_frame_mask: torch.Tensor,
) -> torch.Tensor:
    if target_shape.shape != candidate_shape.shape:
        raise RuntimeError("target/candidate shape geometry mismatch")
    if target_frame_mask.ndim != 1 or int(target_frame_mask.numel()) != int(target_shape.shape[1]):
        raise RuntimeError("time mask geometry mismatch")
    return torch.where(target_frame_mask.unsqueeze(0), target_shape, candidate_shape)


def _shape_error_by_frame(target_shape: torch.Tensor, candidate_shape: torch.Tensor) -> torch.Tensor:
    return (target_shape - candidate_shape).abs().mean(dim=0)


def run_transient_spectral_shape_localization(
    root: Path,
    *,
    scan_items: int = DEFAULT_SCAN_ITEMS,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root / "models" / "lykenox_identity" / "training" / "residual_statistics_source_v1" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"candidate checkpoint missing: {checkpoint}")
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    )
    raw_dir = output_dir / "raw"
    audition_dir = output_dir / "audition"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audition_dir.mkdir(parents=True, exist_ok=True)

    candidate = _load_candidate(checkpoint)
    utterances = collect_owned_vocoder_utterances(root, split="val", max_items=scan_items)
    by_id = {utterance.utterance_id: utterance for utterance in utterances}
    if TARGET_UTTERANCE_ID not in by_id:
        raise RuntimeError(f"target held-out utterance not found: {TARGET_UTTERANCE_ID}")
    utterance = by_id[TARGET_UTTERANCE_ID]

    with torch.no_grad():
        frames = int(utterance.mel_frames)
        expected_samples = frames * HOP_LENGTH
        reference = utterance.waveform.cpu().to(torch.float32).contiguous()
        if int(reference.numel()) != expected_samples:
            raise RuntimeError("held-out waveform length contract changed")

        target_residual, oracle_cepstrum, _ = extract_owned_real_residual(reference, frame_count=frames)
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
            seed=_utterance_seed(TARGET_UTTERANCE_ID, DEFAULT_SEED + 1800000),
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
        target_mag = target_spec.abs()
        candidate_mag = candidate_spec.abs()
        target_phase = torch.angle(target_spec)
        target_level, target_shape = _decompose_log_magnitude(target_mag)
        candidate_level, candidate_shape = _decompose_log_magnitude(candidate_mag)

        stft_frames = int(target_shape.shape[1])
        frame_times = torch.arange(stft_frames, dtype=torch.float32) * (float(HOP_LENGTH) / float(SAMPLE_RATE))
        magnitudes: dict[str, torch.Tensor] = {
            "candidate_shape_full": _compose_magnitude(candidate_level, candidate_shape),
            "target_shape_full": _compose_magnitude(candidate_level, target_shape),
            "identity_target_full": target_mag,
        }
        for seconds in PREFIX_SECONDS:
            mask = frame_times < float(seconds)
            shape = _mixed_shape(target_shape, candidate_shape, mask)
            label = f"target_shape_first_{str(seconds).replace('.', 'p')}s"
            magnitudes[label] = _compose_magnitude(candidate_level, shape)

        after_mask = frame_times >= float(COMPLEMENT_SECONDS)
        after_shape = _mixed_shape(target_shape, candidate_shape, after_mask)
        magnitudes["target_shape_after_1p0s"] = _compose_magnitude(candidate_level, after_shape)

        residuals = {
            key: _residual_from_magnitude_phase(
                magnitude,
                target_phase,
                length=int(target_residual.numel()),
                dtype=target_residual.dtype,
            )
            for key, magnitude in magnitudes.items()
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
        renders["reference"] = reference
        for value in renders.values():
            if value.shape != reference.shape or not bool(torch.isfinite(value).all()):
                raise RuntimeError("transient spectral-shape render violated waveform contract")

        labels = {
            "reference": "reference",
            "candidate_shape_full": "candidate_shape_candidate_level_target_phase_baseline",
            "target_shape_full": "target_shape_candidate_level_full_render",
            "identity_target_full": "identity_roundtrip_ceiling",
            "target_shape_first_0p5s": "target_shape_first_0p5s_candidate_else_render",
            "target_shape_first_1p0s": "target_shape_first_1p0s_candidate_else_render",
            "target_shape_first_1p5s": "target_shape_first_1p5s_candidate_else_render",
            "target_shape_first_2p0s": "target_shape_first_2p0s_candidate_else_render",
            "target_shape_after_1p0s": "candidate_shape_first_1p0s_target_shape_after_render",
        }

        raw_paths: dict[str, str] = {}
        for key, label in labels.items():
            path = raw_dir / f"{TARGET_UTTERANCE_ID}__{label}.wav"
            _write(path, renders[key])
            raw_paths[label] = str(path)

        audition_gain = _common_audition_gain(renders, reference)
        audition_paths: dict[str, str] = {}
        for key, label in labels.items():
            path = audition_dir / f"{TARGET_UTTERANCE_ID}__{label}__AUDITION.wav"
            _write(path, renders[key] * audition_gain)
            audition_paths[label] = str(path)

        error_by_frame = _shape_error_by_frame(target_shape, candidate_shape)
        top_count = min(12, int(error_by_frame.numel()))
        top_values, top_indices = torch.topk(error_by_frame, k=top_count)
        top_error_frames = [
            {
                "frame": int(index),
                "time_seconds": float(frame_times[index]),
                "spectral_shape_log_l1": float(value),
            }
            for value, index in zip(top_values, top_indices)
        ]
        prefix_error: dict[str, float] = {}
        for seconds in PREFIX_SECONDS:
            mask = frame_times < float(seconds)
            prefix_error[f"first_{str(seconds).replace('.', 'p')}s"] = float(error_by_frame[mask].mean())
        suffix_mask = frame_times >= float(COMPLEMENT_SECONDS)
        suffix_error = float(error_by_frame[suffix_mask].mean())

    report: dict[str, object] = {
        "status": "ready_for_speech0022_transient_spectral_shape_localization_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "split": "val",
        "utterance_id": TARGET_UTTERANCE_ID,
        "checkpoint": str(checkpoint),
        "prefix_seconds": list(PREFIX_SECONDS),
        "complement_seconds": COMPLEMENT_SECONDS,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "renderer_modified": False,
        "product_posthoc_gain_normalization_used": False,
        "audition_monitor_gain_used": True,
        "audition_monitor_gain_common_within_utterance": True,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "predicted_duration_modified": False,
        "third_party_model_used": False,
        "metrics_can_accept_product_quality": False,
        "known_listening_evidence": (
            "speech_0022 candidate spectral shape plus target frame level makes the grinder artifact clearer; "
            "target spectral shape plus candidate frame level sounds good and removes the chicken-like artifact; "
            "reference and identity roundtrip are clean"
        ),
        "hypothesis_under_test": (
            "a real chicken-cry transient near utterance onset is misrepresented by candidate spectral shape "
            "and may contaminate later frames with a grinder-like texture"
        ),
        "spectral_shape_error": {
            "mean_all_frames": float(error_by_frame.mean()),
            "prefix_mean": prefix_error,
            "mean_after_1p0s": suffix_error,
            "top_error_frames": top_error_frames,
        },
        "audition_gain_linear": audition_gain,
        "audition_gain_db": 20.0 * math.log10(max(audition_gain, 1.0e-12)),
        "raw_paths": raw_paths,
        "audition_paths": audition_paths,
        "listening_interpretation": {
            "short_prefix_replacement_cleans_later_audio": (
                "onset_transient_is_primary_local_spectral_shape_contamination_source"
            ),
            "only_full_target_shape_is_clean": (
                "spectral_shape_failure_is_distributed_across_utterance_not_only_onset_transient"
            ),
            "target_shape_after_1s_cleans_later_audio": (
                "later_candidate_spectral_shape_frames_drive_persistent_artifact; onset_is_not_sufficient_explanation"
            ),
        },
        "next_action": (
            "listen to speech_0022 AUDITION baseline, prefix spectral-shape swaps, after-1s complement, "
            "full target-shape control, identity and reference; do not train"
        ),
    }
    _atomic_json(output_dir / "speech0022_transient_spectral_shape_localization_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--scan-items", type=int, default=DEFAULT_SCAN_ITEMS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_transient_spectral_shape_localization(
        args.root,
        scan_items=args.scan_items,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
