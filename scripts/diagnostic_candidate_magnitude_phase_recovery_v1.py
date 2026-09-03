"""No-training phase-recovery gate for the accepted residual magnitude forensic result.

Human listening established on speech_0021 that candidate STFT magnitude combined with the real
residual phase renders cleanly. This diagnostic therefore keeps the candidate magnitude fixed and
asks whether a coherent phase can be recovered from that magnitude alone with deterministic
Griffin-Lim projection. A target-magnitude Griffin-Lim control is emitted beside the known-good
candidate-magnitude + target-phase ceiling.

No training, optimizer, renderer modification, post-hoc gain normalization, EQ, denoise, duration
modification, third-party model, checkpoint other than the existing rejected statistics-source
checkpoint, or remote service is used. Policy: LYX-POL-001.
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


DIAGNOSTIC_VERSION = "owned-candidate-magnitude-phase-recovery-v1"
POLICY_ID = "LYX-POL-001"
OUTPUT_DIR_NAME = "vocoder_candidate_magnitude_phase_recovery_v1"
GRIFFIN_LIM_ITERATIONS = 64
GRIFFIN_LIM_SEED = 20260903
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
        raise RuntimeError("phase recovery violated exact-length contract")
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("phase recovery produced non-finite waveform")
    return waveform.to(torch.float32).contiguous()


def _griffin_lim_from_magnitude(
    magnitude: torch.Tensor,
    *,
    length: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
    if magnitude.ndim != 2:
        raise ValueError("magnitude must have shape [bins, frames]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    magnitude = magnitude.to(torch.float32).clamp_min(0.0).contiguous()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    phase = (
        torch.rand(magnitude.shape, generator=generator, dtype=magnitude.dtype)
        * (2.0 * math.pi)
        - math.pi
    )
    phase[0, :] = 0.0
    if N_FFT % 2 == 0:
        phase[-1, :] = 0.0
    spectrum = torch.polar(magnitude, phase)

    for _ in range(iterations):
        waveform = _istft(spectrum, length=length, dtype=magnitude.dtype)
        rebuilt = _stft(waveform)
        if rebuilt.shape != magnitude.shape:
            raise RuntimeError("Griffin-Lim STFT geometry changed")
        unit_phase = rebuilt / rebuilt.abs().clamp_min(EPSILON)
        spectrum = magnitude.to(unit_phase.dtype) * unit_phase

    return _istft(spectrum, length=length, dtype=magnitude.dtype)


def _candidate_mag_target_phase(
    target_residual: torch.Tensor,
    candidate_residual: torch.Tensor,
) -> torch.Tensor:
    target_spec = _stft(target_residual)
    candidate_spec = _stft(candidate_residual)
    if target_spec.shape != candidate_spec.shape:
        raise RuntimeError("target/candidate STFT geometry mismatch")
    hybrid = torch.polar(candidate_spec.abs(), torch.angle(target_spec))
    return _istft(hybrid, length=int(target_residual.numel()), dtype=target_residual.dtype)


def run_candidate_magnitude_phase_recovery(
    root: Path,
    *,
    utterance_ids: tuple[str, ...] = DEFAULT_UTTERANCE_IDS,
    scan_items: int = DEFAULT_SCAN_ITEMS,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    iterations: int = GRIFFIN_LIM_ITERATIONS,
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
            target_gl = _griffin_lim_from_magnitude(
                target_spec.abs(),
                length=expected_samples,
                iterations=iterations,
                seed=GRIFFIN_LIM_SEED + 1,
            )
            candidate_gl = _griffin_lim_from_magnitude(
                candidate_spec.abs(),
                length=expected_samples,
                iterations=iterations,
                seed=GRIFFIN_LIM_SEED + 2,
            )
            candidate_target_phase = _candidate_mag_target_phase(
                target_residual,
                candidate_residual,
            )

            residuals = {
                "target_real_residual": target_residual,
                "candidate_statistics_residual": candidate_residual,
                "candidate_mag_target_phase_residual": candidate_target_phase,
                "target_mag_griffinlim_residual": target_gl,
                "candidate_mag_griffinlim_residual": candidate_gl,
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
                    raise RuntimeError("phase-recovery render/reference shape mismatch")
                if not bool(torch.isfinite(render).all()):
                    raise RuntimeError("phase-recovery render contains non-finite values")

            paths: dict[str, str] = {}
            reference_path = output_dir / f"{utterance_id}__reference.wav"
            _write(reference_path, reference)
            paths["reference"] = str(reference_path)
            labels = {
                "target_real_residual": "identity_roundtrip_ceiling",
                "candidate_statistics_residual": "candidate_statistics_render",
                "candidate_mag_target_phase_residual": "candidate_mag_target_phase_ceiling",
                "target_mag_griffinlim_residual": "target_mag_griffinlim64_render",
                "candidate_mag_griffinlim_residual": "candidate_mag_griffinlim64_render",
            }
            for key, label in labels.items():
                path = output_dir / f"{utterance_id}__{label}.wav"
                _write(path, renders[key])
                paths[label] = str(path)

            items.append({"utterance_id": utterance_id, "paths": paths})

    report: dict[str, object] = {
        "status": "ready_for_candidate_magnitude_phase_recovery_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "split": "val",
        "utterance_ids": list(wanted),
        "checkpoint": str(checkpoint),
        "griffin_lim_iterations": int(iterations),
        "griffin_lim_seed": GRIFFIN_LIM_SEED,
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
            "speech_0021 candidate magnitude plus target residual phase sounds correct; "
            "candidate phase/temporal coherence is the isolated primary failure"
        ),
        "listening_interpretation": {
            "candidate_griffinlim_good": (
                "phase_can_be_recovered_from_predicted_magnitude_without_a_phase_model"
            ),
            "target_griffinlim_good_candidate_griffinlim_bad": (
                "candidate_magnitude_is_not_sufficient_for_phase_recovery_despite_target_phase_ceiling"
            ),
            "both_griffinlim_bad": (
                "griffin_lim_is_not_the_required_phase_mechanism; next isolate temporal_phase_increment_representation"
            ),
        },
        "items": items,
        "next_action": (
            "listen_to_candidate_mag_griffinlim64_render_against_candidate_mag_target_phase_ceiling_and_"
            "identity_roundtrip_ceiling; do_not_train_or_modify_renderer"
        ),
    }
    _atomic_json(output_dir / "candidate_magnitude_phase_recovery_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--utterance-id", action="append", dest="utterance_ids", default=None)
    parser.add_argument("--scan-items", type=int, default=DEFAULT_SCAN_ITEMS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=GRIFFIN_LIM_ITERATIONS)
    args = parser.parse_args()
    requested = tuple(args.utterance_ids) if args.utterance_ids else DEFAULT_UTTERANCE_IDS
    print(
        json.dumps(
            run_candidate_magnitude_phase_recovery(
                args.root,
                utterance_ids=requested,
                scan_items=args.scan_items,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                iterations=args.iterations,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
