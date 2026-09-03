"""No-training forensic: isolate residual magnitude vs phase against the two accepted oracle paths.

This diagnostic is intentionally narrow. It reproduces the real-residual identity ceiling and the
current residual-statistics candidate for two fixed held-out utterances, then swaps STFT magnitude and
phase between the real residual and candidate residual before the unchanged minimum-phase renderer.

No model training, optimizer step, renderer modification, post-hoc gain normalization, EQ, denoise,
duration modification, or third-party model/service is used. Human listening decides the result.
Policy: LYX-POL-001.
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

from lykenox_voice_engine.models.vocoder.network_minimum_phase_residual_statistics_source_v1 import (
    RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE,
    LykenoxResidualStatisticsSourceV1,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import (
    PITCH_CONDITIONING_V2,
    extract_pitch_conditioning_v2,
)
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
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_SEED,
    _utterance_seed,
    synthesize_residual_from_statistics,
)


DIAGNOSTIC_VERSION = "owned-residual-phase-magnitude-forensic-v1"
POLICY_ID = "LYX-POL-001"
OUTPUT_DIR_NAME = "vocoder_residual_phase_magnitude_forensic_v1"
DEFAULT_UTTERANCE_IDS = (
    "speech_0021_6cd35984e877_seg_001",
    "speech_0022_ba721f6129b9_seg_005",
)
DEFAULT_SCAN_ITEMS = 256


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


def _load_candidate(path: Path) -> LykenoxResidualStatisticsSourceV1:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("residual-statistics checkpoint schema mismatch")
    if payload.get("architecture") != RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE:
        raise RuntimeError("residual-statistics architecture mismatch")
    if payload.get("conditioning_contract") != PITCH_CONDITIONING_V2:
        raise RuntimeError("residual-statistics conditioning contract mismatch")
    model = LykenoxResidualStatisticsSourceV1().cpu()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


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
    value = torch.istft(
        spectrum,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        length=length,
    )
    if int(value.numel()) != length:
        raise RuntimeError("hybrid residual violated exact-length contract")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("hybrid residual contains non-finite values")
    return value.to(torch.float32).contiguous()


def _hybrid_residuals(
    target_residual: torch.Tensor,
    candidate_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_residual.shape != candidate_residual.shape:
        raise RuntimeError("target/candidate residual length mismatch")
    target_spec = _stft(target_residual)
    candidate_spec = _stft(candidate_residual)
    if target_spec.shape != candidate_spec.shape:
        raise RuntimeError("target/candidate STFT geometry mismatch")

    target_mag_candidate_phase_spec = torch.polar(
        target_spec.abs(),
        torch.angle(candidate_spec),
    )
    candidate_mag_target_phase_spec = torch.polar(
        candidate_spec.abs(),
        torch.angle(target_spec),
    )
    length = int(target_residual.numel())
    return (
        _istft(
            target_mag_candidate_phase_spec,
            length=length,
            dtype=target_residual.dtype,
        ),
        _istft(
            candidate_mag_target_phase_spec,
            length=length,
            dtype=target_residual.dtype,
        ),
    )


def _phase_alignment_score(target: torch.Tensor, candidate: torch.Tensor) -> float:
    target_spec = _stft(target)
    candidate_spec = _stft(candidate)
    phase_delta = torch.angle(target_spec) - torch.angle(candidate_spec)
    weights = target_spec.abs().square()
    score = (torch.cos(phase_delta) * weights).sum() / weights.sum().clamp_min(1.0e-20)
    return float(score)


def _log_magnitude_l1(target: torch.Tensor, candidate: torch.Tensor) -> float:
    target_mag = _stft(target).abs().clamp_min(1.0e-6)
    candidate_mag = _stft(candidate).abs().clamp_min(1.0e-6)
    return float((torch.log(target_mag) - torch.log(candidate_mag)).abs().mean())


def run_residual_phase_magnitude_forensic(
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
        raise RuntimeError(
            "requested held-out utterances not found within scan: " + ", ".join(missing)
        )

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
                anchor_periodicity_threshold=float(
                    PITCH_CONFIG["voiced_periodicity_threshold"]
                ),
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
                raise RuntimeError("candidate residual length differs from real residual")

            (
                target_mag_candidate_phase_residual,
                candidate_mag_target_phase_residual,
            ) = _hybrid_residuals(target_residual, candidate_residual)

            residuals = {
                "target_real_residual": target_residual,
                "candidate_statistics_residual": candidate_residual,
                "target_mag_candidate_phase_residual": target_mag_candidate_phase_residual,
                "candidate_mag_target_phase_residual": candidate_mag_target_phase_residual,
            }
            renders = {
                name: render_time_varying_minimum_phase(
                    residual.unsqueeze(0),
                    oracle_cepstrum.unsqueeze(0),
                    hop_length=HOP_LENGTH,
                    n_fft=N_FFT,
                ).squeeze(0)
                for name, residual in residuals.items()
            }
            for render in renders.values():
                if render.shape != reference.shape:
                    raise RuntimeError("forensic render/reference shape mismatch")
                if not bool(torch.isfinite(render).all()):
                    raise RuntimeError("forensic render contains non-finite values")

            stem = utterance_id
            paths: dict[str, str] = {}
            reference_path = output_dir / f"{stem}__reference.wav"
            _write(reference_path, reference)
            paths["reference"] = str(reference_path)

            residual_names = {
                "target_real_residual": "target_real_residual",
                "candidate_statistics_residual": "candidate_statistics_residual",
                "target_mag_candidate_phase_residual": "target_mag_candidate_phase_residual",
                "candidate_mag_target_phase_residual": "candidate_mag_target_phase_residual",
            }
            for key, file_label in residual_names.items():
                path = output_dir / f"{stem}__{file_label}.wav"
                _write(path, residuals[key])
                paths[key] = str(path)

            render_names = {
                "target_real_residual": "identity_roundtrip_ceiling",
                "candidate_statistics_residual": "candidate_statistics_render",
                "target_mag_candidate_phase_residual": "target_mag_candidate_phase_render",
                "candidate_mag_target_phase_residual": "candidate_mag_target_phase_render",
            }
            for key, file_label in render_names.items():
                path = output_dir / f"{stem}__{file_label}.wav"
                _write(path, renders[key])
                paths[file_label] = str(path)

            items.append(
                {
                    "utterance_id": utterance_id,
                    "phase_alignment_score_target_vs_candidate": _phase_alignment_score(
                        target_residual,
                        candidate_residual,
                    ),
                    "log_magnitude_l1_target_vs_candidate": _log_magnitude_l1(
                        target_residual,
                        candidate_residual,
                    ),
                    "paths": paths,
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_phase_magnitude_forensic_listening",
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
        "metrics_can_accept_product_quality": False,
        "listening_interpretation": {
            "target_mag_candidate_phase_bad_candidate_mag_target_phase_good": (
                "candidate_phase_or_temporal_coherence_is_primary_failure"
            ),
            "target_mag_candidate_phase_good_candidate_mag_target_phase_bad": (
                "candidate_magnitude_or_microdynamic_structure_is_primary_failure"
            ),
            "both_hybrids_bad": (
                "phase_magnitude_coupling_or_nonseparable_temporal_structure_is_primary_failure"
            ),
            "both_hybrids_good": (
                "inspect_candidate_residual_synthesis_interaction_before_any_new_architecture"
            ),
        },
        "items": items,
        "next_action": (
            "listen_only_to_reference_identity_candidate_and_two_phase_magnitude_hybrid_renders; "
            "do_not_train_or_modify_renderer"
        ),
    }
    _atomic_json(output_dir / "residual_phase_magnitude_forensic_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--utterance-id",
        action="append",
        dest="utterance_ids",
        default=None,
        help="Exact held-out utterance id. Repeat to analyze more than one.",
    )
    parser.add_argument("--scan-items", type=int, default=DEFAULT_SCAN_ITEMS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    requested = (
        tuple(args.utterance_ids)
        if args.utterance_ids
        else DEFAULT_UTTERANCE_IDS
    )
    print(
        json.dumps(
            run_residual_phase_magnitude_forensic(
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
