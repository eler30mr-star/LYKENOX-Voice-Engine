"""Held-out renderer for the LYKENOX residual-statistics source V1.

The model predicts no waveform samples.  It predicts source cepstral shape, residual RMS and residual
periodicity, which drive one continuous full-utterance carrier with no frame phase/noise reset.  The
fixed minimum-phase vocal-tract renderer remains unchanged and uses the oracle Step-3f cepstrum here
only to isolate source quality.  Historical Continuous Source V2 is rendered side by side.

No post-hoc gain normalization, EQ or denoise is used. Metrics may reject/localize only; complete
human listening remains required. Policy: LYX-POL-001.
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

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v2 import (
    LykenoxContinuousResidualSourceV2,
)
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
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import _ola_vectors
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v2 import (
    CHECKPOINT_SCHEMA_VERSION as V2_CHECKPOINT_SCHEMA,
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
    POLICY_ID,
    _utterance_seed,
    synthesize_residual_from_statistics,
)


EVALUATION_VERSION = "owned-residual-statistics-source-heldout-v1"
OUTPUT_DIR_NAME = "vocoder_minimum_phase_residual_statistics_source_v1"
HOP_GRID_FREQUENCY_HZ = SAMPLE_RATE / float(HOP_LENGTH)


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


def _lag_correlation(value: torch.Tensor, lag: int) -> float:
    if lag < 1 or value.numel() <= lag + 8:
        return float("nan")
    left = value[:-lag].to(torch.float64)
    right = value[lag:].to(torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1.0e-20)
    return float((left * right).sum() / denominator)


def _spectral_flatness(value: torch.Tensor, *, low_hz: float = 3500.0) -> float:
    n = max(2048, int(value.numel()))
    if value.numel() < n:
        value = F.pad(value, (0, n - int(value.numel())))
    window = torch.hann_window(int(value.numel()), dtype=value.dtype)
    spectrum = torch.fft.rfft(value * window).abs().clamp_min(1.0e-8)
    frequencies = torch.fft.rfftfreq(int(value.numel()), 1.0 / float(SAMPLE_RATE))
    band = spectrum[(frequencies >= low_hz) & (frequencies <= 11000.0)]
    if band.numel() < 4:
        return float("nan")
    return float(torch.exp(torch.log(band).mean()) / band.mean().clamp_min(1.0e-8))


def _grid_harmonic_power_fraction(value: torch.Tensor, *, low_hz: float = 3500.0) -> float:
    n = max(4096, int(2 ** math.ceil(math.log2(max(int(value.numel()), 4096)))))
    spectrum = torch.fft.rfft(value.to(torch.float32), n=n)
    power = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(n, 1.0 / float(SAMPLE_RATE))
    band_mask = (frequencies >= low_hz) & (frequencies <= 11000.0)
    total = power[band_mask].sum().clamp_min(1.0e-20)
    bin_hz = SAMPLE_RATE / float(n)
    half_width = max(bin_hz * 1.5, 6.0)
    grid_mask = torch.zeros_like(band_mask)
    harmonic = HOP_GRID_FREQUENCY_HZ
    while harmonic <= 11000.0:
        if harmonic >= low_hz:
            grid_mask |= (frequencies >= harmonic - half_width) & (frequencies <= harmonic + half_width)
        harmonic += HOP_GRID_FREQUENCY_HZ
    grid_mask &= band_mask
    return float(power[grid_mask].sum() / total)


def _residual_metrics(value: torch.Tensor) -> dict[str, float]:
    return {
        "hop_lag_correlation": _lag_correlation(value, HOP_LENGTH),
        "high_band_spectral_flatness": _spectral_flatness(value),
        "high_band_grid_harmonic_power_fraction": _grid_harmonic_power_fraction(value),
    }


def _load_candidate(path: Path) -> tuple[LykenoxResidualStatisticsSourceV1, dict[str, object]]:
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
    return model, payload


def _load_v2(path: Path) -> LykenoxContinuousResidualSourceV2:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("checkpoint_schema_version") != V2_CHECKPOINT_SCHEMA:
        raise RuntimeError("V2 checkpoint schema mismatch")
    model = LykenoxContinuousResidualSourceV2().cpu()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def render_heldout_residual_statistics_source_v1(
    root: Path,
    *,
    heldout_items: int = 3,
    checkpoint: Path | None = None,
    v2_checkpoint: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = Path(checkpoint).resolve() if checkpoint is not None else root / "models" / "lykenox_identity" / "training" / "residual_statistics_source_v1" / "best.pt"
    v2_checkpoint = Path(v2_checkpoint).resolve() if v2_checkpoint is not None else root / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2" / "best.pt"
    if not checkpoint.exists() or not v2_checkpoint.exists():
        raise FileNotFoundError("required residual-statistics/V2 checkpoint missing")
    candidate, candidate_payload = _load_candidate(checkpoint)
    baseline = _load_v2(v2_checkpoint)
    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    utterances = collect_owned_vocoder_utterances(root, "val", max_items=heldout_items)
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frames = int(utterance.mel_frames)
            expected_samples = frames * HOP_LENGTH
            target_residual, oracle_cepstrum, _ = extract_owned_real_residual(utterance.waveform.cpu(), frame_count=frames)
            conditioning = extract_pitch_conditioning_v2(
                utterance.waveform.cpu().to(torch.float32),
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
                seed=_utterance_seed(utterance.utterance_id, DEFAULT_SEED + 1800000),
            )
            v2_vectors = baseline.generate(
                utterance.mel.unsqueeze(0).cpu(),
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
            )
            v2_residual = _ola_vectors(v2_vectors, output_samples=expected_samples).squeeze(0)
            candidate_wave = render_time_varying_minimum_phase(candidate_residual.unsqueeze(0), oracle_cepstrum.unsqueeze(0), hop_length=HOP_LENGTH, n_fft=N_FFT).squeeze(0)
            v2_wave = render_time_varying_minimum_phase(v2_residual.unsqueeze(0), oracle_cepstrum.unsqueeze(0), hop_length=HOP_LENGTH, n_fft=N_FFT).squeeze(0)
            identity = render_time_varying_minimum_phase(target_residual.unsqueeze(0), oracle_cepstrum.unsqueeze(0), hop_length=HOP_LENGTH, n_fft=N_FFT).squeeze(0)
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            if not (candidate_wave.shape == v2_wave.shape == identity.shape == reference.shape):
                raise RuntimeError("residual-statistics heldout output length mismatch")

            stem = utterance.utterance_id
            candidate_path = output_dir / f"{stem}__residual_statistics_source_v1.wav"
            v2_path = output_dir / f"{stem}__v2_baseline_source.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write(candidate_path, candidate_wave)
            _write(v2_path, v2_wave)
            _write(identity_path, identity)
            _write(reference_path, reference)
            reference_rms = _rms(reference)
            items.append({
                "utterance_id": stem,
                "candidate_rms_ratio": _rms(candidate_wave) / max(reference_rms, 1.0e-12),
                "v2_baseline_rms_ratio": _rms(v2_wave) / max(reference_rms, 1.0e-12),
                "mean_predicted_source_periodicity": float(source_periodicity.mean()),
                "target_residual_metrics": _residual_metrics(target_residual),
                "candidate_residual_metrics": _residual_metrics(candidate_residual),
                "v2_residual_metrics": _residual_metrics(v2_residual),
                "residual_statistics_source_v1": str(candidate_path),
                "v2_baseline_source": str(v2_path),
                "identity_roundtrip_ceiling": str(identity_path),
                "reference": str(reference_path),
            })

    report: dict[str, object] = {
        "status": "ready_for_residual_statistics_source_v1_gate",
        "evaluation_version": EVALUATION_VERSION,
        "policy_id": POLICY_ID,
        "architecture": RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "renderer_version": RENDERER_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_training_update": int(candidate_payload.get("update", -1)),
        "source_representation": "frame_statistics_plus_continuous_absolute_phase_carrier",
        "deterministic_waveform_regression": False,
        "carrier_phase_resets_per_frame": False,
        "carrier_noise_resets_per_frame": False,
        "hop_grid_frequency_hz": HOP_GRID_FREQUENCY_HZ,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required_after_structural_gate": True,
        "items": items,
        "next_action": "reject_before_listening_if_93_75hz_hop_grid_signature_remains; otherwise listen against V2 ceiling and reference",
    }
    _atomic_json(output_dir / "residual_statistics_source_v1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--v2-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(render_heldout_residual_statistics_source_v1(
        args.root,
        heldout_items=args.heldout_items,
        checkpoint=args.checkpoint,
        v2_checkpoint=args.v2_checkpoint,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
