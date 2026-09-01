"""Pure DSP band-split excitation diagnostic for the owned minimum-phase renderer.

This Step 3d diagnostic changes exactly one renderer variable relative to the original
minimum-phase oracle: the final broadband periodic/aperiodic mixture is replaced by a crude
two-band split around 2 kHz.  The owned pitch-v1 conditioning, pulse generation, deterministic
aperiodic noise, pulse low-pass, minimum-phase filter renderer, frame crossfade, reference
oracle cepstrum, and exact duration are otherwise unchanged.

The production renderer is not modified.  No model, optimizer, training, checkpoint IO,
post-hoc gain normalization, EQ, denoise, enhancement, or duration modification is used.
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

from lykenox_voice_engine.training.speech_vocoder_loss_v2 import _centered_stft_magnitude
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    FULL_UTTERANCE_DATA_VERSION,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    _deterministic_aperiodic_noise,
    _fixed_lowpass_kernel,
    fixed_linear_frame_to_sample,
    reference_log_magnitude_to_one_sided_cepstrum,
    render_time_varying_minimum_phase,
)


DIAGNOSTIC_VERSION = "owned-minimum-phase-oracle-band-split-excitation-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
NOISE_SEED = 0
STFT_EPSILON = 1.0e-5
BAND_SPLIT_HZ = 2000.0
BAND_SPLIT_TAPS = 63
HIGH_PERIODIC_STRENGTH_SCALE = 0.5


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _write_float_wav(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = waveform.detach().cpu().to(torch.float32).contiguous().numpy()
    sf.write(str(path), values, SAMPLE_RATE, subtype="FLOAT")


def _reference_log_magnitude(
    waveform: torch.Tensor,
    *,
    frame_count: int,
) -> tuple[torch.Tensor, int]:
    if waveform.ndim != 1:
        raise ValueError("oracle reference waveform must be mono [samples]")
    expected_samples = int(frame_count) * HOP_LENGTH
    if int(waveform.numel()) != expected_samples:
        raise ValueError("oracle waveform length must equal frame_count * hop_length")

    magnitude = _centered_stft_magnitude(
        waveform.unsqueeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
    )
    analysis_frames = int(magnitude.shape[-1])
    if int(magnitude.shape[1]) != N_FFT // 2 + 1:
        raise RuntimeError("oracle STFT produced the wrong rFFT bin count")
    if analysis_frames < frame_count:
        raise RuntimeError("oracle STFT produced fewer frames than conditioning")
    aligned = magnitude[0, :, :frame_count].transpose(0, 1).contiguous()
    log_magnitude = torch.log(aligned.clamp_min(STFT_EPSILON))
    if log_magnitude.shape != (frame_count, N_FFT // 2 + 1):
        raise RuntimeError("oracle log-magnitude alignment failed")
    if not bool(torch.isfinite(log_magnitude).all()):
        raise RuntimeError("oracle log-magnitude contains non-finite values")
    return log_magnitude, analysis_frames


def _fixed_highpass_kernel(
    *,
    device: torch.device,
    dtype: torch.dtype,
    taps: int = BAND_SPLIT_TAPS,
    cutoff_hz: float = BAND_SPLIT_HZ,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Complementary high-pass formed as delta minus the owned fixed low-pass kernel."""

    lowpass = _fixed_lowpass_kernel(
        device=device,
        dtype=dtype,
        taps=taps,
        cutoff_hz=cutoff_hz,
        sample_rate=sample_rate,
    )
    delta = torch.zeros_like(lowpass)
    delta[(taps - 1) // 2] = 1.0
    return delta - lowpass


def build_neutral_excitation_band_split(
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    noise_seed: int = NOISE_SEED,
) -> torch.Tensor:
    """Copy of production excitation with only the final broadband mixture replaced."""

    for name, value in (("f0_hz", f0_hz), ("voiced", voiced), ("periodicity", periodicity)):
        if value.is_complex() or not value.is_floating_point():
            raise ValueError(f"{name} must be a real floating tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, frame_count]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    if f0_hz.shape != voiced.shape or f0_hz.shape != periodicity.shape:
        raise ValueError("f0_hz, voiced and periodicity must share shape")

    # Production-equivalent frame -> sample interpolation and periodic strength.
    f0 = fixed_linear_frame_to_sample(f0_hz.clamp_min(0.0), hop_length=hop_length)
    voiced_sample = fixed_linear_frame_to_sample(voiced.clamp(0.0, 1.0), hop_length=hop_length)
    periodicity_sample = fixed_linear_frame_to_sample(
        periodicity.clamp(0.0, 1.0), hop_length=hop_length
    )
    periodic_strength = (voiced_sample * periodicity_sample).clamp(0.0, 1.0)

    # Production-equivalent pulse source.
    phase_increment = torch.where(
        f0 > 0.0,
        f0 / float(sample_rate),
        torch.zeros_like(f0),
    )
    accumulated = torch.cumsum(phase_increment, dim=-1)
    previous = F.pad(accumulated[..., :-1], (1, 0), value=0.0)
    pulse = (torch.floor(accumulated) > torch.floor(previous)).to(f0.dtype)
    pulse_scale = torch.where(
        f0 > 1.0,
        torch.sqrt(float(sample_rate) / f0.clamp_min(1.0)),
        torch.zeros_like(f0),
    )
    pulse = pulse * pulse_scale

    pulse_kernel = _fixed_lowpass_kernel(
        device=f0.device,
        dtype=f0.dtype,
        sample_rate=sample_rate,
    )
    pulse_padding = (pulse_kernel.numel() - 1) // 2
    bandlimited_pulse = F.conv1d(
        pulse.unsqueeze(1),
        pulse_kernel.view(1, 1, -1),
        padding=pulse_padding,
    ).squeeze(1)

    # Keep the original deterministic hash-noise source so this test isolates band mixing.
    base_noise = _deterministic_aperiodic_noise(
        f0.shape[-1],
        device=f0.device,
        dtype=f0.dtype,
        seed=int(noise_seed),
    ).unsqueeze(0).expand(f0.shape[0], -1)

    aperiodic_strength = torch.sqrt((1.0 - periodic_strength.square()).clamp_min(0.0))
    raw_mix = periodic_strength * bandlimited_pulse + aperiodic_strength * base_noise

    low_kernel = _fixed_lowpass_kernel(
        device=f0.device,
        dtype=f0.dtype,
        taps=BAND_SPLIT_TAPS,
        cutoff_hz=BAND_SPLIT_HZ,
        sample_rate=sample_rate,
    )
    high_kernel = _fixed_highpass_kernel(
        device=f0.device,
        dtype=f0.dtype,
        taps=BAND_SPLIT_TAPS,
        cutoff_hz=BAND_SPLIT_HZ,
        sample_rate=sample_rate,
    )
    padding = (low_kernel.numel() - 1) // 2

    low_band = F.conv1d(
        raw_mix.unsqueeze(1),
        low_kernel.view(1, 1, -1),
        padding=padding,
    ).squeeze(1)

    boosted_periodic_strength_high = (
        periodic_strength * HIGH_PERIODIC_STRENGTH_SCALE
    ).clamp(0.0, 1.0)
    high_aperiodic_strength = torch.sqrt(
        (1.0 - boosted_periodic_strength_high.square()).clamp_min(0.0)
    )
    high_band_reboosted = (
        boosted_periodic_strength_high * bandlimited_pulse
        + high_aperiodic_strength * base_noise
    )
    high_band_reboosted = F.conv1d(
        high_band_reboosted.unsqueeze(1),
        high_kernel.view(1, 1, -1),
        padding=padding,
    ).squeeze(1)

    excitation = low_band + high_band_reboosted
    if not bool(torch.isfinite(excitation).all()):
        raise ValueError("band-split excitation produced non-finite values")
    return excitation


def render_owned_minimum_phase_vocoder_path_band_split(
    cepstrum: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
    *,
    noise_seed: int = NOISE_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cepstrum.ndim != 3:
        raise ValueError("cepstrum must have shape [batch, frame_count, order]")
    frame_shape = cepstrum.shape[:2]
    if f0_hz.shape != frame_shape or voiced.shape != frame_shape or periodicity.shape != frame_shape:
        raise ValueError("conditioning tensors must match cepstrum batch/frame dimensions")
    excitation = build_neutral_excitation_band_split(
        f0_hz,
        voiced,
        periodicity,
        noise_seed=noise_seed,
    )
    # Keep the production filter renderer and its original crossfade unchanged.
    waveform = render_time_varying_minimum_phase(excitation, cepstrum)
    return waveform, excitation


def run_band_split_oracle(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("band-split oracle must use a held-out split")
    if max_items < 1 or max_items > 3:
        raise ValueError("max_items must be between 1 and 3")

    root = Path(root).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_oracle_band_split_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_oracle_v1"
    )

    utterances = collect_owned_vocoder_utterances(root, split=split, max_items=max_items)
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            expected_samples = frame_count * HOP_LENGTH
            if int(utterance.waveform.numel()) != expected_samples:
                raise RuntimeError("held-out oracle reference length contract changed")

            log_magnitude, analysis_frames = _reference_log_magnitude(
                utterance.waveform.cpu(),
                frame_count=frame_count,
            )
            cepstrum = reference_log_magnitude_to_one_sided_cepstrum(
                log_magnitude,
                cepstral_order=CEPSTRAL_ORDER,
                n_fft=N_FFT,
            )
            prediction, _ = render_owned_minimum_phase_vocoder_path_band_split(
                cepstrum.unsqueeze(0),
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
                noise_seed=NOISE_SEED,
            )
            prediction = prediction.squeeze(0)
            reference = utterance.waveform.cpu()
            if prediction.shape != reference.shape or int(prediction.numel()) != expected_samples:
                raise RuntimeError("band-split oracle violated exact output length")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__oracle_prediction__band_split.wav"
            reference_path = output_dir / f"{stem}__reference__band_split.wav"
            original_prediction = original_dir / f"{stem}__oracle_prediction.wav"
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(reference_path, reference)

            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "reference_stft_analysis_frames": analysis_frames,
                    "band_split_hz": BAND_SPLIT_HZ,
                    "high_periodic_strength_scale": HIGH_PERIODIC_STRENGTH_SCALE,
                    "noise_seed": NOISE_SEED,
                    "band_split_prediction": str(prediction_path),
                    "reference": str(reference_path),
                    "original_oracle_prediction": str(original_prediction),
                    "exact_output_length": True,
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_band_split_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "full_utterance_data_version": FULL_UTTERANCE_DATA_VERSION,
        "device": "cpu",
        "split": split,
        "item_count": len(items),
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "cepstral_order": CEPSTRAL_ORDER,
        "band_split_hz": BAND_SPLIT_HZ,
        "band_split_taps": BAND_SPLIT_TAPS,
        "high_periodic_strength_scale": HIGH_PERIODIC_STRENGTH_SCALE,
        "noise_generator": "production_deterministic_hash_noise_unchanged",
        "crossfade_used": True,
        "only_final_excitation_band_mixing_changed": True,
        "production_renderer_modified": False,
        "model_used": False,
        "model_instantiated": False,
        "training_executed": False,
        "optimizer_created": False,
        "parameter_update_executed": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "posthoc_enhancement_used": False,
        "items": items,
        "next_action": "listen_to_band_split_vs_original_oracle_before_any_production_renderer_change",
    }
    _atomic_json(output_dir / "band_split_oracle_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max-items", type=int, default=DEFAULT_ITEMS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_band_split_oracle(
                args.root,
                split=args.split,
                max_items=args.max_items,
                output_dir=args.output_dir,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
