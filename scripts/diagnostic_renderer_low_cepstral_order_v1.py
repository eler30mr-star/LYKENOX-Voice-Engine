"""Pure DSP diagnostic for minimum-phase oracle with cepstral order 32.

This isolates only cepstral truncation from the original Step-3 oracle. The production
renderer, broadband excitation, deterministic hash noise, pitch conditioning, and filter
crossfade remain unchanged. No model, optimizer, training, checkpoint, post-processing, or
duration modification is used.
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

from lykenox_voice_engine.training.speech_vocoder_loss_v2 import _centered_stft_magnitude
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    FULL_UTTERANCE_DATA_VERSION,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER as PRODUCTION_CEPSTRAL_ORDER,
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    reference_log_magnitude_to_one_sided_cepstrum,
    render_owned_minimum_phase_vocoder_path,
)


DIAGNOSTIC_VERSION = "owned-minimum-phase-renderer-cepstral-order-32-oracle-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
NOISE_SEED = 0
STFT_EPSILON = 1.0e-5
CEPSTRAL_ORDER_TEST = 32


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
        raise ValueError("oracle waveform length must equal frame_count * hop length")

    magnitude = _centered_stft_magnitude(
        waveform.unsqueeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
    )
    analysis_frames = int(magnitude.shape[-1])
    if int(magnitude.shape[1]) != N_FFT // 2 + 1:
        raise RuntimeError("oracle STFT produced wrong rFFT bin count")
    if analysis_frames < frame_count:
        raise RuntimeError("oracle STFT produced fewer frames than conditioning")
    aligned = magnitude[0, :, :frame_count].transpose(0, 1).contiguous()
    log_magnitude = torch.log(aligned.clamp_min(STFT_EPSILON))
    if log_magnitude.shape != (frame_count, N_FFT // 2 + 1):
        raise RuntimeError("oracle log-magnitude alignment failed")
    if not bool(torch.isfinite(log_magnitude).all()):
        raise RuntimeError("oracle log-magnitude contains non-finite values")
    return log_magnitude, analysis_frames


def run_low_cepstral_order_oracle(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("renderer oracle must use held-out data")
    if max_items < 1 or max_items > 3:
        raise ValueError("max_items must be between 1 and 3")
    if PRODUCTION_CEPSTRAL_ORDER != 64:
        raise RuntimeError("production cepstral order changed; diagnostic baseline is no longer 64")

    root = Path(root).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_oracle_cepstral_order_32_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    utterances = collect_owned_vocoder_utterances(
        root,
        split=split,
        max_items=max_items,
    )
    original_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_oracle_v1"
    )
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            expected_samples = frame_count * HOP_LENGTH
            if int(utterance.waveform.numel()) != expected_samples:
                raise RuntimeError("held-out reference length contract changed")
            if (
                int(utterance.f0_hz.numel()) != frame_count
                or int(utterance.voiced.numel()) != frame_count
                or int(utterance.periodicity.numel()) != frame_count
            ):
                raise RuntimeError("held-out pitch conditioning length mismatch")

            log_magnitude, analysis_frames = _reference_log_magnitude(
                utterance.waveform.cpu(),
                frame_count=frame_count,
            )
            cepstrum = reference_log_magnitude_to_one_sided_cepstrum(
                log_magnitude,
                cepstral_order=CEPSTRAL_ORDER_TEST,
                n_fft=N_FFT,
            )
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER_TEST):
                raise RuntimeError("order-32 oracle cepstrum shape mismatch")

            prediction, _ = render_owned_minimum_phase_vocoder_path(
                cepstrum.unsqueeze(0),
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
                noise_seed=NOISE_SEED,
            )
            prediction = prediction.squeeze(0)
            reference = utterance.waveform.cpu()
            if prediction.shape != reference.shape:
                raise RuntimeError("order-32 prediction/reference shape mismatch")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__oracle_prediction__cepstral_order_32.wav"
            reference_path = output_dir / f"{stem}__reference__cepstral_order_32.wav"
            original_prediction = original_dir / f"{stem}__oracle_prediction.wav"
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(reference_path, reference)

            voiced_f0 = utterance.f0_hz[utterance.f0_hz > 0.0]
            max_f0_hz = float(voiced_f0.max().item()) if voiced_f0.numel() else 0.0
            min_pitch_period_samples = (
                SAMPLE_RATE / max_f0_hz if max_f0_hz > 0.0 else None
            )
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "reference_stft_analysis_frames": analysis_frames,
                    "reference_log_magnitude_bins": N_FFT // 2 + 1,
                    "production_cepstral_order": PRODUCTION_CEPSTRAL_ORDER,
                    "diagnostic_cepstral_order": CEPSTRAL_ORDER_TEST,
                    "maximum_voiced_f0_hz": max_f0_hz,
                    "minimum_pitch_period_samples": min_pitch_period_samples,
                    "prediction_samples": int(prediction.numel()),
                    "reference_samples": int(reference.numel()),
                    "noise_seed": NOISE_SEED,
                    "cepstral_order_32_prediction": str(prediction_path),
                    "reference": str(reference_path),
                    "original_oracle_prediction": str(original_prediction),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_cepstral_order_32_listening",
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
        "production_cepstral_order": PRODUCTION_CEPSTRAL_ORDER,
        "diagnostic_cepstral_order": CEPSTRAL_ORDER_TEST,
        "noise_seed": NOISE_SEED,
        "only_cepstral_order_changed": True,
        "broadband_excitation_unchanged": True,
        "band_split_excitation_used": False,
        "gaussian_noise_used": False,
        "production_hash_noise_used": True,
        "production_crossfade_used": True,
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
        "next_action": "listen_to_order_32_vs_original_order_64_oracle_before_any_renderer_change",
    }
    _atomic_json(output_dir / "cepstral_order_32_oracle_report.json", report)
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
            run_low_cepstral_order_oracle(
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
