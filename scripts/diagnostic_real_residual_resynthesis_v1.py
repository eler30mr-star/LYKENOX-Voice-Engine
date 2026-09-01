"""Pure DSP real-residual analysis/resynthesis diagnostic for the minimum-phase renderer.

This test isolates the estimated minimum-phase spectral envelope from the synthetic excitation.
For each owned held-out validation utterance it derives the same order-64 oracle cepstrum used
by the minimum-phase renderer, reconstructs the corresponding complex minimum-phase transfer,
divides the real waveform STFT by that transfer to obtain a real residual estimate, reconstructs
the residual waveform with ISTFT, and feeds that residual directly into the production fixed
minimum-phase time-varying filter renderer.

No learned model, synthetic pulse/noise excitation, optimizer, checkpoint, training, gain
normalization, EQ, denoise, enhancement, or duration modification is used. The production
renderer is not modified. Outputs are raw FLOAT WAV files for direct listening under LYX-POL-001.
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

from scripts.diagnostic_minimum_phase_oracle_v1 import (
    _reference_log_magnitude,
    _safe_name,
)
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
    one_sided_real_cepstrum_to_minimum_phase_fir,
    reference_log_magnitude_to_one_sided_cepstrum,
    render_time_varying_minimum_phase,
)


DIAGNOSTIC_VERSION = "owned-minimum-phase-real-residual-resynthesis-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3
TRANSFER_EPSILON = 1.0e-6


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_float_wav(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = waveform.detach().cpu().to(torch.float32).contiguous().numpy()
    sf.write(str(path), values, SAMPLE_RATE, subtype="FLOAT")


def _minimum_phase_transfer_from_cepstrum(cepstrum: torch.Tensor) -> torch.Tensor:
    """Return the exact complex transfer used internally by the production FIR conversion."""

    if cepstrum.ndim != 2:
        raise ValueError("cepstrum must have shape [frames, order]")
    if not cepstrum.is_floating_point() or cepstrum.is_complex():
        raise ValueError("cepstrum must be real floating point")
    if not bool(torch.isfinite(cepstrum).all()):
        raise ValueError("cepstrum contains non-finite values")
    order = int(cepstrum.shape[-1])
    if order < 1 or order > N_FFT // 2:
        raise ValueError("invalid cepstral order")

    causal = torch.zeros(
        int(cepstrum.shape[0]),
        N_FFT,
        dtype=cepstrum.dtype,
        device=cepstrum.device,
    )
    causal[:, 0] = cepstrum[:, 0]
    if order > 1:
        causal[:, 1:order] = 2.0 * cepstrum[:, 1:]
    complex_log_transfer = torch.fft.rfft(causal, n=N_FFT, dim=-1)
    transfer = torch.exp(complex_log_transfer)
    if transfer.shape != (int(cepstrum.shape[0]), N_FFT // 2 + 1):
        raise RuntimeError("minimum-phase transfer has unexpected shape")
    if not bool(torch.isfinite(transfer.real).all() and torch.isfinite(transfer.imag).all()):
        raise RuntimeError("minimum-phase transfer contains non-finite values")
    return transfer


def _centered_complex_stft(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 1:
        raise ValueError("waveform must be mono [samples]")
    window = torch.hann_window(N_FFT, dtype=waveform.dtype, device=waveform.device)
    spectrum = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )
    if int(spectrum.shape[0]) != N_FFT // 2 + 1:
        raise RuntimeError("real-residual STFT produced wrong bin count")
    return spectrum


def _extract_real_residual(
    waveform: torch.Tensor,
    cepstrum: torch.Tensor,
    *,
    expected_samples: int,
) -> tuple[torch.Tensor, int, int]:
    """Invert the estimated minimum-phase envelope on the real waveform STFT."""

    transfer = _minimum_phase_transfer_from_cepstrum(cepstrum)
    spectrum = _centered_complex_stft(waveform)
    analysis_frames = int(spectrum.shape[-1])
    conditioning_frames = int(cepstrum.shape[0])
    if analysis_frames < conditioning_frames:
        raise RuntimeError("complex STFT has fewer frames than oracle cepstrum")

    terminal_extension_frames = analysis_frames - conditioning_frames
    if terminal_extension_frames:
        terminal = transfer[-1:, :].expand(terminal_extension_frames, -1)
        transfer_for_stft = torch.cat((transfer, terminal), dim=0)
    else:
        transfer_for_stft = transfer
    if int(transfer_for_stft.shape[0]) != analysis_frames:
        raise RuntimeError("transfer/STFT frame alignment failed")

    residual_spectrum = spectrum / (
        transfer_for_stft.transpose(0, 1).to(spectrum.dtype) + TRANSFER_EPSILON
    )
    if not bool(
        torch.isfinite(residual_spectrum.real).all()
        and torch.isfinite(residual_spectrum.imag).all()
    ):
        raise RuntimeError("residual spectrum contains non-finite values")

    window = torch.hann_window(N_FFT, dtype=waveform.dtype, device=waveform.device)
    residual = torch.istft(
        residual_spectrum,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        length=expected_samples,
    )
    if int(residual.numel()) != expected_samples:
        raise RuntimeError("real residual violated exact output-length contract")
    if not bool(torch.isfinite(residual).all()):
        raise RuntimeError("real residual contains non-finite values")
    return residual.to(torch.float32).contiguous(), analysis_frames, terminal_extension_frames


def run_real_residual_resynthesis(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("real-residual diagnostic must use held-out data")
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
        / "vocoder_minimum_phase_oracle_real_residual_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    utterances = collect_owned_vocoder_utterances(
        root,
        split=split,
        max_items=max_items,
    )
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            expected_samples = frame_count * HOP_LENGTH
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            if int(reference.numel()) != expected_samples:
                raise RuntimeError("held-out waveform length contract changed")

            log_magnitude, magnitude_analysis_frames = _reference_log_magnitude(
                reference,
                frame_count=frame_count,
            )
            cepstrum = reference_log_magnitude_to_one_sided_cepstrum(
                log_magnitude,
                cepstral_order=CEPSTRAL_ORDER,
                n_fft=N_FFT,
            )
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("oracle cepstrum shape mismatch")

            # Explicitly call the same FIR conversion used by production so this diagnostic
            # proves the estimated envelope is realizable through the official transform.
            fir = one_sided_real_cepstrum_to_minimum_phase_fir(
                cepstrum,
                n_fft=N_FFT,
            )
            if fir.shape != (frame_count, N_FFT):
                raise RuntimeError("minimum-phase FIR shape mismatch")

            residual, complex_analysis_frames, terminal_extension_frames = _extract_real_residual(
                reference,
                cepstrum,
                expected_samples=expected_samples,
            )
            prediction = render_time_varying_minimum_phase(
                residual.unsqueeze(0),
                cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            if prediction.shape != reference.shape:
                raise RuntimeError("real-residual resynthesis/reference shape mismatch")
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError("real-residual resynthesis contains non-finite values")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__real_residual_resynthesis.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(reference_path, reference)

            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "reference_magnitude_analysis_frames": magnitude_analysis_frames,
                    "reference_complex_stft_analysis_frames": complex_analysis_frames,
                    "terminal_transfer_extension_frames": terminal_extension_frames,
                    "terminal_transfer_extension_rule": "repeat_last_conditioning_transfer_only_for_centered_stft_terminal_frames",
                    "cepstral_order": CEPSTRAL_ORDER,
                    "transfer_epsilon": TRANSFER_EPSILON,
                    "residual_samples": int(residual.numel()),
                    "prediction_samples": int(prediction.numel()),
                    "reference_samples": int(reference.numel()),
                    "exact_output_length": int(prediction.numel()) == expected_samples,
                    "residual_peak_abs": float(residual.abs().max()),
                    "prediction_peak_abs": float(prediction.abs().max()),
                    "real_residual_resynthesis": str(prediction_path),
                    "reference": str(reference_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_real_residual_listening",
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
        "residual_source": "owned_reference_waveform_stft_divided_by_oracle_minimum_phase_transfer",
        "resynthesis_excitation_source": "reconstructed_real_residual_waveform",
        "synthetic_excitation_used": False,
        "build_neutral_excitation_used": False,
        "model_used": False,
        "model_instantiated": False,
        "training_executed": False,
        "optimizer_created": False,
        "parameter_update_executed": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "production_renderer_modified": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "posthoc_enhancement_used": False,
        "items": items,
        "next_action": "listen_to_real_residual_resynthesis_vs_reference_before_any_renderer_or_excitation_change",
    }
    _atomic_json(output_dir / "real_residual_resynthesis_report.json", report)
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
            run_real_residual_resynthesis(
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
