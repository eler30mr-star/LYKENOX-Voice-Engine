"""Pure DSP A/B diagnostic for minimum-phase frame-filter crossfade.

This script reproduces the existing minimum-phase oracle path but changes exactly one renderer
operation: each hop uses only the current frame's minimum-phase FIR.  It does not blend the
previous-filter output with the current-filter output.  The production renderer is not modified.

The diagnostic uses the same owned held-out utterances, reference waveform log magnitude,
one-sided cepstrum, pitch/voicing/periodicity, and fixed noise seed as
``diagnostic_minimum_phase_oracle_v1.py`` so the crossfade is the isolated variable.

No model, optimizer, training, checkpoint access, gain normalization, EQ, denoise, enhancement,
or duration modification is allowed.  Outputs are raw FLOAT WAV files for direct A/B listening
under LYX-POL-001.
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

from scripts.diagnostic_minimum_phase_oracle_v1 import (
    NOISE_SEED,
    _reference_log_magnitude,
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
    build_neutral_excitation,
    one_sided_real_cepstrum_to_minimum_phase_fir,
    reference_log_magnitude_to_one_sided_cepstrum,
)


DIAGNOSTIC_VERSION = "owned-minimum-phase-renderer-oracle-no-crossfade-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3


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


def render_time_varying_minimum_phase_no_crossfade(
    excitation: torch.Tensor,
    cepstrum: torch.Tensor,
    *,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> torch.Tensor:
    """Diagnostic renderer that applies only the current frame FIR to each complete hop.

    This intentionally permits frame-boundary discontinuities.  Those clicks are outside the
    hypothesis under test; the only purpose is to remove mixing between two differently phased
    filtered versions of the same excitation.
    """

    if excitation.ndim != 2:
        raise ValueError("excitation must have shape [batch, samples]")
    if cepstrum.ndim != 3:
        raise ValueError("cepstrum must have shape [batch, frame_count, order]")
    if excitation.shape[0] != cepstrum.shape[0]:
        raise ValueError("excitation and cepstrum batch sizes must match")
    if excitation.shape[-1] != cepstrum.shape[1] * hop_length:
        raise ValueError("excitation length must equal frame_count*hop_length")
    if not excitation.is_floating_point() or excitation.is_complex():
        raise ValueError("excitation must be real floating point")
    if not cepstrum.is_floating_point() or cepstrum.is_complex():
        raise ValueError("cepstrum must be real floating point")
    if not bool(torch.isfinite(excitation).all()) or not bool(torch.isfinite(cepstrum).all()):
        raise ValueError("diagnostic renderer inputs must be finite")

    filters = one_sided_real_cepstrum_to_minimum_phase_fir(cepstrum, n_fft=n_fft)
    _, frame_count, _ = filters.shape
    padded = F.pad(excitation, (n_fft - 1, 0))
    output_blocks: list[torch.Tensor] = []

    for frame_index in range(frame_count):
        start = frame_index * hop_length
        local = padded[:, start : start + hop_length + n_fft - 1]
        windows = local.unfold(-1, n_fft, 1)
        current_filter = filters[:, frame_index, :]
        current = (windows * current_filter.flip(-1).unsqueeze(1)).sum(dim=-1)
        block = current
        output_blocks.append(block)

    waveform = torch.cat(output_blocks, dim=-1)
    expected = frame_count * hop_length
    if waveform.shape[-1] != expected:
        raise RuntimeError("no-crossfade renderer violated exact output-length contract")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("no-crossfade renderer produced non-finite waveform")
    return waveform


def render_owned_minimum_phase_vocoder_path_no_crossfade(
    cepstrum: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
    *,
    noise_seed: int = NOISE_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the production excitation unchanged, then render without frame crossfade."""

    if cepstrum.ndim != 3:
        raise ValueError("cepstrum must have shape [batch, frame_count, order]")
    frame_shape = cepstrum.shape[:2]
    if f0_hz.shape != frame_shape or voiced.shape != frame_shape or periodicity.shape != frame_shape:
        raise ValueError("conditioning tensors must match cepstrum batch/frame dimensions")

    excitation = build_neutral_excitation(
        f0_hz,
        voiced,
        periodicity,
        noise_seed=noise_seed,
    )
    waveform = render_time_varying_minimum_phase_no_crossfade(excitation, cepstrum)
    return waveform, excitation


def run_no_crossfade_oracle(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("no-crossfade oracle must use held-out data, not train")
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
        / "vocoder_minimum_phase_oracle_no_crossfade_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    original_oracle_dir = (
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
            reference = utterance.waveform.cpu()
            if int(reference.numel()) != expected_samples:
                raise RuntimeError("held-out no-crossfade reference length contract changed")
            if (
                int(utterance.f0_hz.numel()) != frame_count
                or int(utterance.voiced.numel()) != frame_count
                or int(utterance.periodicity.numel()) != frame_count
            ):
                raise RuntimeError("held-out no-crossfade pitch conditioning length mismatch")

            log_magnitude, analysis_frames = _reference_log_magnitude(
                reference,
                frame_count=frame_count,
            )
            cepstrum = reference_log_magnitude_to_one_sided_cepstrum(
                log_magnitude,
                cepstral_order=CEPSTRAL_ORDER,
                n_fft=N_FFT,
            )
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("no-crossfade oracle cepstrum shape mismatch")

            prediction, _ = render_owned_minimum_phase_vocoder_path_no_crossfade(
                cepstrum.unsqueeze(0),
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
                noise_seed=NOISE_SEED,
            )
            prediction = prediction.squeeze(0)
            if int(prediction.numel()) != expected_samples or prediction.shape != reference.shape:
                raise RuntimeError("no-crossfade oracle prediction/reference length mismatch")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__oracle_prediction__no_crossfade.wav"
            reference_path = output_dir / f"{stem}__reference__no_crossfade.wav"
            original_prediction_path = original_oracle_dir / f"{stem}__oracle_prediction.wav"
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(reference_path, reference)

            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "reference_stft_analysis_frames": analysis_frames,
                    "reference_log_magnitude_bins": N_FFT // 2 + 1,
                    "cepstral_order": CEPSTRAL_ORDER,
                    "prediction_samples": int(prediction.numel()),
                    "reference_samples": int(reference.numel()),
                    "exact_output_length": int(prediction.numel()) == expected_samples,
                    "noise_seed": NOISE_SEED,
                    "crossfade_used": False,
                    "no_crossfade_prediction": str(prediction_path),
                    "reference": str(reference_path),
                    "original_crossfade_prediction": str(original_prediction_path),
                    "original_crossfade_prediction_exists": original_prediction_path.exists(),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_no_crossfade_ab_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "production_renderer_version_under_test": RENDERER_VERSION,
        "production_renderer_modified": False,
        "full_utterance_data_version": FULL_UTTERANCE_DATA_VERSION,
        "device": "cpu",
        "split": split,
        "item_count": len(items),
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "cepstral_order": CEPSTRAL_ORDER,
        "noise_seed": NOISE_SEED,
        "reference_spectral_source": "same_owned_reference_waveform_oracle_as_step_3",
        "pitch_source": "same_owned_full_utterance_pitch_cache_v1_unmodified_as_step_3",
        "excitation_path_changed": False,
        "cepstrum_oracle_path_changed": False,
        "crossfade_used": False,
        "only_intended_difference": "each_hop_uses_current_frame_filter_output_only",
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
        "next_action": "listen_ab_no_crossfade_vs_original_crossfade_before_any_renderer_fix",
    }
    _atomic_json(output_dir / "no_crossfade_oracle_report.json", report)
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
            run_no_crossfade_oracle(
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
