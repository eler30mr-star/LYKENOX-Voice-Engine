"""Held-out identity roundtrip for the residual-codebook window/OLA representation.

Purpose: isolate representation geometry from codeword substitution. For each complete held-out
utterance, extract the same owned real residual proven clean in Step 3f, analyze it with the
residual-codebook 512-sample sqrt-Hann / 256-sample hop representation, synthesize the exact same
analysis vectors back without any codeword replacement or gain change, then pass the reconstructed
residual through the unchanged production minimum-phase filter.

If this remains perceptually clean, the codebook window/OLA representation is exonerated and the
remaining CELP-style failure is confined to codeword substitution/selection. No model, optimizer,
checkpoint, external voice component, remote service, post-hoc gain normalization, EQ, denoise,
or duration modification is used. CPU only under LYX-POL-001.
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

from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    CODEVECTOR_SAMPLES,
    POLICY_ID,
    RESIDUAL_CODEBOOK_VERSION,
    residual_analysis_vectors,
    residual_synthesis_from_analysis_vectors,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_time_varying_minimum_phase,
)

DIAGNOSTIC_VERSION = "owned-residual-codebook-identity-roundtrip-v1"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _write_float_wav(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        waveform.detach().cpu().to(torch.float32).contiguous().numpy(),
        SAMPLE_RATE,
        subtype="FLOAT",
    )


def _rms(waveform: torch.Tensor) -> float:
    return float(torch.sqrt(waveform.to(torch.float64).square().mean().clamp_min(1.0e-30)))


def run_identity_roundtrip(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("identity roundtrip is held-out diagnostic only")
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
        / "vocoder_minimum_phase_residual_codebook_identity_roundtrip_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    utterances = collect_owned_vocoder_utterances(root, split=split, max_items=max_items)
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            expected_samples = frame_count * HOP_LENGTH
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            target_residual, cepstrum, extension_frames = extract_owned_real_residual(
                reference,
                frame_count=frame_count,
            )
            if cepstrum.shape != (frame_count, CEPSTRAL_ORDER):
                raise RuntimeError("held-out oracle cepstrum geometry changed")

            analysis_vectors = residual_analysis_vectors(target_residual)
            reconstructed_residual = residual_synthesis_from_analysis_vectors(
                analysis_vectors,
                output_samples=expected_samples,
            )
            if reconstructed_residual.shape != target_residual.shape:
                raise RuntimeError("residual identity roundtrip length mismatch")

            residual_error = reconstructed_residual - target_residual
            prediction = render_time_varying_minimum_phase(
                reconstructed_residual.unsqueeze(0),
                cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            if prediction.shape != reference.shape:
                raise RuntimeError("identity roundtrip prediction length mismatch")
            if not bool(
                torch.isfinite(reconstructed_residual).all()
                and torch.isfinite(prediction).all()
            ):
                raise RuntimeError("identity roundtrip produced non-finite audio")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__identity_roundtrip_resynthesis.wav"
            reconstructed_path = output_dir / f"{stem}__identity_roundtrip_residual.wav"
            target_residual_path = output_dir / f"{stem}__target_real_residual.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(reconstructed_path, reconstructed_residual)
            _write_float_wav(target_residual_path, target_residual)
            _write_float_wav(reference_path, reference)

            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "analysis_vector_count": int(analysis_vectors.shape[0]),
                    "codevector_samples": CODEVECTOR_SAMPLES,
                    "codevector_hop_samples": HOP_LENGTH,
                    "terminal_transfer_extension_frames": extension_frames,
                    "residual_max_abs_error": float(residual_error.abs().max()),
                    "residual_rms_error": _rms(residual_error),
                    "target_residual_rms": _rms(target_residual),
                    "reconstructed_residual_rms": _rms(reconstructed_residual),
                    "prediction_rms": _rms(prediction),
                    "reference_rms": _rms(reference),
                    "identity_roundtrip_resynthesis": str(prediction_path),
                    "identity_roundtrip_residual": str(reconstructed_path),
                    "target_real_residual": str(target_residual_path),
                    "reference": str(reference_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_residual_codebook_identity_roundtrip_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "device": "cpu",
        "renderer_version": RENDERER_VERSION,
        "codebook_representation_version": RESIDUAL_CODEBOOK_VERSION,
        "heldout_split": split,
        "codeword_substitution_used": False,
        "oracle_codeword_selection_used": False,
        "oracle_gain_used": False,
        "exact_original_analysis_vectors_resynthesized": True,
        "model_used": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "third_party_voice_component_used": False,
        "remote_inference_used": False,
        "production_renderer_modified": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "if_clean_exonerate_window_ola_and_fix_codeword_selection_before_any_training",
    }
    _atomic_json(output_dir / "residual_codebook_identity_roundtrip_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=DEFAULT_ITEMS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_identity_roundtrip(
                args.root,
                split="val",
                max_items=args.heldout_items,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
