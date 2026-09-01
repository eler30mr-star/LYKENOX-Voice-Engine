"""Held-out oracle for the owned calibrated Rosenberg/band-aperiodic excitation candidate.

This diagnostic keeps the proven order-64 minimum-phase envelope/filter path fixed and changes
only the source excitation.  It requires the two owned train-derived calibration artifacts,
renders three complete validation utterances, and writes raw FLOAT WAVs for listening against
both the original synthetic-excitation oracle and the real-residual resynthesis ceiling.

No model, optimizer, training, checkpoint, output gain normalization, EQ, denoise, enhancement,
or duration modification is used.  Product renderer code is not modified by this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostic_minimum_phase_oracle_v1 import _reference_log_magnitude, _safe_name
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    FULL_UTTERANCE_DATA_VERSION,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_glottal_excitation_v1 import (
    GLOTTAL_EXCITATION_VERSION,
    OwnedCalibratedGlottalExcitationV1,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    reference_log_magnitude_to_one_sided_cepstrum,
    render_time_varying_minimum_phase,
)


DIAGNOSTIC_VERSION = "owned-calibrated-glottal-minimum-phase-oracle-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_SPLIT = "val"
DEFAULT_ITEMS = 3


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _noise_seed(utterance_id: str) -> int:
    digest = hashlib.sha256(utterance_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _write_float_wav(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        waveform.detach().cpu().to(torch.float32).contiguous().numpy(),
        SAMPLE_RATE,
        subtype="FLOAT",
    )


def run_calibrated_glottal_oracle(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_ITEMS,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("calibrated glottal oracle must use held-out data")
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
        / "vocoder_minimum_phase_oracle_calibrated_glottal_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    excitation_model = OwnedCalibratedGlottalExcitationV1.from_root(root)
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
                raise RuntimeError("held-out reference length contract changed")

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
                raise RuntimeError("oracle cepstrum shape mismatch")

            seed = _noise_seed(utterance.utterance_id)
            excitation = excitation_model.build(
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
                noise_seed=seed,
            )
            if excitation.shape != (1, expected_samples):
                raise RuntimeError("calibrated excitation violated exact length contract")
            prediction = render_time_varying_minimum_phase(
                excitation,
                cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            if prediction.shape != reference.shape:
                raise RuntimeError("calibrated oracle prediction/reference shape mismatch")

            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__calibrated_glottal.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            original_oracle = (
                root
                / "models"
                / "lykenox_identity"
                / "evaluation"
                / "vocoder_minimum_phase_oracle_v1"
                / f"{stem}__oracle_prediction.wav"
            )
            real_residual = (
                root
                / "models"
                / "lykenox_identity"
                / "evaluation"
                / "vocoder_minimum_phase_oracle_real_residual_v1"
                / f"{stem}__real_residual_resynthesis.wav"
            )
            _write_float_wav(prediction_path, prediction)
            _write_float_wav(reference_path, reference)

            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "conditioning_frames": frame_count,
                    "reference_stft_analysis_frames": analysis_frames,
                    "cepstral_order": CEPSTRAL_ORDER,
                    "noise_seed_from_utterance_id": seed,
                    "prediction_samples": int(prediction.numel()),
                    "reference_samples": int(reference.numel()),
                    "excitation_peak_abs": float(excitation.abs().max()),
                    "prediction_peak_abs": float(prediction.abs().max()),
                    "calibrated_glottal_prediction": str(prediction_path),
                    "reference": str(reference_path),
                    "original_synthetic_excitation_oracle": str(original_oracle),
                    "real_residual_resynthesis_ceiling": str(real_residual),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_calibrated_glottal_oracle_listening",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "policy_id": POLICY_ID,
        "renderer_version": RENDERER_VERSION,
        "excitation_version": GLOTTAL_EXCITATION_VERSION,
        "full_utterance_data_version": FULL_UTTERANCE_DATA_VERSION,
        "device": "cpu",
        "split": split,
        "item_count": len(items),
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "cepstral_order": CEPSTRAL_ORDER,
        "glottal_calibration_artifact": str(excitation_model.glottal_path),
        "glottal_calibration_sha256": _sha256_file(excitation_model.glottal_path),
        "band_aperiodicity_artifact": str(excitation_model.aperiodicity_path),
        "band_aperiodicity_sha256": _sha256_file(excitation_model.aperiodicity_path),
        "envelope_filter_path_changed": False,
        "synthetic_excitation_path_changed": True,
        "production_renderer_modified": False,
        "model_used": False,
        "training_executed": False,
        "optimizer_created": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "posthoc_enhancement_used": False,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_against_reference_original_oracle_and_real_residual_ceiling_before_any_production_integration",
    }
    _atomic_json(output_dir / "calibrated_glottal_oracle_report.json", report)
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
            run_calibrated_glottal_oracle(
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
