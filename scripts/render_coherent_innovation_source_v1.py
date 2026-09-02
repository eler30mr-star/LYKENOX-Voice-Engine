"""Render complete held-out audio from the LYKENOX coherent + innovation residual source."""

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

from lykenox_voice_engine.models.vocoder.network_minimum_phase_coherent_innovation_source_v1 import (
    COHERENT_INNOVATION_ARCHITECTURE,
    HOP_LENGTH,
    LykenoxCoherentInnovationResidualSourceV1,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_vocoder_coherent_innovation_source_train_v1 import (
    CHECKPOINT_SCHEMA_VERSION,
    POLICY_ID,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import _ola_vectors
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_time_varying_minimum_phase,
)


EVALUATION_VERSION = "owned-coherent-innovation-source-heldout-listening-v1"
EVALUATION_SEED = 20260923


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


def _seed(utterance_id: str) -> int:
    digest = hashlib.sha256(f"{EVALUATION_SEED}|{utterance_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def render_heldout(
    root: Path,
    *,
    heldout_items: int = 3,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    if heldout_items < 1 or heldout_items > 3:
        raise ValueError("heldout_items must be in [1,3]")
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root / "models" / "lykenox_identity" / "training" / "coherent_innovation_source_v1" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("coherent-innovation checkpoint schema mismatch")
    if payload.get("architecture") != COHERENT_INNOVATION_ARCHITECTURE:
        raise RuntimeError("coherent-innovation checkpoint architecture mismatch")

    model = LykenoxCoherentInnovationResidualSourceV1().cpu()
    model.load_state_dict(payload["model_state"])
    model.eval()
    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_minimum_phase_coherent_innovation_source_v1"
    )
    utterances = collect_owned_vocoder_utterances(root, "val", max_items=heldout_items)
    items: list[dict[str, object]] = []

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            expected_samples = frame_count * HOP_LENGTH
            target_residual, oracle_cepstrum, _ = extract_owned_real_residual(
                utterance.waveform.cpu(), frame_count=frame_count
            )
            vectors, _, _, log_rms, innovation_mix = model.forward_with_components(
                utterance.mel.unsqueeze(0).cpu(),
                utterance.f0_hz.unsqueeze(0).cpu(),
                utterance.voiced.unsqueeze(0).cpu(),
                utterance.periodicity.unsqueeze(0).cpu(),
                innovation_seed=_seed(utterance.utterance_id),
            )
            predicted_residual = _ola_vectors(vectors, output_samples=expected_samples).squeeze(0)
            prediction = render_time_varying_minimum_phase(
                predicted_residual.unsqueeze(0),
                oracle_cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            identity = render_time_varying_minimum_phase(
                target_residual.unsqueeze(0),
                oracle_cepstrum.unsqueeze(0),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            ).squeeze(0)
            reference = utterance.waveform.cpu().to(torch.float32).contiguous()
            if prediction.shape != reference.shape or identity.shape != reference.shape:
                raise RuntimeError("coherent-innovation heldout output length mismatch")

            stem = utterance.utterance_id
            prediction_path = output_dir / f"{stem}__coherent_innovation_source.wav"
            residual_path = output_dir / f"{stem}__coherent_innovation_predicted_residual.wav"
            identity_path = output_dir / f"{stem}__identity_roundtrip_ceiling.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write(prediction_path, prediction)
            _write(residual_path, predicted_residual)
            _write(identity_path, identity)
            _write(reference_path, reference)
            reference_rms = _rms(reference)
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "prediction_rms": _rms(prediction),
                    "reference_rms": reference_rms,
                    "prediction_to_reference_rms_ratio": _rms(prediction) / max(reference_rms, 1.0e-12),
                    "innovation_mix_mean": float(innovation_mix.mean()),
                    "innovation_mix_min": float(innovation_mix.min()),
                    "innovation_mix_max": float(innovation_mix.max()),
                    "predicted_log_rms_mean": float(log_rms.mean()),
                    "coherent_innovation_source": str(prediction_path),
                    "predicted_residual": str(residual_path),
                    "identity_roundtrip_ceiling": str(identity_path),
                    "reference": str(reference_path),
                }
            )

    report: dict[str, object] = {
        "status": "ready_for_coherent_innovation_source_heldout_listening",
        "evaluation_version": EVALUATION_VERSION,
        "policy_id": POLICY_ID,
        "architecture": COHERENT_INNOVATION_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "checkpoint": str(checkpoint),
        "checkpoint_training_update": int(payload.get("update", -1)),
        "v2_warm_start": payload.get("warm_start", {}),
        "codebook_used": False,
        "teacher_forcing_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "oracle_cepstrum_used_only_to_isolate_source_quality": True,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required": True,
        "items": items,
        "next_action": "listen_for_robotic_timbre_reduction_vs_v2_and_identity_roundtrip",
    }
    _atomic_json(output_dir / "coherent_innovation_source_v1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--heldout-items", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(render_heldout(args.root, heldout_items=args.heldout_items, checkpoint=args.checkpoint), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
