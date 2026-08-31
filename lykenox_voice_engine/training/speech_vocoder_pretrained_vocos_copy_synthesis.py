"""Full-utterance copy-synthesis gate using an external pretrained Vocos vocoder.

Purpose: stop iterating on scratch vocoders until a proven pretrained 24 kHz renderer is
shown to reconstruct the same LYKENOX reference recordings cleanly. This probe:

* uses three complete held-out validation WAVs;
* uses Vocos' own pretrained 24 kHz frontend and decoder via ``vocos(waveform)``;
* performs no training, checkpoint writes, gain normalization, EQ, denoising, duration
  modification, or acoustic-model inference;
* writes FLOAT WAVs so the probe itself does not clip or normalize generated samples;
* treats listening as authoritative. Metrics are diagnostic only.

The pretrained model is downloaded by the ``vocos`` package/Hugging Face cache on first
use. That cache is external to LYKENOX model checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from lykenox_voice_engine.audio.io import load_audio
from lykenox_voice_engine.training.speech_aligner_train import _manifest_path
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import (
    _wave_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import (
    target_relative_presence_loss,
)


PROBE_VERSION = "pretrained-vocos-24khz-full-utterance-copy-synthesis-v1"
MODEL_ID = "charactr/vocos-mel-24khz"
SAMPLE_RATE = 24000
HOP_LENGTH = 256
VALIDATION_INDICES = (0, 1, 2)
OUTPUT_DIR_NAME = "pretrained_vocos_24khz_full_utterance_copy_synthesis_v1"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v6_clarity_best": training / "vocoder_direct_waveform_v6_clarity_guard_v1" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
        "mel_fidelity_best": training / "acoustic_mel_fidelity_v1" / "best.pt",
        "postnet_best": training / "acoustic_mel_postnet_v1" / "best.pt",
        "detail_best": training / "acoustic_mel_detail_head_v1" / "best.pt",
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _validation_rows(root: Path) -> list[dict[str, str]]:
    manifest = _manifest_path(root, "val")
    rows: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8", newline="") as source:
        for raw in csv.DictReader(source):
            wav_path = Path(raw["wav_path"])
            if not wav_path.is_absolute():
                wav_path = (manifest.parent / wav_path).resolve()
            rows.append(
                {
                    "utterance_id": str(raw["utterance_id"]),
                    "text": str(raw["text"]),
                    "wav_path": str(wav_path),
                }
            )
    return rows


def _reference_wave(path: Path) -> torch.Tensor:
    waveform, sample_rate = load_audio(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(sample_rate) != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            int(sample_rate),
            SAMPLE_RATE,
        )
    return waveform[0].to(torch.float32).contiguous()


def _load_vocos():
    try:
        from vocos import Vocos
    except ImportError as exc:
        raise RuntimeError(
            "The pretrained Vocos probe needs the optional inference dependency. "
            "Install it into the active venv with: python -m pip install vocos"
        ) from exc
    model = Vocos.from_pretrained(MODEL_ID)
    return model.cpu().eval()


def _paired_metrics(reference: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    common = min(int(reference.numel()), int(generated.numel()))
    if common < 1024:
        raise RuntimeError("pretrained Vocos copy-synthesis output is unexpectedly short")
    ref = reference[:common].unsqueeze(0)
    gen = generated[:common].unsqueeze(0)
    ref_wave = _wave_metrics(ref[0], SAMPLE_RATE)
    gen_wave = _wave_metrics(gen[0], SAMPLE_RATE)
    presence = target_relative_presence_loss(
        gen,
        ref,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    ref_rms = max(float(ref_wave["rms"]), 1e-12)
    gen_rms = max(float(gen_wave["rms"]), 1e-12)
    ref_centroid = max(float(ref_wave["spectral_centroid_hz"]), 1e-12)
    gen_centroid = float(gen_wave["spectral_centroid_hz"])
    return {
        "common_samples": float(common),
        "rms_relative_db": 20.0 * math.log10(gen_rms / ref_rms),
        "spectral_centroid_relative_pct": 100.0 * (gen_centroid / ref_centroid - 1.0),
        "presence_1k_8k_error_db": float(presence.presence_1k_8k_error_db.detach()),
    }


def run_pretrained_vocos_copy_synthesis(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected(root)
    before = {name: _sha256(path) for name, path in protected.items()}
    rows = _validation_rows(root)
    missing_indices = [index for index in VALIDATION_INDICES if index >= len(rows)]
    if missing_indices:
        raise RuntimeError(f"validation manifest lacks requested indices: {missing_indices}")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = _load_vocos()
    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "copy_synthesis_report.json"

    items: list[dict[str, object]] = []
    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            row = rows[dataset_index]
            reference = _reference_wave(Path(row["wav_path"]))
            generated_batch = model(reference.unsqueeze(0))
            if not isinstance(generated_batch, torch.Tensor):
                raise RuntimeError("Vocos returned a non-tensor waveform")
            if generated_batch.ndim == 1:
                generated = generated_batch
            elif generated_batch.ndim == 2 and generated_batch.shape[0] == 1:
                generated = generated_batch[0]
            else:
                raise RuntimeError(
                    f"Unexpected Vocos waveform shape: {tuple(generated_batch.shape)}"
                )
            generated = generated.detach().cpu().to(torch.float32).contiguous()
            if not bool(torch.isfinite(generated).all()):
                raise RuntimeError("Vocos generated non-finite samples")

            prefix = f"{audit_index:02d}"
            reference_path = output_dir / f"{prefix}_reference.wav"
            generated_path = output_dir / f"{prefix}_pretrained_vocos.wav"
            sf.write(str(reference_path), reference.numpy(), SAMPLE_RATE, subtype="FLOAT")
            sf.write(str(generated_path), generated.numpy(), SAMPLE_RATE, subtype="FLOAT")
            metrics = _paired_metrics(reference, generated)
            items.append(
                {
                    "audit_index": audit_index,
                    "dataset_index": dataset_index,
                    "utterance_id": row["utterance_id"],
                    "text": row["text"],
                    "reference_wav": str(reference_path),
                    "pretrained_vocos_wav": str(generated_path),
                    "reference_samples": int(reference.numel()),
                    "generated_samples": int(generated.numel()),
                    "length_delta_samples": int(generated.numel()) - int(reference.numel()),
                    "metrics_diagnostic_only": {
                        key: round(float(value), 6) for key, value in metrics.items()
                    },
                }
            )

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    report: dict[str, object] = {
        "status": "needs_listening" if checkpoints_unchanged else "fail",
        "probe_version": PROBE_VERSION,
        "model_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "validation_indices": list(VALIDATION_INDICES),
        "full_utterance": True,
        "vocos_own_pretrained_frontend_used": True,
        "lykenox_acoustic_model_used": False,
        "lykenox_vocoder_checkpoint_used": False,
        "training_started": False,
        "persistent_training_authorized": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "checkpoints_unchanged": checkpoints_unchanged,
        "metrics_can_accept_voice_quality": False,
        "audible_full_utterance_acceptance_required": True,
        "items": items,
        "report_path": str(report_path),
        "next_gate": (
            "listen_reference_vs_pretrained_vocos_full_utterances"
            if checkpoints_unchanged
            else "investigate_unexpected_checkpoint_mutation"
        ),
    }
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_pretrained_vocos_copy_synthesis(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
