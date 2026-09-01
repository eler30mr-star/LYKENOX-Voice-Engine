"""Owned CELP-style residual codebook artifact for LYKENOX diagnostics.

This module builds a fixed excitation codebook exclusively from owned ``train`` real residuals
extracted with the positive Step-3f method.  It is NOT a learned voice model, checkpoint, external
codec, or production inference path.  No validation residual is admitted into the codebook.

The representation uses 512-sample sqrt-Hann analysis windows at the existing 256-sample hop.
A deterministic hash sampler retains a bounded number of real residual windows per conditioning
bucket (voicing state, F0 bin, periodicity bin).  The paired synthesis window gives exact overlap-
add reconstruction when the original analysis vectors are selected with gain 1.

Policy: LYX-POL-001.  CPU only.  Owned data only.  No third-party voice assets or weights.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
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
)


RESIDUAL_CODEBOOK_VERSION = "owned-residual-celp-codebook-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_ITEMS = 1_000_000
DEFAULT_MAX_PER_BUCKET = 128
CODEVECTOR_SAMPLES = HOP_LENGTH * 2
F0_BIN_HZ = 20.0
PERIODICITY_BIN_WIDTH = 0.20


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


def _sqrt_hann(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    window = torch.hann_window(CODEVECTOR_SAMPLES, periodic=True, dtype=dtype, device="cpu")
    return torch.sqrt(window.clamp_min(0.0))


def residual_analysis_vectors(residual: torch.Tensor) -> torch.Tensor:
    """Return sqrt-Hann analysis vectors with one-hop padding on both sides.

    For ``T*HOP_LENGTH`` samples this produces ``T+1`` vectors.  Using the same sqrt-Hann as
    synthesis and cropping the one-hop padding reconstructs the original residual exactly up to
    floating-point error when the same vectors are selected with gain 1.
    """

    if residual.ndim != 1 or not residual.is_floating_point() or residual.is_complex():
        raise ValueError("residual must be a real floating mono tensor")
    if int(residual.numel()) < HOP_LENGTH or int(residual.numel()) % HOP_LENGTH != 0:
        raise ValueError("residual length must be a positive multiple of hop length")
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("residual contains non-finite values")
    residual = residual.detach().cpu().to(torch.float32).contiguous()
    padded = F.pad(residual, (HOP_LENGTH, HOP_LENGTH))
    frames = padded.unfold(0, CODEVECTOR_SAMPLES, HOP_LENGTH).contiguous()
    expected = int(residual.numel()) // HOP_LENGTH + 1
    if int(frames.shape[0]) != expected or int(frames.shape[1]) != CODEVECTOR_SAMPLES:
        raise RuntimeError("residual codevector analysis geometry changed")
    return frames * _sqrt_hann(dtype=frames.dtype).unsqueeze(0)


def residual_synthesis_from_analysis_vectors(vectors: torch.Tensor, *, output_samples: int) -> torch.Tensor:
    """Overlap-add selected analysis vectors with the paired sqrt-Hann synthesis window."""

    if vectors.ndim != 2 or int(vectors.shape[1]) != CODEVECTOR_SAMPLES:
        raise ValueError("vectors must have shape [frames, codevector_samples]")
    if output_samples < HOP_LENGTH or output_samples % HOP_LENGTH != 0:
        raise ValueError("output_samples must be a positive multiple of hop length")
    expected_frames = output_samples // HOP_LENGTH + 1
    if int(vectors.shape[0]) != expected_frames:
        raise ValueError("vector count does not match requested output length")
    vectors = vectors.detach().cpu().to(torch.float32).contiguous()
    window = _sqrt_hann(dtype=vectors.dtype)
    padded_samples = output_samples + 2 * HOP_LENGTH
    output = torch.zeros(padded_samples, dtype=vectors.dtype)
    for frame_index in range(expected_frames):
        start = frame_index * HOP_LENGTH
        output[start : start + CODEVECTOR_SAMPLES] += vectors[frame_index] * window
    cropped = output[HOP_LENGTH : HOP_LENGTH + output_samples].contiguous()
    if int(cropped.numel()) != output_samples or not bool(torch.isfinite(cropped).all()):
        raise RuntimeError("residual codevector synthesis failed")
    return cropped


def _voicing_state(voiced: float) -> str:
    if voiced < 0.25:
        return "unvoiced"
    if voiced < 0.75:
        return "mixed"
    return "voiced"


def _conditioning_bucket(*, f0_hz: float, voiced: float, periodicity: float) -> tuple[str, int, int]:
    state = _voicing_state(float(voiced))
    if state == "unvoiced" or not math.isfinite(float(f0_hz)) or float(f0_hz) <= 0.0:
        f0_bin = -1
    else:
        f0_bin = int(math.floor(float(f0_hz) / F0_BIN_HZ) * F0_BIN_HZ)
    clipped_periodicity = min(1.0, max(0.0, float(periodicity)))
    periodicity_bin = min(
        int(round(1.0 / PERIODICITY_BIN_WIDTH)) - 1,
        int(math.floor(clipped_periodicity / PERIODICITY_BIN_WIDTH)),
    )
    return state, f0_bin, periodicity_bin


def _bucket_key(bucket: tuple[str, int, int]) -> str:
    state, f0_bin, periodicity_bin = bucket
    return f"{state}|f0={f0_bin}|p={periodicity_bin}"


def _selection_score(*, utterance_id: str, frame_index: int) -> int:
    payload = f"{RESIDUAL_CODEBOOK_VERSION}|{utterance_id}|{frame_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def _push_bounded(
    heap: list[tuple[int, str, torch.Tensor]],
    *,
    score: int,
    stable_key: str,
    vector: torch.Tensor,
    limit: int,
) -> None:
    entry = (-int(score), stable_key, vector.detach().cpu().to(torch.float32).clone())
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    current_largest_score = -heap[0][0]
    if score < current_largest_score:
        heapq.heapreplace(heap, entry)


def build_owned_residual_codebook(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_per_bucket: int = DEFAULT_MAX_PER_BUCKET,
    tensor_path: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, object]:
    if split != "train":
        raise ValueError("identity residual codebook must be built exclusively from owned train data")
    if max_items < 1:
        raise ValueError("max_items must be positive")
    if max_per_bucket < 1 or max_per_bucket > 4096:
        raise ValueError("max_per_bucket must be in [1, 4096]")

    root = Path(root).resolve()
    base = root / "models" / "lykenox_identity" / "calibration"
    tensor_path = (
        Path(tensor_path).resolve()
        if tensor_path is not None
        else base / "residual_codebook_v1.pt"
    )
    index_path = (
        Path(index_path).resolve()
        if index_path is not None
        else base / "residual_codebook_v1.json"
    )
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    utterances = collect_owned_vocoder_utterances(root, split=split, max_items=max_items)
    heaps: dict[tuple[str, int, int], list[tuple[int, str, torch.Tensor]]] = {}
    provenance: list[dict[str, object]] = []
    candidate_window_count = 0

    with torch.no_grad():
        for utterance in utterances:
            frame_count = int(utterance.mel_frames)
            residual, _, extension_frames = extract_owned_real_residual(
                utterance.waveform.cpu(),
                frame_count=frame_count,
            )
            vectors = residual_analysis_vectors(residual)
            if int(vectors.shape[0]) != frame_count + 1:
                raise RuntimeError("codebook residual vector count mismatch")

            accepted_for_utterance = 0
            for frame_index in range(frame_count + 1):
                conditioning_index = min(frame_index, frame_count - 1)
                f0_hz = float(utterance.f0_hz[conditioning_index])
                voiced = float(utterance.voiced[conditioning_index])
                periodicity = float(utterance.periodicity[conditioning_index])
                bucket = _conditioning_bucket(
                    f0_hz=f0_hz,
                    voiced=voiced,
                    periodicity=periodicity,
                )
                score = _selection_score(
                    utterance_id=utterance.utterance_id,
                    frame_index=frame_index,
                )
                stable_key = f"{utterance.utterance_id}:{frame_index:08d}"
                heap = heaps.setdefault(bucket, [])
                _push_bounded(
                    heap,
                    score=score,
                    stable_key=stable_key,
                    vector=vectors[frame_index],
                    limit=max_per_bucket,
                )
                candidate_window_count += 1
                accepted_for_utterance += 1

            wav_path = Path(utterance.wav_path)
            provenance.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "wav_path": str(wav_path),
                    "wav_sha256": _sha256_file(wav_path),
                    "conditioning_frames": frame_count,
                    "candidate_codevector_windows": accepted_for_utterance,
                    "pitch_cache_version": utterance.pitch_cache_version,
                    "conditioning_contract_version": utterance.conditioning_contract_version,
                    "terminal_transfer_extension_frames": extension_frames,
                }
            )

    if not heaps:
        raise RuntimeError("no owned train residual vectors were available for codebook construction")

    all_vectors: list[torch.Tensor] = []
    bucket_records: list[dict[str, object]] = []
    offset = 0
    for bucket in sorted(heaps, key=_bucket_key):
        selected = sorted(heaps[bucket], key=lambda item: (-item[0], item[1]))
        vectors = [item[2] for item in selected]
        if not vectors:
            continue
        stacked = torch.stack(vectors, dim=0).to(torch.float32).contiguous()
        count = int(stacked.shape[0])
        state, f0_bin, periodicity_bin = bucket
        bucket_records.append(
            {
                "key": _bucket_key(bucket),
                "voicing_state": state,
                "f0_bin_hz": f0_bin,
                "periodicity_bin": periodicity_bin,
                "periodicity_min": periodicity_bin * PERIODICITY_BIN_WIDTH,
                "periodicity_max": min(1.0, (periodicity_bin + 1) * PERIODICITY_BIN_WIDTH),
                "start_index": offset,
                "count": count,
            }
        )
        all_vectors.append(stacked)
        offset += count

    codewords = torch.cat(all_vectors, dim=0).contiguous()
    if codewords.ndim != 2 or int(codewords.shape[1]) != CODEVECTOR_SAMPLES:
        raise RuntimeError("assembled residual codebook has invalid geometry")
    if not bool(torch.isfinite(codewords).all()):
        raise RuntimeError("assembled residual codebook contains non-finite values")

    temporary_tensor = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    torch.save(codewords, temporary_tensor)
    os.replace(temporary_tensor, tensor_path)
    tensor_sha256 = _sha256_file(tensor_path)

    artifact: dict[str, object] = {
        "status": "built_from_owned_train_real_residual",
        "codebook_version": RESIDUAL_CODEBOOK_VERSION,
        "policy_id": POLICY_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "source_split": split,
        "identity_data_owned_or_authorized_required": True,
        "third_party_voice_data_used": False,
        "third_party_model_or_checkpoint_used": False,
        "remote_inference_used": False,
        "training_executed": False,
        "optimizer_created": False,
        "gradient_updates_executed": False,
        "production_active": False,
        "artifact_role": "diagnostic_owned_residual_codebook_not_model_checkpoint",
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "cepstral_order": CEPSTRAL_ORDER,
        "renderer_version": RENDERER_VERSION,
        "full_utterance_data_version": FULL_UTTERANCE_DATA_VERSION,
        "source_residual_method": "owned_reference_stft_divided_by_order64_oracle_minimum_phase_transfer",
        "analysis_window": "sqrt_hann_periodic",
        "codevector_samples": CODEVECTOR_SAMPLES,
        "codevector_hop_samples": HOP_LENGTH,
        "terminal_vector_rule": "one_hop_padding_produces_frame_count_plus_one_analysis_vectors",
        "f0_bin_width_hz": F0_BIN_HZ,
        "periodicity_bin_width": PERIODICITY_BIN_WIDTH,
        "max_per_bucket": max_per_bucket,
        "candidate_window_count": candidate_window_count,
        "retained_codeword_count": int(codewords.shape[0]),
        "bucket_count": len(bucket_records),
        "tensor_path": str(tensor_path),
        "tensor_sha256": tensor_sha256,
        "buckets": bucket_records,
        "provenance": provenance,
        "metrics_can_accept_product_quality": False,
        "human_full_utterance_listening_required_before_any_product_integration": True,
        "next_action": "run_heldout_celp_style_oracle_search_without_training",
    }
    _atomic_json(index_path, artifact)
    return artifact


def load_owned_residual_codebook(
    tensor_path: Path,
    index_path: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor_path = Path(tensor_path).resolve()
    index_path = Path(index_path).resolve()
    metadata = json.loads(index_path.read_text(encoding="utf-8"))
    if metadata.get("codebook_version") != RESIDUAL_CODEBOOK_VERSION:
        raise RuntimeError("unsupported residual codebook version")
    if metadata.get("policy_id") != POLICY_ID:
        raise RuntimeError("residual codebook policy mismatch")
    if metadata.get("source_split") != "train":
        raise RuntimeError("residual codebook is not derived exclusively from train")
    if metadata.get("third_party_voice_data_used") is not False:
        raise RuntimeError("residual codebook provenance does not prove owned-only data")
    if metadata.get("third_party_model_or_checkpoint_used") is not False:
        raise RuntimeError("residual codebook provenance includes prohibited third-party model data")
    if metadata.get("training_executed") is not False:
        raise RuntimeError("residual codebook artifact unexpectedly reports model training")
    if _sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise RuntimeError("residual codebook tensor hash mismatch")
    codewords = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(codewords, torch.Tensor):
        raise RuntimeError("residual codebook tensor artifact is not a tensor")
    codewords = codewords.to(torch.float32).contiguous()
    if codewords.ndim != 2 or int(codewords.shape[1]) != CODEVECTOR_SAMPLES:
        raise RuntimeError("residual codebook tensor geometry mismatch")
    if int(codewords.shape[0]) != int(metadata.get("retained_codeword_count", -1)):
        raise RuntimeError("residual codebook count metadata mismatch")
    if not bool(torch.isfinite(codewords).all()):
        raise RuntimeError("residual codebook contains non-finite values")
    return codewords, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--max-per-bucket", type=int, default=DEFAULT_MAX_PER_BUCKET)
    parser.add_argument("--tensor-path", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            build_owned_residual_codebook(
                args.root,
                split="train",
                max_items=args.max_items,
                max_per_bucket=args.max_per_bucket,
                tensor_path=args.tensor_path,
                index_path=args.index_path,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
