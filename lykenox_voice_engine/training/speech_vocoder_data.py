"""Deterministic real mel/waveform segments for persistent LYKENOX vocoder training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
import torchaudio

from lykenox_voice_engine.audio.io import load_audio
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import LykenoxVocoderConfig
from lykenox_voice_engine.training.speech_aligner_train import _dataset


VOCODER_SEGMENT_CONTRACT_VERSION = "vocoder-segment-v1"


@dataclass(frozen=True)
class VocoderSegment:
    split: str
    utterance_id: str
    wav_path: str
    start_frame: int
    mel_frames: int
    mel: torch.Tensor
    waveform: torch.Tensor


def _mono_waveform(path: Path, config: LykenoxVocoderConfig) -> torch.Tensor:
    waveform, sample_rate = load_audio(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            config.sample_rate,
        )
    peak = waveform.abs().max().clamp_min(1e-8)
    if peak > 1.0:
        waveform = waveform / peak
    return waveform.squeeze(0).to(torch.float32).contiguous()


def _stable_start(
    *,
    seed: int,
    split: str,
    utterance_id: str,
    max_start: int,
    boundary_margin_frames: int,
) -> int:
    if max_start < 0:
        raise ValueError("max_start must be non-negative")
    margin = max(0, int(boundary_margin_frames))
    if max_start >= margin * 2:
        low = margin
        high = max_start - margin
    else:
        low = 0
        high = max_start
    digest = hashlib.sha256(
        f"{seed}:{split}:{utterance_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return low + value % (high - low + 1)


def collect_vocoder_segments(
    root: Path,
    split: str,
    *,
    segment_mel_frames: int,
    max_items: int,
    seed: int = 1337,
    boundary_margin_frames: int = 4,
) -> tuple[list[VocoderSegment], list[dict[str, object]]]:
    """Collect deterministic mel/wave pairs with exact hop-aligned sample lengths.

    The active speech mel frontend uses centered STFT features. We therefore avoid the
    utterance edges when possible and keep one stable sample-domain convention:
    mel frame ``start`` is paired with waveform sample ``start * hop_length``. This is
    the same convention used by the successful CPU feasibility benchmark and is now
    versioned so later frontend changes cannot silently shift vocoder supervision.
    """

    if segment_mel_frames < 16:
        raise ValueError("segment_mel_frames must be >= 16")
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    root = Path(root).resolve()
    speech_config = LykenoxSpeechConfig()
    vocoder_config = LykenoxVocoderConfig(
        mel_bins=speech_config.mel_bins,
        sample_rate=speech_config.sample_rate,
        hop_length=speech_config.hop_length,
    )
    dataset = _dataset(root, split, speech_config)
    selected: list[VocoderSegment] = []
    skipped: list[dict[str, object]] = []

    for index in range(len(dataset)):
        item = dataset[index]
        utterance_id = str(item["utterance_id"])
        mel = item["mel"].to(torch.float32).contiguous()
        wav_path = Path(str(item["wav_path"]))
        waveform = _mono_waveform(wav_path, vocoder_config)
        mel_frames = int(mel.shape[0])
        waveform_frames = int(waveform.numel()) // vocoder_config.hop_length
        usable_frames = min(mel_frames, waveform_frames)
        if usable_frames < segment_mel_frames:
            skipped.append(
                {
                    "utterance_id": utterance_id,
                    "reason": "too_short",
                    "mel_frames": mel_frames,
                    "waveform_hop_frames": waveform_frames,
                }
            )
            continue

        max_start = usable_frames - segment_mel_frames
        start_frame = _stable_start(
            seed=seed,
            split=split,
            utterance_id=utterance_id,
            max_start=max_start,
            boundary_margin_frames=boundary_margin_frames,
        )
        start_sample = start_frame * vocoder_config.hop_length
        sample_count = segment_mel_frames * vocoder_config.hop_length
        mel_segment = mel[start_frame : start_frame + segment_mel_frames]
        wave_segment = waveform[start_sample : start_sample + sample_count]
        if int(mel_segment.shape[0]) != segment_mel_frames:
            raise RuntimeError(f"Mel segment length contract failed for {utterance_id}")
        if int(wave_segment.numel()) != sample_count:
            raise RuntimeError(f"Waveform segment length contract failed for {utterance_id}")

        selected.append(
            VocoderSegment(
                split=split,
                utterance_id=utterance_id,
                wav_path=str(wav_path),
                start_frame=start_frame,
                mel_frames=segment_mel_frames,
                mel=mel_segment,
                waveform=wave_segment,
            )
        )
        if len(selected) >= max_items:
            break

    if not selected:
        raise RuntimeError(
            f"No {split} vocoder segments satisfy {segment_mel_frames} mel frames"
        )
    return selected, skipped
