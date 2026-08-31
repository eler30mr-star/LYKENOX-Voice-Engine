"""Deterministic real mel/waveform segments for persistent LYKENOX vocoder training.

The historical ``vocoder-segment-v1`` contract is retained for forensic reproducibility.
New LYKENOX vocoder work must use the owned v2 contract, where mel, F0 and voicing are all
sliced from their completed full-utterance feature caches on the same frame grid.  This
prevents crop-local pitch extraction from silently changing voicing semantics or reflected
analysis context at every training crop boundary.
"""

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
from lykenox_voice_engine.training.speech_pitch import PitchFrames
from lykenox_voice_engine.training.speech_pitch_cache import (
    PITCH_CACHE_VERSION,
    load_indexed_pitch_target,
)


# Historical contract. Do not silently reinterpret rejected/accepted old artifacts.
VOCODER_SEGMENT_CONTRACT_VERSION = "vocoder-segment-v1"

# Active contract for any future LYKENOX-owned vocoder work.
OWNED_VOCODER_SEGMENT_CONTRACT_VERSION = (
    "vocoder-segment-v2-full-utterance-mel-pitch-conditioning"
)


@dataclass(frozen=True)
class VocoderSegment:
    """Historical v1 mel/wave segment without versioned pitch conditioning."""

    split: str
    utterance_id: str
    wav_path: str
    start_frame: int
    mel_frames: int
    mel: torch.Tensor
    waveform: torch.Tensor


@dataclass(frozen=True)
class OwnedVocoderSegment:
    """V2 segment with frame-exact conditioning sliced from owned utterance caches."""

    split: str
    utterance_id: str
    wav_path: str
    start_frame: int
    mel_frames: int
    mel: torch.Tensor
    f0_hz: torch.Tensor
    voiced: torch.Tensor
    periodicity: torch.Tensor
    waveform: torch.Tensor
    conditioning_contract_version: str = OWNED_VOCODER_SEGMENT_CONTRACT_VERSION
    pitch_cache_version: str = PITCH_CACHE_VERSION


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


def _slice_owned_segment(
    *,
    split: str,
    utterance_id: str,
    wav_path: Path,
    mel: torch.Tensor,
    waveform: torch.Tensor,
    pitch: PitchFrames,
    start_frame: int,
    segment_mel_frames: int,
    hop_length: int,
) -> OwnedVocoderSegment:
    """Slice mel/pitch/waveform from one shared full-utterance frame origin."""

    full_frames = int(mel.shape[0])
    if (
        int(pitch.f0_hz.numel()) != full_frames
        or int(pitch.voiced.numel()) != full_frames
        or int(pitch.periodicity.numel()) != full_frames
    ):
        raise RuntimeError(
            f"Full-utterance mel/pitch frame mismatch for {utterance_id}"
        )
    if start_frame < 0 or start_frame + segment_mel_frames > full_frames:
        raise ValueError("owned segment frame range is outside full utterance")

    start_sample = int(start_frame) * int(hop_length)
    sample_count = int(segment_mel_frames) * int(hop_length)
    end_frame = int(start_frame) + int(segment_mel_frames)
    mel_segment = mel[start_frame:end_frame].to(torch.float32).contiguous()
    f0_segment = pitch.f0_hz[start_frame:end_frame].to(torch.float32).contiguous()
    voiced_segment = pitch.voiced[start_frame:end_frame].to(torch.float32).contiguous()
    periodicity_segment = (
        pitch.periodicity[start_frame:end_frame].to(torch.float32).contiguous()
    )
    wave_segment = waveform[start_sample : start_sample + sample_count].to(
        torch.float32
    ).contiguous()

    expected_frames = int(segment_mel_frames)
    if tuple(mel_segment.shape[:1]) != (expected_frames,):
        raise RuntimeError(f"Mel segment length contract failed for {utterance_id}")
    for name, values in (
        ("f0", f0_segment),
        ("voiced", voiced_segment),
        ("periodicity", periodicity_segment),
    ):
        if tuple(values.shape) != (expected_frames,):
            raise RuntimeError(
                f"{name} segment length contract failed for {utterance_id}"
            )
    if int(wave_segment.numel()) != sample_count:
        raise RuntimeError(f"Waveform segment length contract failed for {utterance_id}")

    return OwnedVocoderSegment(
        split=split,
        utterance_id=utterance_id,
        wav_path=str(wav_path),
        start_frame=int(start_frame),
        mel_frames=expected_frames,
        mel=mel_segment,
        f0_hz=f0_segment,
        voiced=voiced_segment,
        periodicity=periodicity_segment,
        waveform=wave_segment,
    )


def collect_vocoder_segments(
    root: Path,
    split: str,
    *,
    segment_mel_frames: int,
    max_items: int,
    seed: int = 1337,
    boundary_margin_frames: int = 4,
) -> tuple[list[VocoderSegment], list[dict[str, object]]]:
    """Collect historical v1 mel/wave pairs for forensic reproducibility only.

    V1 does not carry the approved full-utterance pitch cache. Historical trainers later
    re-extracted pitch from each waveform crop, which changes the context and RMS-relative
    voicing threshold. New vocoder work must call ``collect_owned_vocoder_segments``.
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


def collect_owned_vocoder_segments(
    root: Path,
    split: str,
    *,
    segment_mel_frames: int,
    max_items: int,
    seed: int = 1337,
    boundary_margin_frames: int = 4,
) -> tuple[list[OwnedVocoderSegment], list[dict[str, object]]]:
    """Collect v2 segments using one full-utterance conditioning authority.

    Mel comes from the versioned speech mel cache and F0/voicing/periodicity come from the
    completed owned pitch cache used by the acoustic model. They are sliced with exactly
    the same ``start_frame``. No pitch analysis is rerun on the waveform crop.
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
    selected: list[OwnedVocoderSegment] = []
    skipped: list[dict[str, object]] = []

    for index in range(len(dataset)):
        item = dataset[index]
        utterance_id = str(item["utterance_id"])
        mel = item["mel"].to(torch.float32).contiguous()
        wav_path = Path(str(item["wav_path"]))
        waveform = _mono_waveform(wav_path, vocoder_config)
        pitch = load_indexed_pitch_target(
            root,
            split=split,
            utterance_id=utterance_id,
        )

        mel_frames = int(mel.shape[0])
        pitch_frames = int(pitch.f0_hz.numel())
        waveform_frames = int(waveform.numel()) // vocoder_config.hop_length
        if pitch_frames != mel_frames:
            raise RuntimeError(
                f"Owned pitch/mel frame mismatch for {utterance_id}: "
                f"{pitch_frames} != {mel_frames}"
            )
        usable_frames = min(mel_frames, waveform_frames)
        if usable_frames < segment_mel_frames:
            skipped.append(
                {
                    "utterance_id": utterance_id,
                    "reason": "too_short",
                    "mel_frames": mel_frames,
                    "pitch_frames": pitch_frames,
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
        selected.append(
            _slice_owned_segment(
                split=split,
                utterance_id=utterance_id,
                wav_path=wav_path,
                mel=mel,
                waveform=waveform,
                pitch=pitch,
                start_frame=start_frame,
                segment_mel_frames=segment_mel_frames,
                hop_length=vocoder_config.hop_length,
            )
        )
        if len(selected) >= max_items:
            break

    if not selected:
        raise RuntimeError(
            f"No {split} owned vocoder segments satisfy {segment_mel_frames} mel frames"
        )
    return selected, skipped
