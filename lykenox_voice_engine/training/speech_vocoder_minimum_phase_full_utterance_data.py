"""Full-utterance owned conditioning for minimum-phase vocoder evaluation.

This module exposes complete held-out utterances on the exact same full-utterance mel/pitch
frame grid used by the active owned vocoder data contract.  It exists so audible evaluation
is performed on complete validation utterances rather than tiny training crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import LykenoxVocoderConfig
from lykenox_voice_engine.training.speech_aligner_train import _dataset
from lykenox_voice_engine.training.speech_pitch_cache import (
    PITCH_CACHE_VERSION,
    load_indexed_pitch_target,
)
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    _mono_waveform,
)


FULL_UTTERANCE_DATA_VERSION = "owned-vocoder-full-utterance-v1"


@dataclass(frozen=True)
class OwnedVocoderUtterance:
    split: str
    utterance_id: str
    wav_path: str
    mel_frames: int
    mel: torch.Tensor
    f0_hz: torch.Tensor
    voiced: torch.Tensor
    periodicity: torch.Tensor
    waveform: torch.Tensor
    conditioning_contract_version: str = OWNED_VOCODER_SEGMENT_CONTRACT_VERSION
    pitch_cache_version: str = PITCH_CACHE_VERSION
    full_utterance_data_version: str = FULL_UTTERANCE_DATA_VERSION


def collect_owned_vocoder_utterances(
    root: Path,
    split: str,
    *,
    max_items: int,
) -> list[OwnedVocoderUtterance]:
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
    utterances: list[OwnedVocoderUtterance] = []

    for index in range(len(dataset)):
        item = dataset[index]
        utterance_id = str(item["utterance_id"])
        wav_path = Path(str(item["wav_path"]))
        mel = item["mel"].to(torch.float32).contiguous()
        pitch = load_indexed_pitch_target(
            root,
            split=split,
            utterance_id=utterance_id,
        )
        waveform = _mono_waveform(wav_path, vocoder_config)

        mel_frames = int(mel.shape[0])
        if (
            int(pitch.f0_hz.numel()) != mel_frames
            or int(pitch.voiced.numel()) != mel_frames
            or int(pitch.periodicity.numel()) != mel_frames
        ):
            raise RuntimeError(f"full-utterance mel/pitch mismatch for {utterance_id}")

        waveform_frames = int(waveform.numel()) // vocoder_config.hop_length
        usable_frames = min(mel_frames, waveform_frames)
        if usable_frames < 16:
            continue
        sample_count = usable_frames * vocoder_config.hop_length
        utterances.append(
            OwnedVocoderUtterance(
                split=split,
                utterance_id=utterance_id,
                wav_path=str(wav_path),
                mel_frames=usable_frames,
                mel=mel[:usable_frames].contiguous(),
                f0_hz=pitch.f0_hz[:usable_frames].to(torch.float32).contiguous(),
                voiced=pitch.voiced[:usable_frames].to(torch.float32).contiguous(),
                periodicity=pitch.periodicity[:usable_frames].to(torch.float32).contiguous(),
                waveform=waveform[:sample_count].to(torch.float32).contiguous(),
            )
        )
        if len(utterances) >= max_items:
            break

    if not utterances:
        raise RuntimeError(f"No usable owned full utterances in split={split}")
    return utterances


__all__ = [
    "FULL_UTTERANCE_DATA_VERSION",
    "OwnedVocoderUtterance",
    "collect_owned_vocoder_utterances",
]
