"""LYKENOX speech dataset loader and mel cache.

Reads the engine-neutral prepared CSV manifests owned by LYKENOX and creates
reproducible mel features. No trainer-specific metadata becomes canonical.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from lykenox_voice_engine.audio.io import load_audio
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig


@dataclass(frozen=True)
class SpeechRow:
    utterance_id: str
    wav_path: Path
    text: str


class MelFeatureExtractor:
    """Deterministic audio -> log-mel frontend for LYKENOX Speech."""

    def __init__(self, config: LykenoxSpeechConfig) -> None:
        self.config = config
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.mel_bins,
            power=1.0,
        )

    def __call__(self, wav_path: Path) -> torch.Tensor:
        waveform, sample_rate = load_audio(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.config.sample_rate)
        peak = waveform.abs().max().clamp_min(1e-8)
        if peak > 1.0:
            waveform = waveform / peak
        mel = self.transform(waveform).squeeze(0).transpose(0, 1)
        return torch.log(torch.clamp(mel, min=1e-5))


class LykenoxSpeechDataset(Dataset[dict[str, object]]):
    """Speech dataset backed by LYKENOX CSV manifests and local feature cache."""

    CACHE_VERSION = "mel-v1"

    def __init__(self, csv_path: Path, cache_dir: Path, config: LykenoxSpeechConfig | None = None) -> None:
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or LykenoxSpeechConfig()
        self.frontend = SpanishTextFrontend()
        self.extractor = MelFeatureExtractor(self.config)
        self.rows = self._read_rows()

    def _read_rows(self) -> list[SpeechRow]:
        rows: list[SpeechRow] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                wav_path = Path(row["wav_path"])
                if not wav_path.is_absolute():
                    wav_path = (self.csv_path.parent / wav_path).resolve()
                rows.append(SpeechRow(row["utterance_id"], wav_path, row["text"]))
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        token_ids = torch.tensor(self.frontend.encode(row.text), dtype=torch.long)
        mel = self._load_or_create_mel(row)
        return {
            "utterance_id": row.utterance_id,
            "text": row.text,
            "token_ids": token_ids,
            "mel": mel,
            "wav_path": str(row.wav_path),
        }

    def _load_or_create_mel(self, row: SpeechRow) -> torch.Tensor:
        key_payload = {
            "version": self.CACHE_VERSION,
            "wav": str(row.wav_path),
            "size": row.wav_path.stat().st_size,
            "mtime_ns": row.wav_path.stat().st_mtime_ns,
            "config": self.config.to_dict(),
        }
        digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        cache_path = self.cache_dir / f"{row.utterance_id}-{digest}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        mel = self.extractor(row.wav_path).to(torch.float32).contiguous()
        torch.save(mel, cache_path)
        return mel


def uniform_bootstrap_durations(token_count: int, mel_frames: int) -> torch.Tensor:
    """Temporary plumbing-only duration target.

    This is NOT the production alignment strategy. It exists only so a real-data
    smoke test can exercise audio/text/model/backprop before a proper aligner is
    selected. Frames are distributed as evenly as possible across tokens.
    """
    if token_count < 1 or mel_frames < 1:
        raise ValueError("token_count and mel_frames must be positive")
    base, remainder = divmod(mel_frames, token_count)
    durations = torch.full((token_count,), base, dtype=torch.long)
    if remainder:
        durations[:remainder] += 1
    return durations
