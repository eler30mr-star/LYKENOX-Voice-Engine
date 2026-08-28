"""Validated aligned speech dataset and padded batch contract for LYKENOX Speech.

Long acoustic training must consume only a clean alignment-v3 timing cache. This module
centralizes that boundary so trainers do not reimplement ad-hoc duration loading, padding,
or masking logic. Frame-level F0/voicing targets are optional and, when requested, are
loaded only through the completed versioned pitch cache index.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_aligner_train import _dataset
from lykenox_voice_engine.training.speech_duration_cache import DURATION_CACHE_VERSION
from lykenox_voice_engine.training.speech_pitch_cache import load_indexed_pitch_target


EXPECTED_DURATION_CACHE_VERSION = "alignment-v3"


@dataclass(frozen=True)
class AlignedSpeechBatch:
    """One padded training batch with explicit token and mel validity masks."""

    utterance_ids: list[str]
    texts: list[str]
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    durations: torch.Tensor
    mel: torch.Tensor
    mel_mask: torch.Tensor
    token_lengths: torch.Tensor
    mel_lengths: torch.Tensor
    f0_hz: torch.Tensor | None = None
    voiced: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "AlignedSpeechBatch":
        return AlignedSpeechBatch(
            utterance_ids=self.utterance_ids,
            texts=self.texts,
            token_ids=self.token_ids.to(device),
            token_mask=self.token_mask.to(device),
            durations=self.durations.to(device),
            mel=self.mel.to(device),
            mel_mask=self.mel_mask.to(device),
            token_lengths=self.token_lengths.to(device),
            mel_lengths=self.mel_lengths.to(device),
            f0_hz=None if self.f0_hz is None else self.f0_hz.to(device),
            voiced=None if self.voiced is None else self.voiced.to(device),
        )


def find_clean_duration_root(root: Path) -> Path:
    """Return the newest alignment-v3 cache that passed the duration audit."""

    if DURATION_CACHE_VERSION != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError(
            "The active LYKENOX duration-cache contract is not alignment-v3"
        )
    base = (
        Path(root).resolve()
        / "datasets"
        / "lykenox"
        / "identity_voice"
        / "features"
        / "speech"
        / EXPECTED_DURATION_CACHE_VERSION
    )
    reports = sorted(
        base.rglob("duration_audit.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for report_path in reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            report.get("status") == "pass"
            and report.get("duration_cache_version") == EXPECTED_DURATION_CACHE_VERSION
            and int(report.get("suspicious_utterance_count", 0)) == 0
            and bool(report.get("splits", {}).get("train", {}).get("pass", False))
            and bool(report.get("splits", {}).get("val", {}).get("pass", False))
        ):
            return report_path.parent
    raise FileNotFoundError(
        f"No clean {EXPECTED_DURATION_CACHE_VERSION} duration cache found under {base}"
    )


def duration_record_paths(duration_root: Path, split: str) -> dict[str, Path]:
    """Load the deterministic utterance -> timing-record index for one split."""

    index_path = Path(duration_root) / split / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Duration index not found: {index_path}")
    records: dict[str, Path] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        utterance_id = str(row["utterance_id"])
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = (index_path.parent / path).resolve()
        records[utterance_id] = path
    return records


def validate_aligned_record(
    record: object,
    *,
    utterance_id: str,
    text: str,
    token_ids: torch.Tensor,
    mel_frames: int,
) -> torch.Tensor:
    """Validate one alignment-v3 timing record and return teacher durations."""

    if not isinstance(record, dict):
        raise RuntimeError(f"Invalid aligned duration record for {utterance_id}")
    if record.get("cache_version") != EXPECTED_DURATION_CACHE_VERSION:
        raise RuntimeError(
            f"Wrong duration cache version for {utterance_id}: {record.get('cache_version')}"
        )
    if str(record.get("utterance_id")) != utterance_id:
        raise RuntimeError(f"Duration record utterance mismatch for {utterance_id}")
    if str(record.get("text", "")) != text:
        raise RuntimeError(f"Duration record text mismatch for {utterance_id}")

    expected_tokens = [int(value) for value in token_ids.detach().cpu().tolist()]
    cached_tokens = [int(value) for value in record.get("token_ids", [])]
    if cached_tokens != expected_tokens:
        raise RuntimeError(f"Duration record token mismatch for {utterance_id}")

    values = [int(value) for value in record.get("durations", [])]
    if len(values) != len(expected_tokens):
        raise RuntimeError(f"Duration/token length mismatch for {utterance_id}")
    if any(value < 0 for value in values):
        raise RuntimeError(f"Negative teacher duration for {utterance_id}")
    if sum(values) != int(mel_frames):
        raise RuntimeError(
            f"Duration sum mismatch for {utterance_id}: {sum(values)} != {mel_frames}"
        )
    return torch.tensor(values, dtype=torch.long)


class LykenoxAlignedSpeechDataset(Dataset[dict[str, object]]):
    """Real mel/text examples paired with exact cleaned alignment-v3 durations.

    Set ``include_pitch_targets=True`` only after ``speech-pitch-cache-v1`` has passed.
    Pitch targets are then loaded by utterance ID through its hashed completed index;
    this dataset never re-runs waveform pitch extraction.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        config: LykenoxSpeechConfig,
        *,
        duration_root: Path | None = None,
        include_pitch_targets: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.split = split
        self.config = config
        self.include_pitch_targets = bool(include_pitch_targets)
        self.duration_root = (
            Path(duration_root).resolve()
            if duration_root is not None
            else find_clean_duration_root(self.root)
        )
        self.base = _dataset(self.root, split, config)
        self.record_paths = duration_record_paths(self.duration_root, split)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.base[index]
        utterance_id = str(item["utterance_id"])
        record_path = self.record_paths.get(utterance_id)
        if record_path is None or not record_path.exists():
            raise RuntimeError(f"Missing alignment-v3 record for {utterance_id}")
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        token_ids = item["token_ids"].to(torch.long)
        mel = item["mel"].to(torch.float32)
        durations = validate_aligned_record(
            record,
            utterance_id=utterance_id,
            text=str(item["text"]),
            token_ids=token_ids,
            mel_frames=int(mel.shape[0]),
        )
        result: dict[str, object] = {
            "utterance_id": utterance_id,
            "text": str(item["text"]),
            "token_ids": token_ids,
            "durations": durations,
            "mel": mel,
        }
        if self.include_pitch_targets:
            pitch = load_indexed_pitch_target(
                self.root,
                split=self.split,
                utterance_id=utterance_id,
            )
            if int(pitch.f0_hz.numel()) != int(mel.shape[0]):
                raise RuntimeError(
                    f"Pitch/mel length mismatch for {utterance_id}: "
                    f"{pitch.f0_hz.numel()} != {mel.shape[0]}"
                )
            result["f0_hz"] = pitch.f0_hz.to(torch.float32)
            result["voiced"] = pitch.voiced.to(torch.float32)
        return result


def collate_aligned_speech(
    items: list[dict[str, object]],
    *,
    pad_id: int = 0,
) -> AlignedSpeechBatch:
    """Pad variable-length examples and emit masks for every padded dimension."""

    if not items:
        raise ValueError("Cannot collate an empty aligned speech batch")
    batch_size = len(items)
    token_lengths = torch.tensor(
        [int(item["token_ids"].shape[0]) for item in items],
        dtype=torch.long,
    )
    mel_lengths = torch.tensor(
        [int(item["mel"].shape[0]) for item in items],
        dtype=torch.long,
    )
    max_tokens = int(token_lengths.max().item())
    max_mel = int(mel_lengths.max().item())
    mel_bins = int(items[0]["mel"].shape[1])

    pitch_flags = ["f0_hz" in item or "voiced" in item for item in items]
    if any(pitch_flags) and not all(pitch_flags):
        raise RuntimeError("A batch cannot mix items with and without pitch targets")
    include_pitch = all(pitch_flags)

    token_ids = torch.full((batch_size, max_tokens), pad_id, dtype=torch.long)
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    durations = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    mel = torch.zeros((batch_size, max_mel, mel_bins), dtype=torch.float32)
    mel_mask = torch.zeros((batch_size, max_mel), dtype=torch.bool)
    f0_hz = (
        torch.zeros((batch_size, max_mel), dtype=torch.float32)
        if include_pitch
        else None
    )
    voiced = (
        torch.zeros((batch_size, max_mel), dtype=torch.float32)
        if include_pitch
        else None
    )

    utterance_ids: list[str] = []
    texts: list[str] = []
    for batch_index, item in enumerate(items):
        ids = item["token_ids"].to(torch.long)
        teacher = item["durations"].to(torch.long)
        target = item["mel"].to(torch.float32)
        if teacher.shape != ids.shape:
            raise RuntimeError("Teacher durations must match token IDs before padding")
        if int(teacher.sum().item()) != int(target.shape[0]):
            raise RuntimeError("Teacher duration sum must equal mel frames before padding")
        if int(target.shape[1]) != mel_bins:
            raise RuntimeError("All items in a speech batch must have the same mel bin count")

        token_count = int(ids.shape[0])
        mel_count = int(target.shape[0])
        token_ids[batch_index, :token_count] = ids
        token_mask[batch_index, :token_count] = True
        durations[batch_index, :token_count] = teacher
        mel[batch_index, :mel_count] = target
        mel_mask[batch_index, :mel_count] = True

        if include_pitch:
            assert f0_hz is not None and voiced is not None
            item_f0 = item["f0_hz"].to(torch.float32)
            item_voiced = item["voiced"].to(torch.float32)
            if item_f0.shape != (mel_count,) or item_voiced.shape != (mel_count,):
                raise RuntimeError("Pitch targets must match mel frames before padding")
            if not bool((item_f0[item_voiced < 0.5] == 0.0).all()):
                raise RuntimeError("Unvoiced cached F0 must be exactly zero")
            f0_hz[batch_index, :mel_count] = item_f0
            voiced[batch_index, :mel_count] = item_voiced

        utterance_ids.append(str(item["utterance_id"]))
        texts.append(str(item["text"]))

    if not torch.equal(durations.sum(dim=1), mel_lengths):
        raise RuntimeError("Padded teacher durations do not preserve per-item mel lengths")

    return AlignedSpeechBatch(
        utterance_ids=utterance_ids,
        texts=texts,
        token_ids=token_ids,
        token_mask=token_mask,
        durations=durations,
        mel=mel,
        mel_mask=mel_mask,
        token_lengths=token_lengths,
        mel_lengths=mel_lengths,
        f0_hz=f0_hz,
        voiced=voiced,
    )
