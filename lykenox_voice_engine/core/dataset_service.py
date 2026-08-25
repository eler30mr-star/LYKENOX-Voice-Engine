"""Dataset import and validation service."""

from __future__ import annotations

import shutil
from pathlib import Path

from lykenox_voice_engine.audio.analysis import SUPPORTED_AUDIO_EXTENSIONS, AudioInfo, inspect_audio


class DatasetService:
    """Manage raw and prepared audio clips for one voice profile."""

    def __init__(self, datasets_dir: Path) -> None:
        self.datasets_dir = datasets_dir

    def raw_dir(self, profile_id: str) -> Path:
        """Return the raw dataset directory for a profile."""

        path = self.datasets_dir / profile_id / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def import_files(self, profile_id: str, paths: list[Path]) -> list[AudioInfo]:
        """Copy supported audio files into raw dataset storage and inspect them."""

        imported = []
        target_dir = self.raw_dir(profile_id)
        for source in paths:
            if source.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                imported.append(inspect_audio(source))
                continue
            target = target_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            imported.append(inspect_audio(target))
        return imported

    def list_raw(self, profile_id: str) -> list[AudioInfo]:
        """Inspect all raw files for a profile."""

        return [inspect_audio(path) for path in sorted(self.raw_dir(profile_id).iterdir()) if path.is_file()]
