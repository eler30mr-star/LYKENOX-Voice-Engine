"""Dataset import and inspection service."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from lykenox_voice_engine.config.settings import PROJECT_ROOT

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus"}


@dataclass(frozen=True)
class DatasetItem:
    """Visible dataset row."""

    path: Path
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    peak: float = 0.0
    status: str = "raw"


class DatasetService:
    """Manage authorized singing datasets without destructive edits."""

    def __init__(self, profile_id: str = "lykenox") -> None:
        """Create dataset folders for one profile."""
        self.root = PROJECT_ROOT / "datasets" / profile_id
        self.raw_dir = self.root / "raw"
        self.clean_dir = self.root / "clean"
        self.processed_dir = self.root / "processed"
        self.metadata_dir = self.root / "metadata"
        for path in [self.raw_dir, self.clean_dir, self.processed_dir, self.metadata_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def import_files(self, files: list[str]) -> tuple[int, list[str]]:
        """Copy accepted audio files into raw without deleting originals."""
        copied = 0
        rejected: list[str] = []
        for value in files:
            source = Path(value)
            if not source.is_file() or source.suffix.lower() not in AUDIO_EXTENSIONS:
                rejected.append(str(source))
                continue
            shutil.copy2(source, self.raw_dir / self._unique_name(source.name))
            copied += 1
        return copied, rejected

    def list_items(self) -> list[DatasetItem]:
        """List raw dataset files."""
        return [DatasetItem(path=p) for p in sorted(self.raw_dir.iterdir()) if p.is_file()]

    def _unique_name(self, name: str) -> str:
        """Return a non-conflicting raw file name."""
        candidate = Path(name)
        index = 1
        while (self.raw_dir / candidate.name).exists():
            candidate = Path(f"{Path(name).stem}_{index}{Path(name).suffix}")
            index += 1
        return candidate.name
