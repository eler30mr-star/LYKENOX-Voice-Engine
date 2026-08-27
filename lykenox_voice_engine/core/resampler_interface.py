"""UTAU-style external resampler and wavtool interface."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class UtauClassicRenderer:
    """Invokes UTAU resamplers (like resampler.exe, tips, moresampler) and wavtool."""

    def __init__(self, root: Path):
        self.root = root
        self.renderers_dir = root / "tools" / "renderers"
        self.renderers_dir.mkdir(parents=True, exist_ok=True)
        self.utau_bridge_dir = self.renderers_dir / "utau_basic_bridge"
        self.utau_bridge_dir.mkdir(parents=True, exist_ok=True)

        # Default search for common resamplers
        self.resampler = self._find_executable([
            "resampler.exe",
            "tips.exe",
            "moresampler.exe",
            "utau_basic_bridge/lykenox_utau_bridge.exe",
        ])
        self.wavtool = self._find_executable(["wavtool.exe", "append.exe"])

    def compile_worldline(self) -> bool:
        """Compile the local UTAU bridge using system csc.exe."""
        source = self.utau_bridge_dir / "LykenoxUtauBridge.cs"
        output = self.utau_bridge_dir / "lykenox_utau_bridge.exe"

        if not source.exists():
            return False

        # Common .NET Framework paths
        csc_paths = [
            Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe"),
            Path("C:/Windows/Microsoft.NET/Framework/v4.0.30319/csc.exe"),
        ]

        csc = next((p for p in csc_paths if p.exists()), None)
        if not csc:
            logger.error("csc.exe not found in standard .NET paths.")
            return False

        try:
            cmd = [str(csc), "/out:" + str(output), str(source)]
            subprocess.run(cmd, check=True, capture_output=True)
            if output.exists():
                self.resampler = output
                return True
        except Exception as e:
            logger.error(f"Failed to compile LYKENOX UTAU Bridge: {e}")

        return False

    def _find_executable(self, names: list[str]) -> Path | None:
        for name in names:
            path = self.renderers_dir / name
            if path.exists():
                return path
        return None

    def render_segment(self, input_wav: Path, output_wav: Path, note_midi: int,
                       length_ms: float, entry: Any, tempo: float) -> bool:
        """Call external resampler for a single alias segment."""
        if not self.resampler:
            return False

        # UTAU pitch format: C4 = 60, but UTAU often uses note names or relative pitch
        # Standard CLI: resampler <input> <output> <pitch> <velocity> <flags> <offset> <length> <consonant> <cutoff> <volume> <modulation> <tempo> <pitch_bend>

        note_name = self._midi_to_utau_pitch(note_midi)

        cmd = [
            str(self.resampler),
            str(input_wav),
            str(output_wav),
            note_name,
            "100",  # velocity
            "",     # flags
            str(entry.offset),
            str(length_ms),
            str(entry.consonant),
            str(entry.cutoff),
            "100",  # volume
            "0",    # modulation
            str(tempo)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_wav.exists()
        except Exception as e:
            logger.error(f"Error calling resampler: {e}")
            return False

    def _midi_to_utau_pitch(self, midi: int) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        name = names[midi % 12]
        return f"{name}{octave}"
