"""Tests for reclist and voicebank validation."""

from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from lykenox_voice_engine.core.reclist import load_reclist, missing_aliases
from lykenox_voice_engine.core.voicebank import TARGET_SAMPLE_RATE, VoicebankManager
from lykenox_voice_engine.models.notes import NoteEvent


class TestVoicebank(unittest.TestCase):
    """Validate sample voicebank coverage and rendering behavior."""

    def test_reclist_loads_unique_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reclist.txt"
            path.write_text("a\n# skip\na\nba\n", encoding="utf-8")
            reclist = load_reclist(path)
        self.assertEqual(reclist.aliases, ("a", "ba"))
        self.assertEqual(missing_aliases(["a", "mi"], {"a"}), ["mi"])

    def test_incomplete_voicebank_reports_missing_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _create_root(Path(temp_dir), ["bai", "la"])
            manager = VoicebankManager(root)
            report = manager.coverage_for("baila", [NoteEvent("bai", 60, 0.0, 0.5)])
        self.assertEqual(report.missing, ("bai",))
        self.assertEqual(report.coverage, 0.0)

    def test_render_with_complete_temporary_voicebank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _create_root(Path(temp_dir), ["bai", "la"])
            wav_dir = root / "profiles" / "lykenox" / "voicebank" / "wav"
            _write_wav(wav_dir / "bai.wav")
            _write_wav(wav_dir / "la.wav")
            manager = VoicebankManager(root)
            output = root / "outputs" / "job" / "vocal.wav"
            result = manager.render_to_path(
                "baila",
                [NoteEvent("bai", 60, 0.0, 0.1), NoteEvent("la", 62, 0.1, 0.1)],
                120,
                output,
            )
            self.assertTrue(output.exists())
            self.assertEqual(result["renderer"], "internal_concat_pcm")


def _create_root(root: Path, aliases: list[str]) -> Path:
    voicebank = root / "profiles" / "lykenox" / "voicebank"
    (voicebank / "wav").mkdir(parents=True)
    (root / "datasets" / "lykenox" / "voicebank_raw").mkdir(parents=True)
    (voicebank / "reclist.txt").write_text("\n".join(aliases) + "\n", encoding="utf-8")
    (voicebank / "oto.ini").write_text("", encoding="utf-8")
    return root


def _write_wav(path: Path, duration: float = 0.2) -> None:
    frames = bytearray()
    for index in range(int(TARGET_SAMPLE_RATE * duration)):
        value = int(math.sin(index / 12.0) * 8000)
        frames.extend(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(TARGET_SAMPLE_RATE)
        writer.writeframes(bytes(frames))


if __name__ == "__main__":
    unittest.main()

