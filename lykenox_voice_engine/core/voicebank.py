"""Voicebank validation and rendering for the local sample-based backend."""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import tempfile
import time
import wave
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lykenox_voice_engine.core.oto import OtoEntry, estimate_oto_entry, parse_oto, write_oto
from lykenox_voice_engine.core.pcm import peak, rms
from lykenox_voice_engine.core.reclist import Reclist, load_reclist, missing_aliases
from lykenox_voice_engine.core.spanish_phonemizer import SpanishPhonemizer
from lykenox_voice_engine.models.notes import NoteEvent

TARGET_SAMPLE_RATE = 48_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2


@dataclass(frozen=True)
class WavQuality:
    """Audio quality checks for one accepted voicebank WAV."""

    valid: bool
    sample_rate: int
    channels: int
    sample_width_bits: int
    duration_sec: float
    rms: int
    clipping: bool
    reason: str | None = None


@dataclass(frozen=True)
class CoverageReport:
    """Alias coverage before synthesis."""

    required: tuple[str, ...]
    available: tuple[str, ...]
    missing: tuple[str, ...]
    coverage: float


class VoicebankManager:
    """Manage LYKENOX Spanish Lite voicebank assets and synthesis."""

    def __init__(self, root: Path, profile: str = "lykenox") -> None:
        self.root = root
        self.profile = profile
        self.voicebank_dir = root / "profiles" / profile / "voicebank"
        self.wav_dir = self.voicebank_dir / "wav"
        self.raw_dir = root / "datasets" / profile / "voicebank_raw"
        self.reclist_path = self.voicebank_dir / "reclist.txt"
        self.oto_path = self.voicebank_dir / "oto.ini"
        self.phonemizer = SpanishPhonemizer()

    def load_reclist(self) -> Reclist:
        """Load the configured Spanish Lite reclist."""

        return load_reclist(self.reclist_path)

    def build_voicebank(self) -> dict[str, Any]:
        """Copy accepted recordings to wav/ and generate initial oto.ini entries."""

        self.wav_dir.mkdir(parents=True, exist_ok=True)
        entries: list[OtoEntry] = []
        accepted = 0
        rejected: dict[str, str] = {}
        for alias in self.load_reclist().aliases:
            source = self.raw_dir / f"{alias}.wav"
            if not source.exists():
                continue
            quality = validate_wav(source)
            if not quality.valid:
                rejected[alias] = quality.reason or "WAV invalido"
                continue
            target = self.wav_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            entries.append(estimate_oto_entry(target, alias))
            accepted += 1
        write_oto(self.oto_path, entries)
        write_voicebank_manifest(self.root, self.profile)
        return {"accepted": accepted, "rejected": rejected, "oto": str(self.oto_path)}

    def validate_voicebank(self) -> dict[str, Any]:
        """Validate structure, WAV quality, OTO timing, and coverage."""

        reclist = self.load_reclist()
        oto = parse_oto(self.oto_path)
        available = self.available_aliases()
        wav_quality = {alias: validate_wav(self.wav_dir / f"{alias}.wav") for alias in available}
        invalid = {alias: quality.reason for alias, quality in wav_quality.items() if not quality.valid}
        missing_from_reclist = missing_aliases(list(reclist.aliases), available)
        coverage = _coverage_percent(len(reclist.aliases) - len(missing_from_reclist), len(reclist.aliases))
        return {
            "voicebank": str(self.voicebank_dir),
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": TARGET_CHANNELS,
            "sample_width_bits": TARGET_SAMPLE_WIDTH * 8,
            "reclist_count": len(reclist.aliases),
            "available_count": len(available),
            "voicebank_available": not invalid and coverage == 100.0,
            "voicebank_coverage": coverage,
            "missing_aliases": missing_from_reclist,
            "invalid_wav": invalid,
            "oto_entries": len(oto),
        }

    def required_aliases(self, lyrics: str, notes: list[NoteEvent]) -> list[str]:
        """Return renderer aliases from note lyrics or phonemized lyrics."""

        note_aliases = [note.lyric.strip().lower() for note in notes if note.lyric.strip()]
        if note_aliases:
            return [resolved for alias in note_aliases for resolved in self._aliases_for_note(alias)]
        return list(self.phonemizer.phonemize(lyrics).aliases)

    def _aliases_for_note(self, lyric: str) -> list[str]:
        """Resolve an API note lyric to one or more practical aliases."""

        available = self.available_aliases()
        if lyric in available:
            return [lyric]
        aliases = list(self.phonemizer.phonemize(lyric).aliases)
        return aliases or [lyric]

    def coverage_for(self, lyrics: str, notes: list[NoteEvent]) -> CoverageReport:
        """Compare required synthesis aliases against recorded voicebank aliases."""

        required = self.required_aliases(lyrics, notes)
        available = self.available_aliases()
        missing = missing_aliases(required, available)
        coverage = _coverage_percent(len(set(required)) - len(missing), len(set(required)))
        return CoverageReport(
            required=tuple(required),
            available=tuple(sorted(available)),
            missing=tuple(missing),
            coverage=coverage,
        )

    def available_aliases(self) -> set[str]:
        """Return aliases with WAV files present in the voicebank."""

        if not self.wav_dir.exists():
            return set()
        return {path.stem.lower() for path in self.wav_dir.glob("*.wav")}

    def recorded_aliases(self) -> set[str]:
        """Return aliases with raw recordings present in the dataset."""

        if not self.raw_dir.exists():
            return set()
        return {path.stem.lower() for path in self.raw_dir.glob("*.wav") if not path.name.startswith("temp_")}

    def render_to_path(self, lyrics: str, notes: list[NoteEvent], tempo: int, output_path: Path, renderer_type: str = "internal") -> dict[str, Any]:
        """Render notes using UTAU-style timing and selected renderer."""

        started = time.perf_counter()
        if tempo <= 0:
            raise RuntimeError("Tempo invalido")

        oto = parse_oto(self.oto_path)
        coverage = self.coverage_for(lyrics, notes)
        if coverage.missing:
            raise RuntimeError("Voicebank incompleto. Faltan aliases: " + ", ".join(coverage.missing))

        if not notes:
            notes = _notes_from_aliases(list(coverage.required))

        if renderer_type == "worldline_real":
            from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine

            engine = OpenUtauWorldlineEngine(self.root, TARGET_SAMPLE_RATE)
            report = engine.render_to_path(
                self.wav_dir,
                oto,
                self._aliases_for_note,
                notes,
                tempo,
                output_path,
            )
            return {
                "output_path": str(output_path),
                "duration_sec": round(report.duration_sec, 3),
                "render_time_sec": round(report.render_time_sec, 3),
                "coverage": coverage.coverage,
                "renderer": renderer_type,
                "worldline": report,
            }

        # Configurar el motor de renderizado
        engine = UtauEngine(self.wav_dir, oto, TARGET_SAMPLE_RATE, self._aliases_for_note, self.root)
        engine.renderer_type = renderer_type

        # Generar audio final
        audio_data = engine.render(notes, tempo)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(TARGET_CHANNELS)
            writer.setsampwidth(TARGET_SAMPLE_WIDTH)
            writer.setframerate(TARGET_SAMPLE_RATE)
            writer.writeframes(audio_data)

        return {
            "output_path": str(output_path),
            "duration_sec": round(len(audio_data) / TARGET_SAMPLE_WIDTH / TARGET_SAMPLE_RATE, 3),
            "render_time_sec": round(time.perf_counter() - started, 3),
            "coverage": coverage.coverage,
            "renderer": renderer_type,
        }

    def test_renderer(self, alias: str = "a", output_dir: Path | None = None) -> Path:
        """Verify rendering of a single vowel at different pitches (60, 64, 57)."""

        if output_dir is None:
            output_dir = self.root / "tests" / "render_test"
        output_dir.mkdir(parents=True, exist_ok=True)

        test_notes = [
            NoteEvent(alias, 60, 0.0, 1.0),
            NoteEvent(alias, 64, 1.5, 1.0),
            NoteEvent(alias, 57, 3.0, 1.0),
        ]

        out_path = output_dir / f"test_{alias}_pitches.wav"
        self.render_to_path("", test_notes, 120, out_path)
        return out_path

    def microtest_status(self) -> dict[str, Any]:
        """Return a small honest render readiness report for 'baila conmigo'."""

        notes = [
            NoteEvent("bai", 60, 0.0, 0.45),
            NoteEvent("la", 62, 0.45, 0.45),
            NoteEvent("con", 64, 0.9, 0.45),
            NoteEvent("mi", 62, 1.35, 0.45),
            NoteEvent("go", 60, 1.8, 0.6),
        ]
        coverage = self.coverage_for("baila conmigo", notes)
        return {
            "phrase": "baila conmigo",
            "required": list(coverage.required),
            "missing": list(coverage.missing),
            "coverage": coverage.coverage,
            "can_render": not coverage.missing,
        }


class UtauEngine:
    """Core synthesis engine implementing UTAU timing, pitch shifting, and time stretching."""

    def __init__(self, wav_dir: Path, oto: dict[str, OtoEntry], sample_rate: int, alias_resolver: Callable[[str], list[str]], root: Path):
        self.wav_dir = wav_dir
        self.oto = oto
        self.sample_rate = sample_rate
        self.alias_resolver = alias_resolver
        self.root = root
        self.ffmpeg_path = shutil.which("ffmpeg")
        self.ms_to_f = sample_rate / 1000.0
        self.renderer_type = "internal"  # "internal" o "classic"

        from lykenox_voice_engine.core.resampler_interface import UtauClassicRenderer
        self.classic = UtauClassicRenderer(root)

    def render(self, notes: list[NoteEvent], tempo: float = 120.0) -> bytes:
        """Render a sequence of notes into a single PCM buffer."""

        if not notes:
            return b""

        last_note = notes[-1]
        total_duration_sec = last_note.start + last_note.duration + 1.0
        total_frames = int(total_duration_sec * self.sample_rate)

        mix_buffer = array('f', [0.0] * total_frames)
        prev_midi = notes[0].midi

        for i, note in enumerate(notes):
            aliases = self.alias_resolver(note.lyric.strip().lower())
            if not aliases: continue

            duration_per_alias = note.duration / len(aliases)

            for j, alias in enumerate(aliases):
                entry = self.oto.get(alias)
                if not entry: continue
                wav_path = self.wav_dir / entry.wav
                if not wav_path.exists(): continue

                alias_start_abs = note.start + (j * duration_per_alias)
                audio_start_sec = alias_start_abs - (entry.preutterance / 1000.0)
                audio_start_frame = int(audio_start_sec * self.sample_rate)

                # Calcular fin del segmento
                if j + 1 < len(aliases):
                    next_alias = aliases[j+1]
                    next_entry = self.oto.get(next_alias)
                    end_sec = alias_start_abs + duration_per_alias - (next_entry.preutterance / 1000.0) + (next_entry.overlap / 1000.0) if next_entry else alias_start_abs + duration_per_alias
                elif i + 1 < len(notes):
                    next_note = notes[i+1]
                    next_aliases = self.alias_resolver(next_note.lyric.strip().lower())
                    next_entry = self.oto.get(next_aliases[0]) if next_aliases else None
                    end_sec = next_note.start - (next_entry.preutterance / 1000.0) + (next_entry.overlap / 1000.0) if next_entry else next_note.start
                else:
                    end_sec = alias_start_abs + duration_per_alias + (entry.overlap / 1000.0)

                target_total_duration_ms = max(10.0, (end_sec - audio_start_sec) * 1000.0)

                if self.renderer_type == "classic" and self.classic.resampler:
                    segment_audio = self._process_note_classic(wav_path, entry, note.midi, target_total_duration_ms, tempo)
                else:
                    segment_audio = self._process_note_internal(wav_path, entry, note.midi, prev_midi, target_total_duration_ms)

                self._mix(mix_buffer, segment_audio, audio_start_frame, entry.overlap)
                prev_midi = note.midi

        return self._finalize_audio(mix_buffer)

    def _process_note_classic(self, wav_path: Path, entry: OtoEntry, target_midi: int, target_duration_ms: float, tempo: float) -> array:
        """Process note using external resampler."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        success = self.classic.render_segment(wav_path, tmp_path, target_midi, target_duration_ms, entry, tempo)

        if success:
            with wave.open(str(tmp_path), "rb") as r:
                width = r.getsampwidth()
                frames = r.readframes(r.getnframes())
                count = len(frames) // width
                pcm = array('f', [float(x) for x in struct.unpack(f"<{count}h", frames)])
            try: tmp_path.unlink()
            except: pass
            return pcm

        return array('f', [])

    def _process_note_internal(self, wav_path: Path, entry: OtoEntry, target_midi: int, prev_midi: int, target_duration_ms: float) -> array:
        with wave.open(str(wav_path), "rb") as reader:
            sample_rate = reader.getframerate()
            width = reader.getsampwidth()
            raw_frames = reader.readframes(reader.getnframes())
            duration_ms = (len(raw_frames) / width / sample_rate) * 1000.0

        fixed_start_ms = entry.offset
        fixed_end_ms = entry.offset + entry.consonant
        vowel_start_ms = fixed_end_ms
        vowel_end_ms = duration_ms + entry.cutoff if entry.cutoff < 0 else (entry.offset + entry.cutoff if entry.cutoff > 0 else duration_ms)

        base_midi = self._detect_base_pitch(entry.alias, wav_path.name)
        pitch_ratio = 2.0 ** ((target_midi - base_midi) / 12.0)

        fixed_duration_ms = entry.consonant
        target_vowel_duration_ms = max(1.0, target_duration_ms - fixed_duration_ms)
        source_vowel_duration_ms = max(1.0, vowel_end_ms - vowel_start_ms)
        vowel_stretch_ratio = target_vowel_duration_ms / source_vowel_duration_ms

        fixed_pcm = self._extract_pcm(raw_frames, width, sample_rate, fixed_start_ms, fixed_end_ms)
        vowel_pcm = self._extract_pcm(raw_frames, width, sample_rate, vowel_start_ms, vowel_end_ms)

        fixed_processed = self._resample(fixed_pcm, pitch_ratio, 1.0)

        glide_ms = 50.0
        if abs(target_midi - prev_midi) > 0.1 and target_vowel_duration_ms > glide_ms * 2:
            glide_target_ms = glide_ms
            glide_source_ms = glide_target_ms / vowel_stretch_ratio
            vowel_glide_pcm = vowel_pcm[:int(glide_source_ms * self.ms_to_f)]
            vowel_steady_pcm = vowel_pcm[int(glide_source_ms * self.ms_to_f):]
            avg_glide_midi = (prev_midi + target_midi) / 2.0
            glide_pitch_ratio = 2.0 ** ((avg_glide_midi - base_midi) / 12.0)
            vowel_processed = self._resample(vowel_glide_pcm, glide_pitch_ratio, vowel_stretch_ratio) + self._resample(vowel_steady_pcm, pitch_ratio, vowel_stretch_ratio)
        else:
            vowel_processed = self._resample(vowel_pcm, pitch_ratio, vowel_stretch_ratio)

        return array('f', fixed_processed + vowel_processed)

    def _resample(self, pcm_data: list[float], pitch_ratio: float, stretch_ratio: float) -> list[float]:
        if not pcm_data: return []
        if abs(pitch_ratio - 1.0) < 0.001 and abs(stretch_ratio - 1.0) < 0.001: return pcm_data

        if self.ffmpeg_path:
            input_bytes = struct.pack(f"<{len(pcm_data)}h", *[int(max(-32768, min(32767, x))) for x in pcm_data])
            tempo_val = 1.0 / (pitch_ratio * stretch_ratio)
            atempo_filters = []
            tmp_tempo = tempo_val
            while tmp_tempo > 2.0: atempo_filters.append("atempo=2.0"); tmp_tempo /= 2.0
            while tmp_tempo < 0.5: atempo_filters.append("atempo=0.5"); tmp_tempo /= 0.5
            atempo_filters.append(f"atempo={tmp_tempo:.4f}")
            filters = f"asetrate={int(self.sample_rate * pitch_ratio)},{','.join(atempo_filters)},aresample={self.sample_rate}"
            cmd = [self.ffmpeg_path, "-f", "s16le", "-ar", str(self.sample_rate), "-ac", "1", "-i", "pipe:0", "-af", filters, "-f", "s16le", "-ac", "1", "-ar", str(self.sample_rate), "pipe:1"]
            try:
                process = subprocess.run(cmd, input=input_bytes, capture_output=True, check=True)
                output_len = len(process.stdout) // 2
                return list(struct.unpack(f"<{output_len}h", process.stdout))
            except Exception: pass

        source_len = len(pcm_data)
        new_len = int(source_len / pitch_ratio)
        resampled = []
        for i in range(new_len):
            pos = i * pitch_ratio
            idx = int(pos)
            frac = pos - idx
            if idx + 1 < source_len: resampled.append(pcm_data[idx] * (1 - frac) + pcm_data[idx + 1] * frac)
            else: resampled.append(pcm_data[idx] if idx < source_len else 0)
        target_len = int(len(resampled) * stretch_ratio)
        if target_len > len(resampled): return resampled + [resampled[-1]] * (target_len - len(resampled))
        return resampled[:target_len]

    def _extract_pcm(self, raw_frames: bytes, width: int, sample_rate: int, start_ms: float, end_ms: float) -> list[float]:
        start_f = int(max(0, start_ms) * (sample_rate / 1000.0))
        end_f = int(min(len(raw_frames) // width, end_ms * (sample_rate / 1000.0)))
        if start_f >= end_f: return []
        segment = raw_frames[start_f * width : end_f * width]
        count = len(segment) // width
        if width == 2: return [float(x) for x in struct.unpack(f"<{count}h", segment)]
        return [0.0] * count

    def _mix(self, buffer: array, segment: array, start_frame: int, overlap_ms: float):
        overlap_frames = int(overlap_ms * self.ms_to_f)
        for i in range(len(segment)):
            target_idx = start_frame + i
            if target_idx < 0 or target_idx >= len(buffer): continue
            if i < overlap_frames and target_idx > 0:
                alpha = i / overlap_frames
                buffer[target_idx] = buffer[target_idx] * (1.0 - alpha) + segment[i] * alpha
            else: buffer[target_idx] = segment[i]

    def _detect_base_pitch(self, alias: str, filename: str) -> int:
        for text in [alias, filename]:
            match = re.search(r'_?([A-G]#?\d)', text, re.IGNORECASE)
            if match:
                note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                note_str = match.group(1).upper()
                name = note_str[0]
                accidental = 1 if '#' in note_str else 0
                octave = int(note_str[-1])
                return (octave + 1) * 12 + note_map[name] + accidental
        return 60

    def _finalize_audio(self, buffer: array) -> bytes:
        output = array('h', [0] * len(buffer))
        for i in range(len(buffer)):
            val = int(buffer[i])
            output[i] = max(-32768, min(32767, val))
        return output.tobytes()


def validate_wav(path: Path) -> WavQuality:
    """Validate an accepted sample without destructive processing."""

    if not path.exists():
        return WavQuality(False, 0, 0, 0, 0.0, 0, False, "archivo no existe")
    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            sample_rate = reader.getframerate()
            width = reader.getsampwidth()
            frame_count = reader.getnframes()
            frames = reader.readframes(frame_count)
    except wave.Error as exc:
        return WavQuality(False, 0, 0, 0, 0.0, 0, False, f"WAV invalido: {exc}")
    duration = frame_count / sample_rate if sample_rate else 0.0
    rms_value = rms(frames, width) if frames else 0
    peak_value = peak(frames, width) if frames else 0
    max_peak = (2 ** (8 * width - 1)) - 1
    clipping = peak_value >= int(max_peak * 0.98)
    reason = _quality_reason(sample_rate, channels, width, duration, rms_value, clipping)
    return WavQuality(
        valid=reason is None,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bits=width * 8,
        duration_sec=round(duration, 3),
        rms=rms_value,
        clipping=clipping,
        reason=reason,
    )


def write_voicebank_manifest(root: Path, profile: str = "lykenox") -> None:
    """Ensure the voicebank metadata (JSON, character.txt, character.yaml) exists."""

    manager = VoicebankManager(root, profile)
    vdir = manager.voicebank_dir
    vdir.mkdir(parents=True, exist_ok=True)

    # config.json
    manifest = vdir / "config.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"renderer": "utau_high_quality"}, indent=2), encoding="utf-8")

    # character.txt (Classic UTAU)
    char_txt = vdir / "character.txt"
    if not char_txt.exists():
        content = [
            "name=LYKENOX Spanish Lite",
            f"image={profile}.png",
            "author=LYKENOX Voice Engine",
            "web=https://github.com/lykenox/voice-engine"
        ]
        char_txt.write_text("\n".join(content), encoding="utf-8")

    # character.yaml (OpenUtau)
    char_yaml = vdir / "character.yaml"
    if not char_yaml.exists():
        content = [
            "name: LYKENOX Spanish Lite",
            "author: LYKENOX Voice Engine",
            "language: es",
            "voice: lite",
            "description: Spanish sample-based voicebank for LYKENOX Voice Engine.",
            "version: 1.0"
        ]
        char_yaml.write_text("\n".join(content), encoding="utf-8")


def _fit_sample_to_duration(wav_path: Path, duration: float) -> bytes:
    """Read a mono 48 kHz 16-bit sample and pad or trim to note duration."""

    quality = validate_wav(wav_path)
    if not quality.valid:
        raise RuntimeError(f"Sample invalido {wav_path.name}: {quality.reason}")
    with wave.open(str(wav_path), "rb") as reader:
        frames = reader.readframes(reader.getnframes())
    target_bytes = max(1, int(duration * TARGET_SAMPLE_RATE) * TARGET_SAMPLE_WIDTH)
    if len(frames) >= target_bytes:
        return frames[:target_bytes]
    return frames + (b"\x00" * (target_bytes - len(frames)))


def _notes_from_aliases(aliases: list[str]) -> list[NoteEvent]:
    """Create simple note events when callers only provide lyrics."""

    return [NoteEvent(alias, 60, index * 0.5, 0.5) for index, alias in enumerate(aliases)]


def _quality_reason(
    sample_rate: int,
    channels: int,
    width: int,
    duration: float,
    rms: int,
    clipping: bool,
) -> str | None:
    """Return the first blocking WAV quality issue, if any."""

    if sample_rate != TARGET_SAMPLE_RATE:
        return f"sample rate debe ser {TARGET_SAMPLE_RATE}, actual {sample_rate}"
    if channels != TARGET_CHANNELS:
        return f"canales deben ser mono, actual {channels}"
    if width != TARGET_SAMPLE_WIDTH:
        return "formato debe ser PCM 16-bit"
    if duration < 0.15 or duration > 30.0:
        return f"duracion fuera de rango: {duration:.2f}s"
    if rms < 120:
        return "RMS demasiado bajo o silencio excesivo"
    if clipping:
        return "clipping detectado"
    return None


def _coverage_percent(covered: int, total: int) -> float:
    """Return coverage as a rounded percentage."""

    if total <= 0:
        return 100.0
    return round(max(0, covered) / total * 100.0, 2)
