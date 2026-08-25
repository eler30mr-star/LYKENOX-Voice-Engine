"""OpenUtau WORLDLINE-R wrapper pinned to the 0.1.565 native engine."""

from __future__ import annotations

import ctypes
import math
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lykenox_voice_engine.core.oto import OtoEntry
from lykenox_voice_engine.engines.openutau_phrase_adapter import (
    OpenUtauPhraseAdapter,
    OpenUtauRequest,
)
from lykenox_voice_engine.models.notes import NoteEvent

SOURCE_VERSION = "0.1.565"
SOURCE_COMMIT = "a60ca5830b9064556157245d4bf8f5920d93e5f8"
SOURCE_REPOSITORY = "https://github.com/openutau/OpenUtau"
FRAME_MS = 10.0
DEFAULT_SAMPLE_RATE = 48_000
REQUIRED_EXPORTS = (
    "Resample",
    "WorldSynthesis",
    "F0",
    "PhraseSynthNew",
    "PhraseSynthAddRequest",
    "PhraseSynthSetCurves",
    "PhraseSynthSynth",
    "PhraseSynthDelete",
)


class WorldlineUnavailableError(RuntimeError):
    """Raised when the official WORLDLINE-R DLL cannot be used."""


@dataclass(frozen=True)
class WorldlineRenderReport:
    """Technical details from one WORLDLINE-R render."""

    duration_sec: float
    render_time_sec: float
    peak: float
    mean_f0_hz: float
    expected_f0_hz: float
    pitch_correct: bool
    oto_applied: bool
    formant_handling: bool
    phrase_synth_used: bool


class SynthRequest(ctypes.Structure):
    """ctypes mirror of OpenUtau cpp/worldline/synth_request.h."""

    _fields_ = [
        ("sample_fs", ctypes.c_int32),
        ("sample_length", ctypes.c_int32),
        ("sample", ctypes.POINTER(ctypes.c_double)),
        ("frq_length", ctypes.c_int32),
        ("frq", ctypes.c_void_p),
        ("tone", ctypes.c_int32),
        ("con_vel", ctypes.c_double),
        ("offset", ctypes.c_double),
        ("required_length", ctypes.c_double),
        ("consonant", ctypes.c_double),
        ("cut_off", ctypes.c_double),
        ("volume", ctypes.c_double),
        ("modulation", ctypes.c_double),
        ("tempo", ctypes.c_double),
        ("pitch_bend_length", ctypes.c_int32),
        ("pitch_bend", ctypes.POINTER(ctypes.c_int32)),
        ("flag_g", ctypes.c_int32),
        ("flag_O", ctypes.c_int32),
        ("flag_P", ctypes.c_int32),
        ("flag_Mt", ctypes.c_int32),
        ("flag_Mb", ctypes.c_int32),
        ("flag_Mv", ctypes.c_int32),
    ]


class OpenUtauWorldlineEngine:
    """Render LYKENOX voicebank phrases with official OpenUtau WORLDLINE-R."""

    def __init__(self, root: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.root = root
        self.sample_rate = sample_rate
        self.dll_path = root / "tools" / "renderers" / "worldline_r" / "worldline.dll"
        self._dll: ctypes.CDLL | None = None
        self._log_callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p)(self._log)

    def health_check(self) -> dict[str, object]:
        """Return source, DLL, export, and availability status."""

        dll_exists = self.dll_path.exists()
        dll_loadable = False
        exports: list[str] = []
        missing_exports = list(REQUIRED_EXPORTS)
        if dll_exists:
            try:
                dll = self._load()
                dll_loadable = True
                exports = [name for name in REQUIRED_EXPORTS if hasattr(dll, name)]
                missing_exports = [name for name in REQUIRED_EXPORTS if name not in exports]
            except OSError:
                dll_loadable = False
        exports_ok = not missing_exports
        return {
            "source_version": SOURCE_VERSION,
            "commit": SOURCE_COMMIT,
            "repository": SOURCE_REPOSITORY,
            "dll_path": str(self.dll_path),
            "dll_exists": dll_exists,
            "dll_loadable": dll_loadable,
            "exports": exports,
            "missing_exports": missing_exports,
            "exports_ok": exports_ok,
            "available": dll_exists and dll_loadable and exports_ok,
        }

    def render_to_path(
        self,
        wav_dir: Path,
        oto: dict[str, OtoEntry],
        alias_resolver: Callable[[str], list[str]],
        notes: list[NoteEvent],
        tempo: float,
        output_path: Path,
    ) -> WorldlineRenderReport:
        """Render a full phrase through OpenUtau WORLDLINE-R PhraseSynth."""

        started = time.perf_counter()
        samples = self.render_samples(wav_dir, oto, alias_resolver, notes, tempo)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = _float_to_pcm16(samples)
        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate)
            writer.writeframes(pcm)
        expected = _midi_to_hz(notes[0].midi if notes else 60)
        mean_f0 = self.estimate_mean_f0(samples)
        return WorldlineRenderReport(
            duration_sec=len(samples) / self.sample_rate,
            render_time_sec=time.perf_counter() - started,
            peak=max((abs(value) for value in samples), default=0.0),
            mean_f0_hz=mean_f0,
            expected_f0_hz=expected,
            pitch_correct=mean_f0 > 0 and abs(1200 * math.log2(mean_f0 / expected)) < 250,
            oto_applied=True,
            formant_handling=True,
            phrase_synth_used=True,
        )

    def render_samples(
        self,
        wav_dir: Path,
        oto: dict[str, OtoEntry],
        alias_resolver: Callable[[str], list[str]],
        notes: list[NoteEvent],
        tempo: float,
    ) -> list[float]:
        """Return phrase samples from the official PhraseSynth entry points."""

        if not self.health_check()["available"]:
            raise WorldlineUnavailableError("WORLDLINE-R DLL no disponible o exports incompletos.")
        dll = self._load()
        adapter = OpenUtauPhraseAdapter(wav_dir, oto, alias_resolver, tempo)
        phrase = adapter.build_phrase(notes)
        synth = dll.PhraseSynthNew()
        if not synth:
            raise WorldlineUnavailableError("PhraseSynthNew no devolvio un handle valido.")
        try:
            for openutau_request in phrase.requests:
                request, keepalive = self._build_request(openutau_request)
                dll.PhraseSynthAddRequest(
                    synth,
                    ctypes.byref(request),
                    openutau_request.pos_ms,
                    openutau_request.skip_over,
                    openutau_request.length_ms,
                    openutau_request.fade_in_ms,
                    openutau_request.fade_out_ms,
                    self._log_callback,
                )
                del keepalive
            f0_values = adapter.sample_f0_curve(phrase)
            frames = len(f0_values)
            f0, gender, tension, breathiness, voicing = _curves_from_f0(f0_values)
            dll.PhraseSynthSetCurves(
                synth,
                f0,
                gender,
                tension,
                breathiness,
                voicing,
                frames,
                self._log_callback,
            )
            out_ptr = ctypes.POINTER(ctypes.c_float)()
            length = dll.PhraseSynthSynth(synth, ctypes.byref(out_ptr), self._log_callback)
            if length <= 0 or not out_ptr:
                raise WorldlineUnavailableError("PhraseSynthSynth no produjo audio.")
            return [float(out_ptr[index]) for index in range(length)]
        finally:
            dll.PhraseSynthDelete(synth)

    def estimate_mean_f0(self, samples: list[float]) -> float:
        """Estimate mean voiced F0 using the same official WORLDLINE-R DLL."""

        if not samples:
            return 0.0
        dll = self._load()
        sample_array = (ctypes.c_float * len(samples))(*samples)
        f0_ptr = ctypes.POINTER(ctypes.c_double)()
        length = dll.F0(sample_array, len(samples), self.sample_rate, FRAME_MS, 0, ctypes.byref(f0_ptr))
        if length <= 0 or not f0_ptr:
            return 0.0
        values = [float(f0_ptr[index]) for index in range(length) if f0_ptr[index] > 0]
        return sum(values) / len(values) if values else 0.0

    def _build_request(
        self,
        item: OpenUtauRequest,
    ) -> tuple[SynthRequest, tuple[object, ...]]:
        """Create a pinned SynthRequest from one voicebank sample."""

        sample_rate, samples = _read_wav_mono_float(item.phone.wav_path)
        sample_array = (ctypes.c_double * len(samples))(*samples)
        pitch_values = item.pitches or (0,)
        pitch_bend = (ctypes.c_int32 * len(pitch_values))(*pitch_values)
        request = SynthRequest(
            sample_rate,
            len(samples),
            sample_array,
            0,
            None,
            item.phone.tone,
            float(item.velocity),
            item.offset,
            item.dur_required,
            item.consonant,
            item.cutoff,
            float(item.volume),
            float(item.modulation),
            item.tempo,
            len(pitch_values),
            pitch_bend,
            0,
            0,
            86,
            0,
            0,
            100,
        )
        return request, (sample_array, pitch_bend)

    def _load(self) -> ctypes.CDLL:
        """Load the official native DLL and configure ctypes signatures."""

        if self._dll is None:
            if not self.dll_path.exists():
                raise OSError(f"No existe {self.dll_path}")
            self._dll = ctypes.CDLL(str(self.dll_path))
            self._dll.PhraseSynthNew.restype = ctypes.c_void_p
            self._dll.PhraseSynthDelete.argtypes = [ctypes.c_void_p]
            self._dll.PhraseSynthAddRequest.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(SynthRequest),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_void_p,
            ]
            self._dll.PhraseSynthSetCurves.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self._dll.PhraseSynthSynth.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p,
            ]
            self._dll.PhraseSynthSynth.restype = ctypes.c_int
            self._dll.Resample.argtypes = [
                ctypes.POINTER(SynthRequest),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ]
            self._dll.Resample.restype = ctypes.c_int
            self._dll.F0.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ]
            self._dll.F0.restype = ctypes.c_int
        return self._dll

    @staticmethod
    def _log(message: bytes | None) -> None:
        """Receive optional native log callbacks."""

        del message


def _request_timing(
    notes: list[NoteEvent],
    note_index: int,
    alias_index: int,
    alias_count: int,
    duration_per_alias: float,
    entry: OtoEntry,
) -> tuple[float, float, float, float, float]:
    """Map note timing and OTO values to PhraseSynthAddRequest timing."""

    note = notes[note_index]
    alias_start_ms = (note.start + alias_index * duration_per_alias) * 1000.0
    pos_ms = max(0.0, alias_start_ms - entry.preutterance)
    next_start_ms = (note.start + (alias_index + 1) * duration_per_alias) * 1000.0
    if alias_index + 1 >= alias_count and note_index + 1 < len(notes):
        next_start_ms = notes[note_index + 1].start * 1000.0
    length_ms = max(20.0, next_start_ms - pos_ms + entry.overlap)
    fade_in_ms = max(1.0, entry.overlap)
    fade_out_ms = max(1.0, min(25.0, length_ms * 0.25))
    return pos_ms, 0.0, length_ms, fade_in_ms, fade_out_ms


def _curves_from_f0(
    f0_values: list[float],
) -> tuple[
    ctypes.Array[ctypes.c_double],
    ctypes.Array[ctypes.c_double],
    ctypes.Array[ctypes.c_double],
    ctypes.Array[ctypes.c_double],
    ctypes.Array[ctypes.c_double],
]:
    """Create WORLDLINE-R phrase curves from OpenUtau-sampled F0 values."""

    frames = len(f0_values)
    f0 = (ctypes.c_double * frames)(*f0_values)
    gender = (ctypes.c_double * frames)(*[0.5] * frames)
    tension = (ctypes.c_double * frames)(*[0.5] * frames)
    breathiness = (ctypes.c_double * frames)(*[0.5] * frames)
    voicing = (ctypes.c_double * frames)(*[1.0] * frames)
    return f0, gender, tension, breathiness, voicing


def _estimated_phrase_ms(notes: list[NoteEvent]) -> float:
    """Return phrase duration including a small release tail."""

    if not notes:
        return 0.0
    return (notes[-1].start + notes[-1].duration + 0.25) * 1000.0


def _midi_to_hz(midi: int) -> float:
    """Convert MIDI note number to frequency in Hz."""

    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _read_wav_mono_float(path: Path) -> tuple[int, list[float]]:
    """Read a mono PCM WAV as normalized floats."""

    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    if width != 2:
        raise ValueError(f"WORLDLINE-R espera WAV PCM 16-bit: {path}")
    values = array("h")
    values.frombytes(frames)
    if channels == 1:
        return sample_rate, [value / 32768.0 for value in values]
    mono = []
    for index in range(0, len(values), channels):
        mono.append(sum(values[index : index + channels]) / channels / 32768.0)
    return sample_rate, mono


def _float_to_pcm16(samples: list[float]) -> bytes:
    """Convert normalized float samples to little-endian PCM16 bytes."""

    pcm = array("h", [int(max(-32768, min(32767, sample * 32767.0))) for sample in samples])
    return pcm.tobytes()
