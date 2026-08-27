"""Minimal MIDI parser for score import tests and local API adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lykenox_voice_engine.models.notes import NoteEvent


@dataclass(frozen=True)
class MidiParseResult:
    """Parsed MIDI tempo, notes, and lyric events."""

    tempo: int
    notes: tuple[NoteEvent, ...]
    lyrics: tuple[str, ...]


def parse_midi(path: Path, track_index: int = 0, external_lyrics: list[str] | None = None) -> MidiParseResult:
    """Parse a Standard MIDI File with note_on/off, tempo, and lyric events."""

    data = path.read_bytes()
    reader = _Reader(data)
    if reader.read(4) != b"MThd":
        raise ValueError("MIDI invalido: falta MThd")
    header_size = reader.u32()
    header = _Reader(reader.read(header_size))
    header.u16()
    track_count = header.u16()
    division = header.u16()
    tempo_bpm = 120
    tracks: list[tuple[list[tuple[int, int, int]], list[str]]] = []
    for _ in range(track_count):
        if reader.read(4) != b"MTrk":
            raise ValueError("MIDI invalido: falta MTrk")
        track_data = reader.read(reader.u32())
        notes, lyrics, tempo = _parse_track(track_data, division)
        if tempo:
            tempo_bpm = tempo
        tracks.append((notes, lyrics))
    if not tracks:
        return MidiParseResult(tempo=tempo_bpm, notes=(), lyrics=())
    selected = min(max(track_index, 0), len(tracks) - 1)
    raw_notes, midi_lyrics = tracks[selected]
    lyric_values = external_lyrics or midi_lyrics
    note_events = []
    for index, (pitch, start_tick, end_tick) in enumerate(raw_notes):
        lyric = lyric_values[index] if index < len(lyric_values) else "la"
        note_events.append(
            NoteEvent(
                lyric=lyric,
                midi=pitch,
                start=round(start_tick / division, 6),
                duration=round((end_tick - start_tick) / division, 6),
            )
        )
    return MidiParseResult(tempo=tempo_bpm, notes=tuple(note_events), lyrics=tuple(midi_lyrics))


class _Reader:
    """Small binary reader for MIDI chunks."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, count: int) -> bytes:
        """Read count bytes or raise ValueError."""

        if self.pos + count > len(self.data):
            raise ValueError("MIDI truncado")
        payload = self.data[self.pos : self.pos + count]
        self.pos += count
        return payload

    def u16(self) -> int:
        """Read a big-endian uint16."""

        return int.from_bytes(self.read(2), "big")

    def u32(self) -> int:
        """Read a big-endian uint32."""

        return int.from_bytes(self.read(4), "big")

    def varlen(self) -> int:
        """Read a MIDI variable-length integer."""

        value = 0
        while True:
            byte = self.read(1)[0]
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value


def _parse_track(track_data: bytes, division: int) -> tuple[list[tuple[int, int, int]], list[str], int | None]:
    """Parse one track body."""

    reader = _Reader(track_data)
    tick = 0
    running_status: int | None = None
    active: dict[int, int] = {}
    notes: list[tuple[int, int, int]] = []
    lyrics: list[str] = []
    tempo: int | None = None
    while reader.pos < len(reader.data):
        tick += reader.varlen()
        status = reader.read(1)[0]
        if status < 0x80:
            if running_status is None:
                raise ValueError("MIDI invalido: running status ausente")
            reader.pos -= 1
            status = running_status
        elif status != 0xFF:
            running_status = status
        if status == 0xFF:
            meta_type = reader.read(1)[0]
            length = reader.varlen()
            payload = reader.read(length)
            if meta_type == 0x05:
                lyrics.append(payload.decode("utf-8", errors="ignore").strip())
            elif meta_type == 0x51 and len(payload) == 3:
                micros = int.from_bytes(payload, "big")
                tempo = round(60_000_000 / micros)
            continue
        command = status & 0xF0
        if command in {0x80, 0x90}:
            pitch = reader.read(1)[0]
            velocity = reader.read(1)[0]
            if command == 0x90 and velocity > 0:
                active[pitch] = tick
            elif pitch in active:
                start = active.pop(pitch)
                notes.append((pitch, start, tick))
        elif command in {0xA0, 0xB0, 0xE0}:
            reader.read(2)
        elif command in {0xC0, 0xD0}:
            reader.read(1)
        else:
            raise ValueError(f"MIDI evento no soportado: 0x{status:02X}")
    return notes, lyrics, tempo
