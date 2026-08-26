"""Build OpenUtau-style phrase requests for WORLDLINE-R."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lykenox_voice_engine.core.oto import OtoEntry
from lykenox_voice_engine.models.notes import NoteEvent

TICKS_PER_BEAT = 480
PITCH_INTERVAL_TICKS = 5


@dataclass(frozen=True)
class EnvelopePoint:
    """One OpenUtau envelope point in milliseconds and percent volume."""

    x: float
    y: float


@dataclass(frozen=True)
class OpenUtauPhone:
    """Minimal RenderPhone equivalent for classic WORLDLINE-R rendering."""

    alias: str
    wav_path: Path
    oto: OtoEntry
    tone: int
    position_ticks: int
    duration_ticks: int
    position_ms: float
    duration_ms: float
    end_ms: float
    preutter_ms: float
    overlap_ms: float
    leading_ms: float
    tail_intrude_ms: float
    tail_overlap_ms: float
    dur_correction_ms: float
    adjusted_tempo: float
    envelope: tuple[EnvelopePoint, EnvelopePoint, EnvelopePoint, EnvelopePoint, EnvelopePoint]


@dataclass(frozen=True)
class OpenUtauRequest:
    """Minimal ResamplerItem equivalent for PhraseSynthAddRequest."""

    phone: OpenUtauPhone
    velocity: int
    volume: int
    modulation: int
    offset: float
    dur_required: float
    consonant: float
    cutoff: float
    skip_over: float
    tempo: float
    pitches: tuple[int, ...]
    pos_ms: float
    length_ms: float
    fade_in_ms: float
    fade_out_ms: float


@dataclass(frozen=True)
class OpenUtauPhrase:
    """Minimal RenderPhrase data needed by WorldlineRenderer."""

    position_ticks: int
    leading_ticks: int
    position_ms: float
    leading_ms: float
    duration_ms: float
    estimated_length_ms: float
    pitches: tuple[float, ...]
    requests: tuple[OpenUtauRequest, ...]


class OpenUtauPhraseAdapter:
    """Translate LYKENOX notes and oto.ini rows into OpenUtau request semantics."""

    def __init__(
        self,
        wav_dir: Path,
        oto: dict[str, OtoEntry],
        alias_resolver: Callable[[str], list[str]],
        tempo: float,
    ) -> None:
        self.wav_dir = wav_dir
        self.oto = oto
        self.alias_resolver = alias_resolver
        self.tempo = tempo

    def build_phrase(self, notes: list[NoteEvent]) -> OpenUtauPhrase:
        """Build an OpenUtau-style phrase with phones, requests, and curves."""

        phones = self._build_phones(notes)
        if not phones:
            raise ValueError("No phones available for WORLDLINE-R phrase.")
        phrase_position = phones[0].position_ticks
        leading_ticks = _ms_to_ticks(self.tempo, phones[0].leading_ms)
        phrase_position_ms = phones[0].position_ms
        phrase_leading_ms = phones[0].leading_ms
        phrase_duration_ms = phones[-1].end_ms - phones[0].position_ms
        phrase_pitches = _phrase_pitches(notes, self.tempo, phrase_position, leading_ticks)
        requests = tuple(
            self._build_request(phone, phrase_position_ms, phrase_leading_ms, phrase_pitches)
            for phone in phones
        )
        return OpenUtauPhrase(
            position_ticks=phrase_position,
            leading_ticks=leading_ticks,
            position_ms=phrase_position_ms,
            leading_ms=phrase_leading_ms,
            duration_ms=phrase_duration_ms,
            estimated_length_ms=phrase_duration_ms + phrase_leading_ms,
            pitches=phrase_pitches,
            requests=requests,
        )

    def sample_f0_curve(self, phrase: OpenUtauPhrase) -> list[float]:
        """Sample phrase pitch like WorldlineRenderer.SampleCurve."""

        frames = max(2, math.ceil(phrase.estimated_length_ms / 10.0))
        values = []
        for index in range(frames):
            pos_ms = phrase.position_ms - phrase.leading_ms + index * 10.0
            ticks = _ms_to_ticks(self.tempo, pos_ms) - (phrase.position_ticks - phrase.leading_ticks)
            pitch_index = max(0, min(len(phrase.pitches) - 1, int(ticks / PITCH_INTERVAL_TICKS)))
            values.append(_tone_to_hz(phrase.pitches[pitch_index] * 0.01))
        return values

    def _build_phones(self, notes: list[NoteEvent]) -> list[OpenUtauPhone]:
        raw = []
        for note in notes:
            aliases = _resolve_aliases(self.alias_resolver, note.lyric.strip().lower(), note.midi)
            if not aliases:
                continue
            duration = note.duration / len(aliases)
            for index, alias in enumerate(aliases):
                entry = self.oto[alias]
                position_ms = (note.start + index * duration) * 1000.0
                end_ms = position_ms + duration * 1000.0
                raw.append((alias, entry, note.midi, position_ms, end_ms))
        phones: list[OpenUtauPhone] = []
        tail_intrudes = [0.0] * len(raw)
        tail_overlaps = [0.0] * len(raw)
        preutters = [max(0.0, item[1].preutterance) for item in raw]
        overlaps = [item[1].overlap for item in raw]
        for index in range(1, len(raw)):
            prev_position = raw[index - 1][3]
            prev_end = raw[index - 1][4]
            position = raw[index][3]
            prev_dur = prev_end - prev_position
            auto_preutter = preutters[index]
            auto_overlap = overlaps[index]
            gap_ms = position - prev_end
            overlapped = False
            max_preutter = auto_preutter
            if gap_ms <= 0:
                overlapped = True
                if auto_preutter - auto_overlap > prev_dur * 0.5:
                    max_preutter = prev_dur * 0.5 / (auto_preutter - auto_overlap) * auto_preutter
            elif gap_ms < auto_preutter:
                max_preutter = gap_ms
            if auto_preutter > max_preutter:
                ratio = max_preutter / auto_preutter if auto_preutter else 0.0
                auto_preutter = max_preutter
                auto_overlap *= ratio
            if auto_preutter > prev_dur and overlapped:
                delta = auto_preutter - prev_dur
                auto_preutter = prev_dur
                auto_overlap -= delta
            preutters[index] = max(0.0, auto_preutter)
            overlaps[index] = auto_overlap
            if overlapped:
                tail_intrudes[index - 1] = max(preutters[index], preutters[index] - overlaps[index])
                tail_overlaps[index - 1] = max(overlaps[index], 0.0)
        for index, (alias, entry, tone, position_ms, end_ms) in enumerate(raw):
            duration_ms = end_ms - position_ms
            dur_correction = preutters[index] - tail_intrudes[index] + tail_overlaps[index]
            envelope = _openutau_envelope(
                duration_ms,
                preutters[index],
                overlaps[index],
                tail_intrudes[index],
                tail_overlaps[index],
            )
            phones.append(
                OpenUtauPhone(
                    alias=alias,
                    wav_path=self.wav_dir / entry.wav,
                    oto=entry,
                    tone=tone,
                    position_ticks=_ms_to_ticks(self.tempo, position_ms),
                    duration_ticks=_ms_to_ticks(self.tempo, duration_ms),
                    position_ms=position_ms,
                    duration_ms=duration_ms,
                    end_ms=end_ms,
                    preutter_ms=preutters[index],
                    overlap_ms=overlaps[index],
                    leading_ms=preutters[index],
                    tail_intrude_ms=tail_intrudes[index],
                    tail_overlap_ms=tail_overlaps[index],
                    dur_correction_ms=dur_correction,
                    adjusted_tempo=self.tempo,
                    envelope=envelope,
                )
            )
        return phones

    def _build_request(
        self,
        phone: OpenUtauPhone,
        phrase_position_ms: float,
        phrase_leading_ms: float,
        phrase_pitches: tuple[float, ...],
    ) -> OpenUtauRequest:
        velocity = 100
        stretch_ratio = math.pow(2.0, 1.0 - velocity * 0.01)
        pitch_leading_ms = phone.oto.preutterance * stretch_ratio
        skip_over = phone.oto.preutterance * stretch_ratio - phone.leading_ms
        dur_required = phone.end_ms - phone.position_ms + phone.dur_correction_ms + skip_over
        dur_required = max(dur_required, phone.oto.consonant)
        dur_required = math.ceil(dur_required / 50.0 + 0.5) * 50.0
        pitch_count_ms = (phone.position_ms + phone.envelope[4].x) - (
            phone.position_ms - pitch_leading_ms
        )
        pitch_count = max(0, math.ceil(_ms_to_ticks(phone.adjusted_tempo, pitch_count_ms) / 5.0))
        pitch_interval_ms = _ticks_to_ms(phone.adjusted_tempo, PITCH_INTERVAL_TICKS)
        phrase_pitch_start_ms = phrase_position_ms - phrase_leading_ms
        pitch_sample_start_ms = phone.position_ms - pitch_leading_ms
        pitches = []
        for index in range(pitch_count):
            sample_pos_ms = pitch_sample_start_ms + pitch_interval_ms * index
            sample_pos_tick = math.floor(_ms_to_ticks(phone.adjusted_tempo, sample_pos_ms))
            sample_interval = _ticks_to_ms(phone.adjusted_tempo, sample_pos_tick + 5) - _ticks_to_ms(
                phone.adjusted_tempo, sample_pos_tick
            )
            sample_index = (sample_pos_tick - _ms_to_ticks(self.tempo, phrase_pitch_start_ms)) / 5.0
            sample_index = max(0.0, min(len(phrase_pitches) - 1, sample_index))
            sample_start = math.floor(sample_index)
            sample_end = math.ceil(sample_index)
            diff_pitch_ms = sample_pos_ms - _ticks_to_ms(
                phone.adjusted_tempo,
                _ms_to_ticks(self.tempo, phrase_pitch_start_ms) + sample_start * 5,
            )
            sample_alpha = diff_pitch_ms / sample_interval if sample_interval else 0.0
            sample_lerped = phrase_pitches[sample_start] + (
                phrase_pitches[sample_end] - phrase_pitches[sample_start]
            ) * sample_alpha
            pitches.append(round(sample_lerped - phone.tone * 100))
        return OpenUtauRequest(
            phone=phone,
            velocity=velocity,
            volume=100,
            modulation=0,
            offset=phone.oto.offset,
            dur_required=dur_required,
            consonant=phone.oto.consonant,
            cutoff=phone.oto.cutoff,
            skip_over=skip_over,
            tempo=phone.adjusted_tempo,
            pitches=tuple(pitches),
            pos_ms=phone.position_ms - phone.leading_ms - (phrase_position_ms - phrase_leading_ms),
            length_ms=phone.envelope[4].x - phone.envelope[0].x,
            fade_in_ms=phone.envelope[1].x - phone.envelope[0].x,
            fade_out_ms=phone.envelope[4].x - phone.envelope[3].x,
        )


def _openutau_envelope(
    duration_ms: float,
    preutter_ms: float,
    overlap_ms: float,
    tail_intrude_ms: float,
    tail_overlap_ms: float,
) -> tuple[EnvelopePoint, EnvelopePoint, EnvelopePoint, EnvelopePoint, EnvelopePoint]:
    """Create UPhoneme.ValidateEnvelope points with default VOL/ATK/DEC."""

    p0 = EnvelopePoint(-preutter_ms, 0.0)
    p1 = EnvelopePoint(p0.x + max(overlap_ms, 5.0), 100.0)
    p2 = EnvelopePoint(max(0.0, p1.x), 100.0)
    p3_x = duration_ms - tail_intrude_ms
    p4_x = p3_x + tail_overlap_ms
    if p3_x == p4_x:
        p3_x = max(p2.x, p3_x - 35.0)
    return (
        p0,
        p1,
        p2,
        EnvelopePoint(p3_x, 100.0),
        EnvelopePoint(p4_x, 0.0),
    )


def _phrase_pitches(notes: list[NoteEvent], tempo: float, position_ticks: int, leading_ticks: int) -> tuple[float, ...]:
    """Create flat OpenUtau phrase pitches in cents, sampled every 5 ticks."""

    pitch_start = position_ticks - leading_ticks
    end_tick = _ms_to_ticks(tempo, (notes[-1].start + notes[-1].duration) * 1000.0)
    length = max(2, int((end_tick - pitch_start) / PITCH_INTERVAL_TICKS) + 1)
    values: list[float] = []
    for index in range(length):
        tick = pitch_start + index * PITCH_INTERVAL_TICKS
        ms = _ticks_to_ms(tempo, tick)
        midi = notes[-1].midi
        for note in notes:
            if note.start * 1000.0 <= ms < (note.start + note.duration) * 1000.0:
                midi = note.midi
                break
        values.append(float(midi * 100))
    return tuple(values)


def _ms_to_ticks(tempo: float, ms: float) -> int:
    """Convert milliseconds to OpenUtau ticks at constant tempo."""

    return int(round(ms * tempo * TICKS_PER_BEAT / 60000.0))


def _ticks_to_ms(tempo: float, ticks: int) -> float:
    """Convert OpenUtau ticks to milliseconds at constant tempo."""

    return ticks * 60000.0 / (tempo * TICKS_PER_BEAT)


def _tone_to_hz(tone: float) -> float:
    """Convert OpenUtau tone value to Hz."""

    return 440.0 * (2.0 ** ((tone - 69.0) / 12.0))


def _resolve_aliases(
    alias_resolver: Callable[..., list[str]],
    lyric: str,
    midi: int,
) -> list[str]:
    """Resolve aliases with optional MIDI-aware multipitch selection."""

    try:
        return alias_resolver(lyric, midi)
    except TypeError:
        return alias_resolver(lyric)
