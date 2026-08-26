"""Adaptive multipitch sample selection from measured vocal pitch."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lykenox_voice_engine.core.multipitch import hz_to_midi, midi_to_hz
from lykenox_voice_engine.core.oto import OtoEntry


@dataclass(frozen=True)
class SampleMetadata:
    """Measured data for one recorded voicebank sample."""

    alias: str
    layer: str
    sample_alias: str
    wav: str
    measured_f0_hz: float
    measured_midi: float
    voiced_ratio: float
    f0_std: float
    duration: float
    rms: int
    peak: int
    oto: dict[str, float | str]


@dataclass(frozen=True)
class ResolvedSample:
    """Selected sample and pitch-shift details for one target note."""

    alias: str
    target_midi: int
    target_f0_hz: float
    selected_sample: str
    wav: str
    layer: str
    measured_source_f0_hz: float
    required_shift_cents: float
    oto: OtoEntry


class AdaptiveVoicebankResolver:
    """Choose the alias variant requiring the smallest real pitch shift."""

    def __init__(
        self,
        metadata: dict[str, list[SampleMetadata]],
        oto: dict[str, OtoEntry],
    ) -> None:
        self.metadata = metadata
        self.oto = oto
        self.selections: list[ResolvedSample] = []

    @classmethod
    def from_file(cls, path: Path, oto: dict[str, OtoEntry]) -> "AdaptiveVoicebankResolver":
        """Load adaptive resolver metadata from JSON."""

        data = json.loads(path.read_text(encoding="utf-8"))
        metadata: dict[str, list[SampleMetadata]] = {}
        for item in data.get("samples", []):
            sample = SampleMetadata(
                alias=str(item["alias"]).lower(),
                layer=str(item["layer"]),
                sample_alias=str(item["sample_alias"]).lower(),
                wav=str(item["wav"]),
                measured_f0_hz=float(item["measured_f0_hz"]),
                measured_midi=float(item["measured_midi"]),
                voiced_ratio=float(item["voiced_ratio"]),
                f0_std=float(item["f0_std"]),
                duration=float(item["duration"]),
                rms=int(item["rms"]),
                peak=int(item["peak"]),
                oto=dict(item["oto"]),
            )
            metadata.setdefault(sample.alias, []).append(sample)
        return cls(metadata, oto)

    def resolve(self, alias: str, target_midi: int) -> ResolvedSample:
        """Return the best available sample for the alias and target MIDI."""

        key = alias.strip().lower()
        candidates = [
            sample
            for sample in self.metadata.get(key, [])
            if sample.measured_f0_hz > 0 and sample.sample_alias in self.oto
        ]
        if not candidates and key in self.oto:
            entry = self.oto[key]
            fallback_f0 = midi_to_hz(target_midi)
            resolved = ResolvedSample(
                alias=key,
                target_midi=target_midi,
                target_f0_hz=round(fallback_f0, 2),
                selected_sample=key,
                wav=entry.wav,
                layer="Fallback",
                measured_source_f0_hz=round(fallback_f0, 2),
                required_shift_cents=0.0,
                oto=entry,
            )
            self.selections.append(resolved)
            return resolved
        if not candidates:
            raise KeyError(f"No hay variantes disponibles para alias '{key}'.")

        target_hz = midi_to_hz(target_midi)
        best = min(candidates, key=lambda sample: abs(cents_distance(target_hz, sample.measured_f0_hz)))
        resolved = ResolvedSample(
            alias=key,
            target_midi=target_midi,
            target_f0_hz=round(target_hz, 2),
            selected_sample=best.sample_alias,
            wav=best.wav,
            layer=best.layer,
            measured_source_f0_hz=round(best.measured_f0_hz, 2),
            required_shift_cents=round(cents_distance(target_hz, best.measured_f0_hz), 1),
            oto=self.oto[best.sample_alias],
        )
        self.selections.append(resolved)
        return resolved

    def aliases_for_note(self, alias: str, target_midi: int) -> list[str]:
        """Resolver callback used by WORLDLINE phrase construction."""

        return [self.resolve(alias, target_midi).selected_sample]

    def average_abs_pitch_shift_cents(self) -> float:
        """Return average absolute shift of all selections."""

        if not self.selections:
            return 0.0
        total = sum(abs(item.required_shift_cents) for item in self.selections)
        return round(total / len(self.selections), 1)

    def selection_table(self) -> list[dict[str, Any]]:
        """Return selection details for reporting."""

        return [
            {
                "alias": item.alias,
                "target_hz": item.target_f0_hz,
                "selected_sample": item.selected_sample,
                "source_f0": item.measured_source_f0_hz,
                "shift_cents": item.required_shift_cents,
                "layer": item.layer,
            }
            for item in self.selections
        ]


def cents_distance(target_hz: float, source_hz: float) -> float:
    """Calculate signed pitch distance in cents."""

    if target_hz <= 0 or source_hz <= 0:
        raise ValueError("target_hz y source_hz deben ser positivos.")
    return 1200.0 * math.log2(target_hz / source_hz)


def sample_metadata_from_entry(
    alias: str,
    layer: str,
    sample_alias: str,
    wav: str,
    measured_f0_hz: float,
    voiced_ratio: float,
    f0_std: float,
    duration: float,
    rms: int,
    peak: int,
    oto: OtoEntry,
) -> SampleMetadata:
    """Create metadata for one accepted adaptive sample."""

    return SampleMetadata(
        alias=alias.lower(),
        layer=layer,
        sample_alias=sample_alias.lower(),
        wav=wav,
        measured_f0_hz=round(measured_f0_hz, 2),
        measured_midi=round(hz_to_midi(measured_f0_hz), 2) if measured_f0_hz > 0 else 0.0,
        voiced_ratio=round(voiced_ratio, 3),
        f0_std=round(f0_std, 2),
        duration=round(duration, 3),
        rms=rms,
        peak=peak,
        oto={
            "offset": oto.offset,
            "consonant": oto.consonant,
            "cutoff": oto.cutoff,
            "preutterance": oto.preutterance,
            "overlap": oto.overlap,
        },
    )
