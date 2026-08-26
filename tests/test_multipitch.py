"""Tests for WORLDLINE multipitch prefix-map selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.core.adaptive_voicebank_resolver import (
    AdaptiveVoicebankResolver,
    SampleMetadata,
    cents_distance,
)
from lykenox_voice_engine.core.multipitch import (
    design_pitch_centers,
    layer_alias,
    read_prefix_map,
    suffix_for_midi,
    write_prefix_map,
)
from lykenox_voice_engine.core.oto import OtoEntry


class TestMultipitch(unittest.TestCase):
    """Validate measured layer design and automatic MIDI suffix selection."""

    def test_design_centers_from_measured_low_voice(self) -> None:
        report = design_pitch_centers([119.0, 122.0, 124.0, 126.0])

        self.assertEqual(report.centers[0].layer, "Low")
        self.assertEqual(report.centers[0].midi, 47)
        self.assertEqual(report.centers[1].midi, 52)
        self.assertEqual(report.centers[2].midi, 57)

    def test_prefix_map_selects_nearest_layer_by_midi(self) -> None:
        report = design_pitch_centers([122.44])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prefix.map"
            write_prefix_map(path, report.centers)
            prefix_map = read_prefix_map(path)

        self.assertEqual(layer_alias("bai", suffix_for_midi(prefix_map, 47)), "bai_low")
        self.assertEqual(layer_alias("bai", suffix_for_midi(prefix_map, 52)), "bai_mid")
        self.assertEqual(layer_alias("bai", suffix_for_midi(prefix_map, 60)), "bai_high")

    def test_cents_distance_is_signed(self) -> None:
        self.assertAlmostEqual(cents_distance(220.0, 110.0), 1200.0, places=2)
        self.assertAlmostEqual(cents_distance(110.0, 220.0), -1200.0, places=2)

    def test_resolver_uses_real_alias_pitch_not_layer_center(self) -> None:
        resolver = AdaptiveVoicebankResolver(
            {
                "la": [
                    _sample("la", "Low", "la_low", 110.0),
                    _sample("la", "Mid", "la_mid", 124.0),
                    _sample("la", "High", "la_high", 151.0),
                ],
                "mi": [
                    _sample("mi", "Low", "mi_low", 118.0),
                    _sample("mi", "Mid", "mi_mid", 135.0),
                    _sample("mi", "High", "mi_high", 143.0),
                ],
            },
            _oto("la_low", "la_mid", "la_high", "mi_low", "mi_mid", "mi_high"),
        )

        la = resolver.resolve("la", 60)
        mi = resolver.resolve("mi", 60)

        self.assertEqual(la.selected_sample, "la_high")
        self.assertEqual(mi.selected_sample, "mi_high")
        self.assertLess(abs(la.required_shift_cents), abs(cents_distance(261.63, 124.0)))
        self.assertNotEqual(la.measured_source_f0_hz, mi.measured_source_f0_hz)

    def test_resolver_falls_back_to_low_when_mid_high_missing(self) -> None:
        resolver = AdaptiveVoicebankResolver(
            {"go": [_sample("go", "Low", "go_low", 121.0)]},
            _oto("go_low"),
        )

        selected = resolver.resolve("go", 64)

        self.assertEqual(selected.selected_sample, "go_low")
        self.assertGreater(selected.required_shift_cents, 0)


def _sample(alias: str, layer: str, sample_alias: str, f0: float) -> SampleMetadata:
    return SampleMetadata(
        alias=alias,
        layer=layer,
        sample_alias=sample_alias,
        wav=f"{sample_alias}.wav",
        measured_f0_hz=f0,
        measured_midi=0.0,
        voiced_ratio=0.9,
        f0_std=3.0,
        duration=2.0,
        rms=1000,
        peak=3000,
        oto={},
    )


def _oto(*aliases: str) -> dict[str, OtoEntry]:
    return {
        alias: OtoEntry(
            wav=f"{alias}.wav",
            alias=alias,
            offset=0.0,
            consonant=80.0,
            cutoff=100.0,
            preutterance=60.0,
            overlap=25.0,
        )
        for alias in aliases
    }


if __name__ == "__main__":
    unittest.main()
