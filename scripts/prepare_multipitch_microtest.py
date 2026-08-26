"""Prepare the LYKENOX WORLDLINE multipitch microtest structure."""

from __future__ import annotations

import json
import shutil
import sys
import wave
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.adaptive_voicebank_resolver import (  # noqa: E402
    sample_metadata_from_entry,
)
from lykenox_voice_engine.core.multipitch import (  # noqa: E402
    LAYERS,
    MICROTEST_ALIASES,
    design_pitch_centers,
    flatten_layer_oto,
    report_to_dict,
    write_prefix_map,
)
from lykenox_voice_engine.core.oto import OtoEntry, parse_oto, write_oto  # noqa: E402
from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine  # noqa: E402
from scripts.generate_worldline_oto import analyze_wav  # noqa: E402


def main() -> None:
    """Create layer-aware OTO/prefix files without requiring all 276 recordings."""

    voicebank = ROOT / "profiles" / "lykenox" / "voicebank"
    wav_dir = voicebank / "wav"
    raw_root = ROOT / "datasets" / "lykenox" / "voicebank_raw"
    for layer in LAYERS:
        (raw_root / layer).mkdir(parents=True, exist_ok=True)

    f0_values = _measure_low_layer(wav_dir)
    pitch_report = design_pitch_centers(f0_values)
    write_prefix_map(voicebank / "prefix.map", pitch_report.centers)

    base_oto = parse_oto(voicebank / "oto.ini")
    entries = _low_entries(base_oto)
    samples = _low_metadata(wav_dir, base_oto)
    layer_status = {"Low": {"present": list(sorted(base_oto)), "missing": []}}
    for layer in ("Mid", "High"):
        layer_entries, layer_samples, missing = _layer_entries(layer, wav_dir)
        entries.extend(layer_entries)
        samples.extend(layer_samples)
        layer_status[layer] = {
            "present": [entry.alias.removesuffix(f"_{layer.lower()}") for entry in layer_entries],
            "missing": missing,
        }

    write_oto(voicebank / "oto_multipitch.ini", entries)
    metadata_path = voicebank / "adaptive_multipitch_metadata.json"
    metadata = {
        "schema": 1,
        "selection": "minimum_abs_cents_distance_per_alias",
        "samples": [asdict(sample) for sample in samples],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    plan = {
        "microtest_aliases": list(MICROTEST_ALIASES),
        "calibration": {
            "note": "Valores guia por registro comodo; no son requisitos rigidos de aceptacion.",
            **report_to_dict(pitch_report),
        },
        "recording_layers": list(LAYERS),
        "layer_instructions": {
            "Low": "Graba con tu registro grave comodo. No persigas una nota exacta.",
            "Mid": "Graba con tu registro medio comodo. Mantén tu timbre natural.",
            "High": "Graba con tu registro agudo comodo. No imites otra voz ni fuerces afinacion.",
        },
        "raw_dirs": {layer: str(raw_root / layer) for layer in LAYERS},
        "voicebank_wav_dir": str(wav_dir),
        "prefix_map": str(voicebank / "prefix.map"),
        "oto_multipitch": str(voicebank / "oto_multipitch.ini"),
        "metadata": str(metadata_path),
        "layer_status": layer_status,
        "ready_for_multipitch_microtest": _ready(layer_status),
    }
    plan_path = voicebank / "multipitch_microtest_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def _measure_low_layer(wav_dir: Path) -> list[float]:
    engine = OpenUtauWorldlineEngine(ROOT)
    values = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        if "_" in wav_path.stem:
            continue
        samples = _read_wav_mono_float(wav_path)
        mean_f0 = engine.estimate_mean_f0(samples)
        if mean_f0 > 0:
            values.append(mean_f0)
    return values


def _low_entries(base_oto: dict[str, OtoEntry]) -> list[OtoEntry]:
    return [
        OtoEntry(
            wav=entry.wav,
            alias=f"{alias}_low",
            offset=entry.offset,
            consonant=entry.consonant,
            cutoff=entry.cutoff,
            preutterance=entry.preutterance,
            overlap=entry.overlap,
        )
        for alias, entry in sorted(base_oto.items())
    ]


def _low_metadata(wav_dir: Path, base_oto: dict[str, OtoEntry]) -> list[object]:
    samples = []
    for alias, entry in sorted(base_oto.items()):
        wav_path = wav_dir / entry.wav
        if not wav_path.exists():
            continue
        f0 = _f0_stats(wav_path)
        samples.append(
            sample_metadata_from_entry(
                alias=alias,
                layer="Low",
                sample_alias=f"{alias}_low",
                wav=entry.wav,
                measured_f0_hz=f0["mean_f0_hz"],
                voiced_ratio=f0["voiced_ratio"],
                f0_std=f0["f0_std"],
                duration=_duration(wav_path),
                rms=_rms_peak(wav_path)[0],
                peak=_rms_peak(wav_path)[1],
                oto=OtoEntry(
                    wav=entry.wav,
                    alias=f"{alias}_low",
                    offset=entry.offset,
                    consonant=entry.consonant,
                    cutoff=entry.cutoff,
                    preutterance=entry.preutterance,
                    overlap=entry.overlap,
                ),
            )
        )
    return samples


def _layer_entries(layer: str, wav_dir: Path) -> tuple[list[OtoEntry], list[object], list[str]]:
    layer_oto: dict[str, OtoEntry] = {}
    samples = []
    missing = []
    for alias in MICROTEST_ALIASES:
        wav_path = wav_dir / f"{alias}_{layer}.wav"
        if not wav_path.exists():
            raw = ROOT / "datasets" / "lykenox" / "voicebank_raw" / layer / f"{alias}.wav"
            if raw.exists():
                shutil.copy2(raw, wav_path)
            else:
                missing.append(alias)
                continue
        entry, _analysis = analyze_wav(wav_path, alias, None)
        layer_oto[alias] = entry
        sample_alias = f"{alias}_{layer}".lower()
        layer_entry = OtoEntry(
            wav=f"{alias}_{layer}.wav",
            alias=sample_alias,
            offset=entry.offset,
            consonant=entry.consonant,
            cutoff=entry.cutoff,
            preutterance=entry.preutterance,
            overlap=entry.overlap,
        )
        f0 = _f0_stats(wav_path)
        rms_value, peak_value = _rms_peak(wav_path)
        samples.append(
            sample_metadata_from_entry(
                alias=alias,
                layer=layer,
                sample_alias=sample_alias,
                wav=layer_entry.wav,
                measured_f0_hz=f0["mean_f0_hz"],
                voiced_ratio=f0["voiced_ratio"],
                f0_std=f0["f0_std"],
                duration=_duration(wav_path),
                rms=rms_value,
                peak=peak_value,
                oto=layer_entry,
            )
        )
    return flatten_layer_oto(layer_oto, layer), samples, missing


def _ready(layer_status: dict[str, dict[str, list[str]]]) -> bool:
    for layer in LAYERS:
        present = set(layer_status[layer]["present"])
        if any(alias not in present for alias in MICROTEST_ALIASES):
            return False
    return True


def _read_wav_mono_float(path: Path) -> list[float]:
    import wave

    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        raw = reader.readframes(reader.getnframes())
    return [
        int.from_bytes(raw[index : index + 2], "little", signed=True) / 32768.0
        for index in range(0, len(raw), 2 * channels)
    ]


def _f0_stats(path: Path) -> dict[str, float]:
    engine = OpenUtauWorldlineEngine(ROOT)
    samples = _read_wav_mono_float(path)
    mean_f0 = engine.estimate_mean_f0(samples)
    return {
        "mean_f0_hz": mean_f0,
        "voiced_ratio": 1.0 if mean_f0 > 0 else 0.0,
        "f0_std": 0.0,
    }


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / float(reader.getframerate())


def _rms_peak(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        raw = reader.readframes(reader.getnframes())
    values = [
        int.from_bytes(raw[index : index + 2], "little", signed=True)
        for index in range(0, len(raw), 2 * channels)
    ]
    if not values:
        return 0, 0
    rms_value = int((sum(value * value for value in values) / len(values)) ** 0.5)
    return rms_value, max(abs(value) for value in values)


if __name__ == "__main__":
    main()
