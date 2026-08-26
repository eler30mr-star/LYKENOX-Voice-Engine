"""Render monopitch vs multipitch WORLDLINE-R microtest for baila conmigo."""

from __future__ import annotations

import json
import math
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.adaptive_voicebank_resolver import (  # noqa: E402
    AdaptiveVoicebankResolver,
    cents_distance,
)
from lykenox_voice_engine.core.multipitch import MICROTEST_ALIASES, midi_to_hz  # noqa: E402
from lykenox_voice_engine.core.oto import parse_oto  # noqa: E402
from lykenox_voice_engine.core.score import load_score  # noqa: E402
from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine  # noqa: E402
from lykenox_voice_engine.models.notes import NoteEvent  # noqa: E402

SCORE_PATH = ROOT / "scores" / "baila_conmigo_microtest.json"


def main() -> None:
    """Render both paths only when real multipitch microtest samples exist."""

    voicebank = ROOT / "profiles" / "lykenox" / "voicebank"
    output_dir = ROOT / "outputs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = OpenUtauWorldlineEngine(ROOT)
    score = load_score(SCORE_PATH)
    notes = list(score.notes)

    mono_path = output_dir / "vocal_monopitch.wav"
    mono_started = time.perf_counter()
    mono_report = engine.render_to_path(
        voicebank / "wav",
        parse_oto(voicebank / "oto.ini"),
        lambda lyric: [lyric],
        notes,
        score.tempo,
        mono_path,
    )

    multipitch_oto = parse_oto(voicebank / "oto_multipitch.ini")
    metadata_path = voicebank / "adaptive_multipitch_metadata.json"
    missing = _missing_multipitch(metadata_path, multipitch_oto)
    payload = {
        "score_file": str(SCORE_PATH),
        "phrase": score.lyrics,
        "aliases": list(MICROTEST_ALIASES),
        "score": [{"alias": note.lyric, "midi": note.midi} for note in notes],
        "monopitch": {
            **_validate(engine, mono_path, mono_report.render_time_sec, notes),
            "selection_table": _monopitch_selection_table(voicebank),
            "average_abs_pitch_shift_cents": _monopitch_average_shift(voicebank),
        },
        "multipitch_ready": not missing,
        "missing_multipitch_aliases": missing,
        "missing_recording_layers": _missing_recording_layers(voicebank),
    }
    if missing:
        out_path = output_dir / "multipitch_microtest_report.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    multi_path = output_dir / "vocal_adaptive_multipitch.wav"
    resolver = AdaptiveVoicebankResolver.from_file(metadata_path, multipitch_oto)
    multi_report = engine.render_to_path(
        voicebank / "wav",
        multipitch_oto,
        resolver.aliases_for_note,
        notes,
        score.tempo,
        multi_path,
    )
    _match_duration(multi_path, mono_path)
    payload["multipitch"] = {
        **_validate(engine, multi_path, multi_report.render_time_sec, notes),
        "selection_table": resolver.selection_table(),
        "average_abs_pitch_shift_cents": resolver.average_abs_pitch_shift_cents(),
    }
    payload["render_time_sec_total"] = round(time.perf_counter() - mono_started, 3)
    out_path = output_dir / "multipitch_microtest_report.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _missing_multipitch(metadata_path: Path, oto: dict[str, object]) -> list[str]:
    if not metadata_path.exists():
        return ["adaptive_multipitch_metadata.json"]
    resolver = AdaptiveVoicebankResolver.from_file(metadata_path, oto)
    missing = []
    for note in load_score(SCORE_PATH).notes:
        try:
            resolver.resolve(note.lyric, note.midi)
        except KeyError:
            missing.append(note.lyric)
    return sorted(set(missing))


def _monopitch_selection_table(voicebank: Path) -> list[dict[str, object]]:
    metadata_path = voicebank / "adaptive_multipitch_metadata.json"
    oto = parse_oto(voicebank / "oto_multipitch.ini")
    resolver = AdaptiveVoicebankResolver.from_file(metadata_path, oto)
    table = []
    for note in load_score(SCORE_PATH).notes:
        low = next(
            sample
            for sample in resolver.metadata[note.lyric]
            if sample.layer == "Low"
        )
        target_hz = midi_to_hz(note.midi)
        table.append(
            {
                "alias": note.lyric,
                "target_hz": round(target_hz, 2),
                "selected_sample": low.sample_alias,
                "source_f0": round(low.measured_f0_hz, 2),
                "shift_cents": round(cents_distance(target_hz, low.measured_f0_hz), 1),
                "layer": "Low",
            }
        )
    return table


def _monopitch_average_shift(voicebank: Path) -> float:
    values = [abs(item["shift_cents"]) for item in _monopitch_selection_table(voicebank)]
    return round(sum(values) / len(values), 1) if values else 0.0


def _missing_recording_layers(voicebank: Path) -> dict[str, list[str]]:
    plan_path = voicebank / "multipitch_microtest_plan.json"
    if not plan_path.exists():
        return {}
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return {
        layer: list(status.get("missing", []))
        for layer, status in data.get("layer_status", {}).items()
        if status.get("missing")
    }


def _validate(
    engine: OpenUtauWorldlineEngine,
    path: Path,
    render_time_sec: float,
    notes: list[NoteEvent],
) -> dict[str, object]:
    samples = _read_wav_mono_float(path)
    return {
        "path": str(path),
        "duration_sec": round(len(samples) / 48000.0, 3),
        "render_time_sec": round(render_time_sec, 3),
        "peak": round(max((abs(value) for value in samples), default=0.0), 4),
        "per_note_f0": [_note_f0(engine, samples, note) for note in notes],
    }


def _note_f0(
    engine: OpenUtauWorldlineEngine,
    samples: list[float],
    note: NoteEvent,
) -> dict[str, object]:
    start = int((note.start + 0.25) * 48000)
    end = int((note.start + note.duration + 0.15) * 48000)
    window = samples[max(0, start) : min(len(samples), end)]
    measured = engine.estimate_mean_f0(window)
    target = 440.0 * (2.0 ** ((note.midi - 69.0) / 12.0))
    cents = 1200.0 * math.log2(measured / target) if measured > 0 else 0.0
    return {
        "alias": note.lyric,
        "midi": note.midi,
        "target_hz": round(target, 2),
        "measured_hz": round(measured, 2),
        "error_cents": round(cents, 1),
    }


def _read_wav_mono_float(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        raw = reader.readframes(reader.getnframes())
    return [
        int.from_bytes(raw[index : index + 2], "little", signed=True) / 32768.0
        for index in range(0, len(raw), 2 * channels)
    ]


def _match_duration(path: Path, reference: Path) -> None:
    samples = _read_wav_mono_float(path)
    reference_samples = _read_wav_mono_float(reference)
    target_len = len(reference_samples)
    if len(samples) < target_len:
        samples = samples + [0.0] * (target_len - len(samples))
    else:
        samples = samples[:target_len]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(48000)
        for sample in samples:
            value = max(-32768, min(32767, int(sample * 32767)))
            writer.writeframesraw(value.to_bytes(2, "little", signed=True))


if __name__ == "__main__":
    main()
