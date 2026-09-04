from __future__ import annotations

from pathlib import Path


def test_recording_v2_page_uses_strict_high_resolution_capture() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "lykenox_voice_engine" / "ui" / "recording_v2_page.py"
    text = path.read_text(encoding="utf-8")

    assert "TARGET_SAMPLE_RATE = 48000" in text
    assert "TARGET_CHANNELS = 1" in text
    assert 'OUTPUT_SUBTYPE = "FLOAT"' in text
    assert "QAudioFormat.Float" in text
    assert "QAudioFormat.Int32" in text
    assert "QAudioFormat.Int16" not in text
    assert "device.isFormatSupported(fmt)" in text
    assert "No se hará fallback silencioso" in text
    assert 'subtype=OUTPUT_SUBTYPE' in text
    assert 'format="WAV"' in text


def test_recording_v2_page_has_no_audio_enhancement_path() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "lykenox_voice_engine" / "ui" / "recording_v2_page.py").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()

    assert "recording_v2_raw_dir" in text
    assert "pilot_manifest.csv" in text
    assert "recording_id" in text
    assert "sf.write" in text
    assert "resample(" not in lowered
    assert "normalize(" not in lowered
    assert "denoise(" not in lowered
    assert "loudnorm" not in lowered
    assert "equalizer" not in lowered


def test_main_window_exposes_recording_v2_before_legacy_recorder() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "lykenox_voice_engine" / "ui" / "main_window.py").read_text(
        encoding="utf-8"
    )

    assert "from lykenox_voice_engine.ui.recording_v2_page import RecordingV2Page" in text
    assert '"Grabar RECORDING_V2"' in text
    assert '"Grabar Identidad Legacy"' in text
    assert text.index('"Grabar RECORDING_V2"') < text.index('"Grabar Identidad Legacy"')
    assert "stack.addWidget(RecordingV2Page(root))" in text
