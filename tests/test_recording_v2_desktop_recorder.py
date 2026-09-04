from __future__ import annotations

from pathlib import Path


def _page_text() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "lykenox_voice_engine" / "ui" / "recording_v2_page.py").read_text(
        encoding="utf-8"
    )


def test_recording_v2_page_uses_strict_high_resolution_capture() -> None:
    text = _page_text()

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


def test_recording_v2_requires_full_listen_before_canonical_save() -> None:
    text = _page_text()

    assert "candidate_listened" in text
    assert "candidate_technical_ok" in text
    assert "QMediaPlayer.MediaStatus.EndOfMedia" in text
    assert "Guardar RAW aprobado" in text
    assert "Probar / escuchar completa" in text
    assert "Confirmar toma limpia" in text
    assert "save_ready = has_candidate and self.candidate_listened and self.candidate_technical_ok" in text
    assert "recording_v2_raw_dir" in text
    assert "recording_id" in text


def test_recording_v2_environment_meter_does_not_process_raw() -> None:
    text = _page_text()
    lowered = text.lower()

    assert "ENVIRONMENT_SECONDS = 3" in text
    assert "Medir ambiente" in text
    assert "environment_rms_dbfs" in text
    assert "Esta medición no modifica el RAW" in text
    assert "metrics" not in lowered or "metrics never accept" in lowered
    assert "resample(" not in lowered
    assert "normalize(" not in lowered
    assert "denoise(" not in lowered
    assert "loudnorm" not in lowered
    assert "equalizer" not in lowered
    assert "deepfilternet" not in lowered


def test_recording_v2_preflight_checks_geometry_and_clipping() -> None:
    text = _page_text()

    assert "_candidate_preflight" in text
    assert "sample rate != 48000" in text
    assert "no es mono" in text
    assert "subtipo != FLOAT" in text
    assert "clipping excesivo" in text
    assert "Preflight: PASS técnico. Falta aprobación auditiva completa." in text


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
