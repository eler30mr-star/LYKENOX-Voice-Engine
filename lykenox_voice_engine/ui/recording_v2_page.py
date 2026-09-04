"""Native RECORDING_V2 pilot recorder for clean identity recapture.

This page records only the currently prepared 10-take RECORDING_V2 pilot. Capture is strict:
48 kHz, mono, Float32 or Int32 device input, written as WAV FLOAT without gain normalization,
resampling, denoise, EQ, compression, dereverb or other DSP. If the selected microphone cannot
provide an exact supported 48 kHz mono high-resolution format, recording is blocked rather than
silently falling back to a lower-quality geometry.

Policy: LYX-POL-001 v1.1.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QCoreApplication, QTimer, QUrl
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lykenox_voice_engine.training.identity_voice_recording_v2 import (
    recording_v2_raw_dir,
    recording_v2_root,
    recording_v2_session_manifest,
)

try:
    from PySide6.QtMultimedia import (
        QAudioFormat,
        QAudioOutput,
        QAudioSource,
        QMediaDevices,
        QMediaPlayer,
    )
except ImportError:
    QAudioFormat = None
    QAudioOutput = None
    QAudioSource = None
    QMediaDevices = None
    QMediaPlayer = None


TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1
OUTPUT_SUBTYPE = "FLOAT"
PILOT_FILENAME = "pilot_manifest.csv"


class RecordingV2Page(QWidget):
    """Record the clean RECORDING_V2 pilot directly from the desktop app."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = Path(root).resolve()
        self.raw_dir = recording_v2_raw_dir(self.root)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = recording_v2_root(self.root) / "capture_work"
        self.history_dir = recording_v2_root(self.root) / "raw_history"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.rows: list[dict[str, str]] = []
        self.current_index = 0
        self.recording = False
        self.audio_source = None
        self.io_device = None
        self.audio_buffer = bytearray()
        self.capture_kind: str | None = None
        self.bytes_per_sample = 0
        self.candidate_path: Path | None = None
        self.record_ms = 0

        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._load_pilot()
        self._load_mics()
        self._refresh_prompt()
        self._probe_selected_device()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        header = QLabel("RECORDING_V2 — Piloto limpio de identidad (10 tomas)")
        header.setStyleSheet("font-weight: bold; font-size: 18px;")
        layout.addWidget(header)

        rule = QLabel(
            "Captura estricta: WAV float32, mono, 48 kHz. LYKENOX no aplica denoise, AGC, EQ, "
            "compresión, limiter, dereverb, normalización ni resampling. Desactiva también las "
            "mejoras del micrófono en Windows/driver si existen."
        )
        rule.setWordWrap(True)
        layout.addWidget(rule)

        mic_row = QHBoxLayout()
        self.mic = QComboBox()
        self.mic.currentIndexChanged.connect(self._probe_selected_device)
        mic_row.addWidget(QLabel("Micrófono:"))
        mic_row.addWidget(self.mic, 1)
        layout.addLayout(mic_row)

        self.format_status = QLabel("Formato: comprobando...")
        self.format_status.setWordWrap(True)
        layout.addWidget(self.format_status)

        self.progress = QLabel("")
        self.recording_id = QLabel("")
        self.recording_id.setStyleSheet("font-weight: bold;")
        self.prompt = QLabel("")
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet("font-size: 22px; color: #0078d4; font-weight: bold;")
        layout.addWidget(self.progress)
        layout.addWidget(self.recording_id)
        layout.addWidget(self.prompt)

        instruction = QLabel(
            "Lee con voz natural. Deja un poco de ambiente antes y después. Si entra carro, gallo, "
            "motor, golpe, viento fuerte u otra voz durante la frase, repite la toma."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        self.timer_label = QLabel("00:00")
        self.level = QLabel("Nivel: ---------- | pico -- dBFS")
        self.status = QLabel("Listo.")
        layout.addWidget(self.timer_label)
        layout.addWidget(self.level)
        layout.addWidget(self.status)

        nav_buttons = QHBoxLayout()
        self.btn_prev = QPushButton("Anterior")
        self.btn_next = QPushButton("Siguiente")
        nav_buttons.addWidget(self.btn_prev)
        nav_buttons.addWidget(self.btn_next)
        layout.addLayout(nav_buttons)

        capture_buttons = QHBoxLayout()
        self.btn_record = QPushButton("Grabar")
        self.btn_stop = QPushButton("Detener")
        self.btn_listen = QPushButton("Escuchar toma")
        self.btn_commit = QPushButton("Guardar RAW")
        for button in (self.btn_record, self.btn_stop, self.btn_listen, self.btn_commit):
            capture_buttons.addWidget(button)
        layout.addLayout(capture_buttons)

        self.items = QListWidget()
        self.items.currentRowChanged.connect(self._select_row)
        layout.addWidget(self.items)

        self.setLayout(layout)

        self.btn_prev.clicked.connect(self._previous)
        self.btn_next.clicked.connect(self._next)
        self.btn_record.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_listen.clicked.connect(self._listen)
        self.btn_commit.clicked.connect(self._commit_raw)

    def _pilot_manifest(self) -> Path:
        return recording_v2_session_manifest(self.root).parent / PILOT_FILENAME

    def _load_pilot(self) -> None:
        path = self._pilot_manifest()
        if not path.exists():
            self.rows = []
            self.status.setText(
                "Falta pilot_manifest.csv. Ejecuta prepare_identity_voice_recording_v2_pilot.py."
            )
            return
        with path.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if len(self.rows) != 10:
            self.status.setText(f"Piloto inválido: se esperaban 10 filas y hay {len(self.rows)}.")
        self.current_index = 0
        self._refresh_item_list()

    def _refresh_item_list(self) -> None:
        self.items.blockSignals(True)
        self.items.clear()
        for row in self.rows:
            raw = self.raw_dir / f"{row['recording_id']}.wav"
            marker = "RAW" if raw.exists() else "PENDIENTE"
            self.items.addItem(
                f"{int(row['pilot_order']):02d} | {marker} | {row['recording_id']} | {row['split']}"
            )
        if self.rows:
            self.items.setCurrentRow(self.current_index)
        self.items.blockSignals(False)

    def _load_mics(self) -> None:
        self.mic.clear()
        if QMediaDevices is None:
            self.format_status.setText("QtMultimedia no está disponible.")
            return
        for device in QMediaDevices.audioInputs():
            self.mic.addItem(device.description(), device)
        if self.mic.count() == 0:
            self.format_status.setText("No se detectó ningún micrófono.")

    def _exact_capture_format(self, device):
        if QAudioFormat is None or device is None:
            return None, None, 0
        for sample_format, kind, width in (
            (QAudioFormat.Float, "float32", 4),
            (QAudioFormat.Int32, "int32", 4),
        ):
            fmt = QAudioFormat()
            fmt.setSampleRate(TARGET_SAMPLE_RATE)
            fmt.setChannelCount(TARGET_CHANNELS)
            fmt.setSampleFormat(sample_format)
            if device.isFormatSupported(fmt):
                return fmt, kind, width
        return None, None, 0

    def _probe_selected_device(self) -> None:
        if QMediaDevices is None or self.mic.count() == 0:
            self.capture_kind = None
            self.bytes_per_sample = 0
            self._update_buttons()
            return
        device = self.mic.currentData() or QMediaDevices.defaultAudioInput()
        _fmt, kind, width = self._exact_capture_format(device)
        self.capture_kind = kind
        self.bytes_per_sample = width
        if kind is None:
            self.format_status.setText(
                "BLOQUEADO: este micrófono no expone 48 kHz mono Float32/Int32 a Qt. "
                "No se hará fallback silencioso a 16-bit u otra frecuencia."
            )
        else:
            self.format_status.setText(
                f"OK: entrada {TARGET_SAMPLE_RATE} Hz mono {kind}; salida WAV {OUTPUT_SUBTYPE} 32-bit. "
                "Sin DSP de LYKENOX."
            )
        self._update_buttons()

    def _select_row(self, row: int) -> None:
        if self.recording or not (0 <= row < len(self.rows)):
            return
        self.current_index = row
        self.candidate_path = None
        self._refresh_prompt()

    def _refresh_prompt(self) -> None:
        if not self.rows:
            self.progress.setText("Piloto no disponible.")
            self.recording_id.setText("")
            self.prompt.setText("")
            self._update_buttons()
            return
        row = self.rows[self.current_index]
        saved = sum(
            (self.raw_dir / f"{item['recording_id']}.wav").exists() for item in self.rows
        )
        self.progress.setText(
            f"Toma {self.current_index + 1}/10 | RAW guardados {saved}/10 | split={row['split']}"
        )
        self.recording_id.setText(f"Archivo: {row['recording_id']}.wav")
        self.prompt.setText(row["text"])
        raw = self.raw_dir / f"{row['recording_id']}.wav"
        if raw.exists():
            self.status.setText(f"RAW ya guardado: {raw.name}. Puedes escuchar o hacer una nueva candidata.")
        else:
            self.status.setText("Listo para grabar esta toma.")
        self._update_buttons()

    def _previous(self) -> None:
        if self.recording or not self.rows:
            return
        self.current_index = max(0, self.current_index - 1)
        self.candidate_path = None
        self.items.setCurrentRow(self.current_index)
        self._refresh_prompt()

    def _next(self) -> None:
        if self.recording or not self.rows:
            return
        self.current_index = min(len(self.rows) - 1, self.current_index + 1)
        self.candidate_path = None
        self.items.setCurrentRow(self.current_index)
        self._refresh_prompt()

    def _start(self) -> None:
        if self.recording or not self.rows or QAudioSource is None or QMediaDevices is None:
            return
        device = self.mic.currentData() or QMediaDevices.defaultAudioInput()
        fmt, kind, width = self._exact_capture_format(device)
        if fmt is None or kind is None:
            QMessageBox.critical(
                self,
                "Formato de captura no válido",
                "El micrófono seleccionado no soporta 48 kHz mono Float32/Int32 mediante Qt. "
                "LYKENOX no degradará el formato automáticamente.",
            )
            return
        self.capture_kind = kind
        self.bytes_per_sample = width
        self.audio_buffer = bytearray()
        self.record_ms = 0
        self.timer_label.setText("00:00")
        self.audio_source = QAudioSource(device, fmt, self)
        self.audio_source.setBufferSize(TARGET_SAMPLE_RATE * width * 10)
        self.io_device = self.audio_source.start()
        self.recording = self.io_device is not None
        if not self.recording:
            self.status.setText("No se pudo abrir el micrófono.")
            self._update_buttons()
            return
        self.timer.start(50)
        self.status.setText("Grabando RAW candidato... sin procesamiento.")
        self._update_buttons()

    def _read_audio(self) -> None:
        if self.io_device and self.io_device.isOpen():
            data = self.io_device.readAll()
            if data.size() > 0:
                self.audio_buffer.extend(data.data())

    def _decode_buffer(self, data: bytes | bytearray | None = None) -> np.ndarray:
        source = bytes(self.audio_buffer if data is None else data)
        if self.bytes_per_sample != 4 or not source:
            return np.zeros(0, dtype=np.float32)
        usable = (len(source) // 4) * 4
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        view = memoryview(source)[:usable]
        if self.capture_kind == "float32":
            return np.frombuffer(view, dtype="<f4").astype(np.float32, copy=True)
        if self.capture_kind == "int32":
            values = np.frombuffer(view, dtype="<i4").astype(np.float32)
            return values / np.float32(2147483648.0)
        return np.zeros(0, dtype=np.float32)

    def _stop(self) -> None:
        if not self.recording:
            return
        self._read_audio()
        self.timer.stop()
        if self.audio_source:
            self.audio_source.stop()
        QCoreApplication.processEvents()
        self._read_audio()
        self.recording = False

        samples = self._decode_buffer()
        if samples.size == 0:
            self.status.setText("Toma vacía; vuelve a grabar.")
            self._update_buttons()
            return
        row = self.rows[self.current_index]
        self.candidate_path = self.work_dir / f"{row['recording_id']}__candidate.wav"
        sf.write(
            str(self.candidate_path),
            samples,
            TARGET_SAMPLE_RATE,
            format="WAV",
            subtype=OUTPUT_SUBTYPE,
        )
        info = sf.info(str(self.candidate_path))
        self.status.setText(
            f"Candidata lista: {self.candidate_path.name} | {info.samplerate} Hz | "
            f"{info.channels} ch | {info.subtype}. Escúchala y luego Guardar RAW."
        )
        self._update_buttons()

    def _tick(self) -> None:
        self._read_audio()
        self.record_ms += 50
        seconds = self.record_ms // 1000
        self.timer_label.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")
        if len(self.audio_buffer) >= max(4096, self.bytes_per_sample * 1024):
            tail_bytes = self.bytes_per_sample * 2048
            samples = self._decode_buffer(self.audio_buffer[-tail_bytes:])
            if samples.size:
                rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
                peak = float(np.max(np.abs(samples)))
                peak_db = 20.0 * np.log10(max(peak, 1.0e-12))
                bars = min(15, max(0, int((20.0 * np.log10(max(rms, 1.0e-12)) + 60.0) / 4.0)))
                self.level.setText(
                    "Nivel: " + ("█" * bars) + ("-" * (15 - bars)) + f" | pico {peak_db:.1f} dBFS"
                )

    def _selected_audio_path(self) -> Path | None:
        if self.candidate_path and self.candidate_path.exists():
            return self.candidate_path
        if not self.rows:
            return None
        raw = self.raw_dir / f"{self.rows[self.current_index]['recording_id']}.wav"
        return raw if raw.exists() else None

    def _listen(self) -> None:
        path = self._selected_audio_path()
        if path is None or QMediaPlayer is None or QAudioOutput is None:
            return
        if not hasattr(self, "_player"):
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._audio_out.setVolume(1.0)
        self._player.play()
        info = sf.info(str(path))
        self.status.setText(
            f"Reproduciendo {path.name} | {info.duration:.2f}s | {info.samplerate} Hz | {info.subtype}"
        )

    def _commit_raw(self) -> None:
        if self.candidate_path is None or not self.candidate_path.exists() or not self.rows:
            return
        row = self.rows[self.current_index]
        raw = self.raw_dir / f"{row['recording_id']}.wav"
        if raw.exists():
            answer = QMessageBox.question(
                self,
                "Reemplazar RAW actual",
                "Ya existe un RAW para este ID. El archivo anterior se archivará sin modificar sus "
                "bytes y la candidata pasará a ser el RAW actual. ¿Continuar?",
            )
            if answer != QMessageBox.Yes:
                return
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archived = self.history_dir / f"{row['recording_id']}__{stamp}.wav"
            shutil.move(str(raw), str(archived))
        shutil.copy2(str(self.candidate_path), str(raw))
        info = sf.info(str(raw))
        if (
            int(info.samplerate) != TARGET_SAMPLE_RATE
            or int(info.channels) != TARGET_CHANNELS
            or str(info.subtype) != OUTPUT_SUBTYPE
        ):
            raw.unlink(missing_ok=True)
            raise RuntimeError("RECORDING_V2 UI wrote an invalid canonical RAW geometry")
        self.status.setText(f"RAW guardado: {raw.name}. No se aplicó procesamiento.")
        self.candidate_path = None
        self._refresh_item_list()
        self._refresh_prompt()

    def _update_buttons(self) -> None:
        has_rows = bool(self.rows)
        format_ok = self.capture_kind in {"float32", "int32"}
        has_audio = self._selected_audio_path() is not None
        has_candidate = self.candidate_path is not None and self.candidate_path.exists()
        self.btn_prev.setEnabled(has_rows and not self.recording and self.current_index > 0)
        self.btn_next.setEnabled(
            has_rows and not self.recording and self.current_index < max(0, len(self.rows) - 1)
        )
        self.btn_record.setEnabled(has_rows and not self.recording and format_ok)
        self.btn_stop.setEnabled(self.recording)
        self.btn_listen.setEnabled((not self.recording) and has_audio)
        self.btn_commit.setEnabled((not self.recording) and has_candidate)


__all__ = [
    "RecordingV2Page",
    "TARGET_SAMPLE_RATE",
    "TARGET_CHANNELS",
    "OUTPUT_SUBTYPE",
]
