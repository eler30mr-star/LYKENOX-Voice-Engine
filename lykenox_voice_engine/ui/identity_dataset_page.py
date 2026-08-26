"""Identity voice dataset recorder for speech and singing."""

from __future__ import annotations

import shutil
import struct
import wave
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl
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

from lykenox_voice_engine.core.identity_dataset import IdentityDatasetService

try:
    from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices, QMediaPlayer, QAudioOutput
except ImportError:
    QAudioFormat = None
    QAudioSource = None
    QMediaDevices = None
    QMediaPlayer = None
    QAudioOutput = None


class IdentityDatasetPage(QWidget):
    """Record full speech and singing phrases for the LYKENOX identity model."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.service = IdentityDatasetService(root)
        self.service.ensure_structure()
        self.recording = False
        self.audio_source = None
        self.io_device = None
        self.audio_buffer = bytearray()
        self.native_sample_rate = 48000
        self.temp_wav: Path | None = None
        self.record_ms = 0
        self.frames_received = 0
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        self.header = QLabel("Dataset de Identidad LYKENOX: habla y canto directo")
        self.header.setStyleSheet("font-weight: bold; font-size: 18px;")
        layout.addWidget(self.header)

        row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["speech", "singing"])
        self.mode.currentIndexChanged.connect(self._refresh)
        self.mic = QComboBox()
        self._load_mics()
        row.addWidget(QLabel("Modo:"))
        row.addWidget(self.mode)
        row.addWidget(QLabel("Micrófono:"))
        row.addWidget(self.mic)
        layout.addLayout(row)

        self.prompt = QLabel("")
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet("font-size: 22px; color: #0078d4; font-weight: bold;")
        self.instruction = QLabel("")
        self.instruction.setWordWrap(True)
        layout.addWidget(self.prompt)
        layout.addWidget(self.instruction)

        self.status = QLabel("Listo.")
        self.timer_label = QLabel("00:00")
        self.level = QLabel("Nivel: ----------")
        self.f0 = QLabel("F0 actual: -- Hz")
        layout.addWidget(self.timer_label)
        layout.addWidget(self.level)
        layout.addWidget(self.f0)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.btn_record = QPushButton("Grabar")
        self.btn_stop = QPushButton("Detener")
        self.btn_listen = QPushButton("Escuchar")
        self.btn_accept = QPushButton("Aceptar toma")
        self.btn_reject = QPushButton("Rechazar")
        for button in [self.btn_record, self.btn_stop, self.btn_listen, self.btn_accept, self.btn_reject]:
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.takes = QListWidget()
        layout.addWidget(self.takes)
        self.setLayout(layout)

        self.btn_record.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_listen.clicked.connect(self._listen)
        self.btn_accept.clicked.connect(self._accept)
        self.btn_reject.clicked.connect(self._reject)

    def _load_mics(self) -> None:
        self.mic.clear()
        if QMediaDevices is None:
            return
        for device in QMediaDevices.audioInputs():
            self.mic.addItem(device.description(), device)

    def _refresh(self) -> None:
        mode = self.mode.currentText()
        prompt = self.service.next_prompt(mode)
        self.current_prompt = prompt
        if mode == "speech":
            self.instruction.setText("Lee con tu voz natural, clara y tranquila. No actúes otra voz.")
        else:
            self.instruction.setText("Canta con tu identidad real. No conviertas ni imites; usa una melodía cómoda.")
        self.prompt.setText(prompt.text)
        self.takes.clear()
        summary = self.service.summary()
        for take in self.service.takes(mode):
            self.takes.addItem(
                f"{take.status} | {take.prompt_id} | {take.duration_sec:.1f}s | "
                f"F0 {take.measured_f0_hz:.1f} Hz | {Path(take.wav_path).name}"
            )
        self.status.setText(
            f"Speech aceptadas: {summary['speech_accepted']} | "
            f"Singing aceptadas: {summary['singing_accepted']}"
        )
        self._update_buttons()

    def _start(self) -> None:
        if self.recording:
            return
        if QAudioSource is None:
            self.status.setText("Audio no disponible.")
            return
        device = self.mic.currentData() or QMediaDevices.defaultAudioInput()
        fmt = QAudioFormat()
        fmt.setSampleRate(48000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Int16)
        if not device.isFormatSupported(fmt):
            fmt = device.preferredFormat()
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.Int16)
        self.native_sample_rate = fmt.sampleRate()
        self.audio_buffer = bytearray()
        self.record_ms = 0
        self.frames_received = 0
        self.audio_source = QAudioSource(device, fmt, self)
        self.audio_source.setBufferSize(self.native_sample_rate * 2 * 20)
        self.io_device = self.audio_source.start()
        self.recording = self.io_device is not None
        self.timer.start(50)
        self.status.setText("Grabando identidad...")
        self._update_buttons()

    def _stop(self) -> None:
        if not self.recording:
            return
        self._read_audio()
        self.timer.stop()
        if self.audio_source:
            self.audio_source.stop()
        self.recording = False
        raw_dir = self.service.base_dir / self.mode.currentText() / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.temp_wav = raw_dir / f"temp_{self.current_prompt.id}.wav"
        self._write_wav(self.temp_wav)
        self.status.setText(f"Toma lista: {self.temp_wav.name}")
        self._update_buttons()

    def _tick(self) -> None:
        self._read_audio()
        self.record_ms += 50
        seconds = self.record_ms // 1000
        self.timer_label.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")
        if len(self.audio_buffer) >= 4000:
            chunk = self.audio_buffer[-4000:]
            values = [struct.unpack_from("<h", chunk, i)[0] for i in range(0, len(chunk) - 1, 2)]
            rms = int((sum(value * value for value in values) / len(values)) ** 0.5)
            peak = max(abs(value) for value in values)
            bars = min(15, rms // 500)
            self.level.setText("Nivel: " + ("█" * bars) + ("-" * (15 - bars)))
            self.f0.setText(f"F0 actual: {_estimate_f0(values, self.native_sample_rate):.0f} Hz" if peak > 500 else "F0 actual: -- Hz")

    def _read_audio(self) -> None:
        if self.io_device and self.io_device.isOpen():
            data = self.io_device.readAll()
            if data.size() > 0:
                chunk = data.data()
                self.audio_buffer.extend(chunk)
                self.frames_received += len(chunk) // 2

    def _write_wav(self, path: Path) -> None:
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.native_sample_rate)
            writer.writeframes(bytes(self.audio_buffer))

    def _listen(self) -> None:
        if not self.temp_wav or not self.temp_wav.exists() or QMediaPlayer is None:
            return
        if not hasattr(self, "_player"):
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(self.temp_wav)))
        self._audio_out.setVolume(1.0)
        self._player.play()

    def _accept(self) -> None:
        if not self.temp_wav or not self.temp_wav.exists():
            return
        meta = self.service.register_take(self.mode.currentText(), self.current_prompt, self.temp_wav)
        self.status.setText(f"Toma {meta.status}: {meta.reason or 'ok'}")
        self.temp_wav = None
        self._refresh()

    def _reject(self) -> None:
        if self.temp_wav and self.temp_wav.exists():
            rejected = self.service.base_dir / self.mode.currentText() / "rejected" / self.temp_wav.name
            rejected.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.temp_wav), str(rejected))
        self.temp_wav = None
        self.status.setText("Toma rechazada.")
        self._refresh()

    def _update_buttons(self) -> None:
        has_take = self.temp_wav is not None and self.temp_wav.exists()
        self.btn_record.setEnabled(not self.recording)
        self.btn_stop.setEnabled(self.recording)
        self.btn_listen.setEnabled((not self.recording) and has_take)
        self.btn_accept.setEnabled((not self.recording) and has_take)
        self.btn_reject.setEnabled((not self.recording) and has_take)


def _estimate_f0(samples: list[int], sample_rate: int) -> float:
    if sample_rate <= 0 or len(samples) < 200:
        return 0.0
    crossings = 0
    previous = samples[0]
    for sample in samples[1:]:
        if (previous <= 0 < sample) or (previous >= 0 > sample):
            crossings += 1
        previous = sample
    duration = len(samples) / float(sample_rate)
    hz = crossings / (2.0 * duration) if duration else 0.0
    return hz if 60.0 <= hz <= 500.0 else 0.0
