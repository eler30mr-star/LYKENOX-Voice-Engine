"""Native RECORDING_V2 pilot recorder for clean identity recapture.

The canonical capture path is deliberately conservative. The app records an exact 48 kHz mono
high-resolution device stream and writes a WAV FLOAT candidate without denoise, resampling, gain
normalization, EQ, compression, dereverb, voice isolation, or other DSP. The user must listen to the
complete candidate before the canonical RAW save button is enabled. A lightweight environment meter
and technical preflight help detect bad recording conditions, but metrics never accept perceptual
quality. Policy: LYX-POL-001 v1.1.
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
ENVIRONMENT_SECONDS = 3


class RecordingV2Page(QWidget):
    """Record and audibly approve the clean RECORDING_V2 pilot from the desktop app."""

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
        self.capture_mode: str | None = None
        self.audio_source = None
        self.io_device = None
        self.audio_buffer = bytearray()
        self.capture_kind: str | None = None
        self.bytes_per_sample = 0
        self.candidate_path: Path | None = None
        self.candidate_listened = False
        self.candidate_technical_ok = False
        self.record_ms = 0
        self.environment_rms_dbfs: float | None = None

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
            "Captura canónica: WAV float32, mono, 48 kHz. El RAW se guarda sin denoise, AGC, EQ, "
            "compresión, limiter, dereverb, normalización, resampling ni aislamiento software. "
            "Desactiva también mejoras de micrófono en Windows/driver. Primero grabas, luego pruebas "
            "la toma completa, y solo entonces puedes guardarla."
        )
        rule.setWordWrap(True)
        layout.addWidget(rule)

        mic_row = QHBoxLayout()
        self.mic = QComboBox()
        self.mic.currentIndexChanged.connect(self._probe_selected_device)
        self.btn_environment = QPushButton(f"Medir ambiente {ENVIRONMENT_SECONDS}s")
        self.btn_environment.clicked.connect(self._start_environment_measurement)
        mic_row.addWidget(QLabel("Micrófono:"))
        mic_row.addWidget(self.mic, 1)
        mic_row.addWidget(self.btn_environment)
        layout.addLayout(mic_row)

        self.format_status = QLabel("Formato: comprobando...")
        self.format_status.setWordWrap(True)
        self.environment_status = QLabel(
            "Ambiente: no medido. Esta medición solo orienta; no limpia ni acepta audio."
        )
        self.environment_status.setWordWrap(True)
        layout.addWidget(self.format_status)
        layout.addWidget(self.environment_status)

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
            "motor, golpe, viento fuerte u otra voz durante la frase, NO guardes: vuelve a grabar."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        self.timer_label = QLabel("00:00")
        self.level = QLabel("Nivel: ---------- | pico -- dBFS")
        self.preflight = QLabel("Preflight: todavía no hay candidata.")
        self.preflight.setWordWrap(True)
        self.status = QLabel("Listo.")
        layout.addWidget(self.timer_label)
        layout.addWidget(self.level)
        layout.addWidget(self.preflight)
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
        self.btn_listen = QPushButton("Probar / escuchar completa")
        self.btn_discard = QPushButton("Descartar candidata")
        self.btn_commit = QPushButton("Guardar RAW aprobado")
        for button in (
            self.btn_record,
            self.btn_stop,
            self.btn_listen,
            self.btn_discard,
            self.btn_commit,
        ):
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
        self.btn_discard.clicked.connect(self._discard_candidate)
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
        self.environment_rms_dbfs = None
        self.environment_status.setText(
            "Ambiente: no medido con este micrófono. Esta medición no modifica el RAW."
        )
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
        self._clear_candidate_state(remove_file=False)
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
        self.recording_id.setText(f"Archivo final: {row['recording_id']}.wav")
        self.prompt.setText(row["text"])
        raw = self.raw_dir / f"{row['recording_id']}.wav"
        if raw.exists():
            self.status.setText(
                f"RAW ya guardado: {raw.name}. Puedes escucharlo o grabar una candidata nueva."
            )
        else:
            self.status.setText("Listo para grabar esta toma.")
        self._update_buttons()

    def _previous(self) -> None:
        if self.recording or not self.rows:
            return
        self.current_index = max(0, self.current_index - 1)
        self._clear_candidate_state(remove_file=False)
        self.items.setCurrentRow(self.current_index)
        self._refresh_prompt()

    def _next(self) -> None:
        if self.recording or not self.rows:
            return
        self.current_index = min(len(self.rows) - 1, self.current_index + 1)
        self._clear_candidate_state(remove_file=False)
        self.items.setCurrentRow(self.current_index)
        self._refresh_prompt()

    def _open_capture(self, mode: str) -> bool:
        if self.recording or QAudioSource is None or QMediaDevices is None:
            return False
        device = self.mic.currentData() or QMediaDevices.defaultAudioInput()
        fmt, kind, width = self._exact_capture_format(device)
        if fmt is None or kind is None:
            QMessageBox.critical(
                self,
                "Formato de captura no válido",
                "El micrófono seleccionado no soporta 48 kHz mono Float32/Int32 mediante Qt. "
                "LYKENOX no degradará el formato automáticamente.",
            )
            return False
        self.capture_kind = kind
        self.bytes_per_sample = width
        self.audio_buffer = bytearray()
        self.record_ms = 0
        self.timer_label.setText("00:00")
        self.audio_source = QAudioSource(device, fmt, self)
        self.audio_source.setBufferSize(TARGET_SAMPLE_RATE * width * 10)
        self.io_device = self.audio_source.start()
        self.recording = self.io_device is not None
        self.capture_mode = mode if self.recording else None
        if not self.recording:
            self.status.setText("No se pudo abrir el micrófono.")
            self._update_buttons()
            return False
        self.timer.start(50)
        self._update_buttons()
        return True

    def _start_environment_measurement(self) -> None:
        if not self._open_capture("environment"):
            return
        self.status.setText(
            f"Midiendo {ENVIRONMENT_SECONDS}s de ambiente. No hables durante esta prueba..."
        )
        QTimer.singleShot(ENVIRONMENT_SECONDS * 1000, self._finish_environment_measurement)

    def _finish_environment_measurement(self) -> None:
        if not self.recording or self.capture_mode != "environment":
            return
        samples = self._finish_capture_stream()
        if samples.size == 0:
            self.environment_status.setText("Ambiente: medición vacía; vuelve a medir.")
            return
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
        peak = float(np.max(np.abs(samples)))
        rms_db = self._dbfs(rms)
        peak_db = self._dbfs(peak)
        self.environment_rms_dbfs = rms_db
        if rms_db > -40.0:
            verdict = "ALTO: busca un momento/lugar más silencioso antes de grabar."
        elif rms_db > -50.0:
            verdict = "REVISAR: usable solo si al escuchar no se percibe contaminación."
        else:
            verdict = "BAJO: buen punto de partida; la escucha sigue mandando."
        self.environment_status.setText(
            f"Ambiente medido: RMS {rms_db:.1f} dBFS | pico {peak_db:.1f} dBFS | {verdict}"
        )
        self.status.setText("Medición de ambiente terminada. No se guardó audio.")
        self._update_buttons()

    def _start(self) -> None:
        if not self.rows:
            return
        self._clear_candidate_state(remove_file=True)
        if self._open_capture("voice"):
            self.preflight.setText("Preflight: grabando candidata...")
            self.status.setText("Grabando candidata RAW... sin procesamiento.")

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

    def _finish_capture_stream(self) -> np.ndarray:
        self._read_audio()
        self.timer.stop()
        if self.audio_source:
            self.audio_source.stop()
        QCoreApplication.processEvents()
        self._read_audio()
        self.recording = False
        self.capture_mode = None
        samples = self._decode_buffer()
        self._update_buttons()
        return samples

    def _stop(self) -> None:
        if not self.recording:
            return
        if self.capture_mode == "environment":
            self._finish_environment_measurement()
            return
        samples = self._finish_capture_stream()
        if samples.size == 0:
            self.status.setText("Toma vacía; vuelve a grabar.")
            self.preflight.setText("Preflight: FAIL — toma vacía.")
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
        self.candidate_listened = False
        self.candidate_technical_ok, summary = self._candidate_preflight(self.candidate_path)
        self.preflight.setText(summary)
        info = sf.info(str(self.candidate_path))
        self.status.setText(
            f"Candidata lista: {info.samplerate} Hz | {info.channels} ch | {info.subtype}. "
            "Ahora pulsa Probar / escuchar completa. Guardar seguirá bloqueado hasta terminar la escucha."
        )
        self._update_buttons()

    @staticmethod
    def _dbfs(value: float) -> float:
        return float(20.0 * np.log10(max(float(value), 1.0e-12)))

    def _candidate_preflight(self, path: Path) -> tuple[bool, str]:
        try:
            info = sf.info(str(path))
            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            return False, f"Preflight: FAIL — no se pudo leer candidata: {exc}"

        failures: list[str] = []
        warnings: list[str] = []
        if int(sample_rate) != TARGET_SAMPLE_RATE or int(info.samplerate) != TARGET_SAMPLE_RATE:
            failures.append("sample rate != 48000")
        if int(info.channels) != TARGET_CHANNELS or int(audio.shape[1]) != TARGET_CHANNELS:
            failures.append("no es mono")
        if str(info.subtype) != OUTPUT_SUBTYPE:
            failures.append("subtipo != FLOAT")
        if audio.size == 0 or not bool(np.isfinite(audio).all()):
            failures.append("audio vacío/no finito")

        if audio.size:
            mono = audio[:, 0].astype(np.float64, copy=False)
            duration = float(info.frames) / float(info.samplerate)
            peak = float(np.max(np.abs(mono)))
            rms = float(np.sqrt(np.mean(np.square(mono))))
            clipping_fraction = float(np.mean(np.abs(mono) >= 0.999))
            peak_db = self._dbfs(peak)
            rms_db = self._dbfs(rms)
            if duration < 0.50:
                failures.append("duración < 0.5 s")
            if clipping_fraction > 0.001:
                failures.append("clipping excesivo")
            elif clipping_fraction > 0.0:
                warnings.append("posibles muestras recortadas")
            if peak_db > -3.0:
                warnings.append(f"pico alto {peak_db:.1f} dBFS")
            if peak_db < -24.0:
                warnings.append(f"pico bajo {peak_db:.1f} dBFS")
            if rms_db < -40.0:
                warnings.append(f"RMS bajo {rms_db:.1f} dBFS")

        if failures:
            return False, "Preflight: FAIL — " + "; ".join(failures)
        suffix = " | avisos: " + "; ".join(warnings) if warnings else ""
        return True, "Preflight: PASS técnico. Falta aprobación auditiva completa." + suffix

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
                peak_db = self._dbfs(peak)
                bars = min(15, max(0, int((self._dbfs(rms) + 60.0) / 4.0)))
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

    def _ensure_player(self) -> bool:
        if QMediaPlayer is None or QAudioOutput is None:
            return False
        if not hasattr(self, "_player"):
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)
            self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        return True

    def _listen(self) -> None:
        path = self._selected_audio_path()
        if path is None or not self._ensure_player():
            return
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._audio_out.setVolume(1.0)
        self._player.play()
        info = sf.info(str(path))
        if self.candidate_path is not None and path.resolve() == self.candidate_path.resolve():
            self.candidate_listened = False
            self.status.setText(
                f"Probando candidata completa: {info.duration:.2f}s. Guardar se habilita al terminar."
            )
        else:
            self.status.setText(
                f"Reproduciendo RAW guardado: {path.name} | {info.duration:.2f}s | {info.subtype}"
            )
        self._update_buttons()

    def _on_media_status_changed(self, status) -> None:
        if QMediaPlayer is None:
            return
        if status != QMediaPlayer.MediaStatus.EndOfMedia:
            return
        if self.candidate_path is None or not self.candidate_path.exists():
            return
        self.candidate_listened = True
        if self.candidate_technical_ok:
            self.status.setText(
                "Escucha completa terminada. Si tú confirmas que está limpia y natural, pulsa Guardar RAW aprobado."
            )
        else:
            self.status.setText(
                "Escucha completa terminada, pero el preflight técnico falló: vuelve a grabar."
            )
        self._update_buttons()

    def _discard_candidate(self) -> None:
        if self.recording:
            return
        self._clear_candidate_state(remove_file=True)
        self.preflight.setText("Preflight: candidata descartada. Graba una nueva toma.")
        self.status.setText("Candidata descartada; no entró al dataset.")
        self._update_buttons()

    def _clear_candidate_state(self, *, remove_file: bool) -> None:
        if remove_file and self.candidate_path is not None:
            self.candidate_path.unlink(missing_ok=True)
        self.candidate_path = None
        self.candidate_listened = False
        self.candidate_technical_ok = False

    def _commit_raw(self) -> None:
        if (
            self.candidate_path is None
            or not self.candidate_path.exists()
            or not self.rows
            or not self.candidate_listened
            or not self.candidate_technical_ok
        ):
            return

        answer = QMessageBox.question(
            self,
            "Confirmar toma limpia",
            "¿Confirmas que escuchaste la toma completa y que la voz suena limpia, natural y sin "
            "carro, gallo, motor, golpe, viento fuerte u otra voz superpuesta?",
        )
        if answer != QMessageBox.Yes:
            return

        row = self.rows[self.current_index]
        raw = self.raw_dir / f"{row['recording_id']}.wav"
        if raw.exists():
            replace = QMessageBox.question(
                self,
                "Reemplazar RAW actual",
                "Ya existe un RAW para este ID. El anterior se archivará byte a byte en raw_history "
                "y esta candidata pasará a ser el RAW actual. ¿Continuar?",
            )
            if replace != QMessageBox.Yes:
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

        self.status.setText(
            f"RAW aprobado guardado: {raw}. WAV FLOAT32 mono 48 kHz, sin procesamiento."
        )
        self._clear_candidate_state(remove_file=True)
        self.preflight.setText("Preflight: RAW aprobado y guardado.")
        self._refresh_item_list()
        self._refresh_prompt()

    def _update_buttons(self) -> None:
        has_rows = bool(self.rows)
        format_ok = self.capture_kind in {"float32", "int32"}
        has_audio = self._selected_audio_path() is not None
        has_candidate = self.candidate_path is not None and self.candidate_path.exists()
        save_ready = has_candidate and self.candidate_listened and self.candidate_technical_ok

        self.btn_prev.setEnabled(has_rows and not self.recording and self.current_index > 0)
        self.btn_next.setEnabled(
            has_rows and not self.recording and self.current_index < max(0, len(self.rows) - 1)
        )
        self.btn_environment.setEnabled((not self.recording) and format_ok)
        self.btn_record.setEnabled(has_rows and not self.recording and format_ok)
        self.btn_stop.setEnabled(self.recording)
        self.btn_listen.setEnabled((not self.recording) and has_audio)
        self.btn_discard.setEnabled((not self.recording) and has_candidate)
        self.btn_commit.setEnabled((not self.recording) and save_ready)


__all__ = [
    "RecordingV2Page",
    "TARGET_SAMPLE_RATE",
    "TARGET_CHANNELS",
    "OUTPUT_SUBTYPE",
    "ENVIRONMENT_SECONDS",
]
