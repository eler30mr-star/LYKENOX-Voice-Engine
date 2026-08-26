"""Voicebank creation page for Spanish Lite recordings."""

from __future__ import annotations

import logging
import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lykenox_voice_engine.core.pcm import peak, rms
from lykenox_voice_engine.core.voicebank import VoicebankManager
from lykenox_voice_engine.engines.utau_engine import UtauSampleEngine
from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine
from lykenox_voice_engine.core.multipitch import design_pitch_centers, report_to_dict, midi_to_note, hz_to_midi

try:
    from PySide6.QtMultimedia import (
        QAudio,
        QAudioFormat,
        QAudioOutput,
        QAudioSource,
        QMediaDevices,
        QMediaPlayer,
    )
except ImportError:
    QAudio = None
    QAudioFormat = None
    QAudioOutput = None
    QAudioSource = None
    QMediaDevices = None
    QMediaPlayer = None

logger = logging.getLogger(__name__)


class VoicebankPage(QWidget):
    """Create and validate a Spanish Lite sample voicebank."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.engine = UtauSampleEngine(root)
        self.manager = VoicebankManager(root)
        self.reclist = self.manager.load_reclist()
        self.current_index = 0

        # Recording state machine: recording -> stopping -> idle
        self.recording = False
        self.stopping = False
        self.audio_source = None
        self.io_device = None
        self.audio_buffer = bytearray()
        self.record_time_ms = 0
        self.frames_received = 0
        self.temp_wav = None
        self.current_session_id = 0
        self.native_sample_rate = 0
        self.is_testing = False
        self.test_log = []  # List of (time_s, rms, peak)

        # Stall / silence detection
        self._prev_frames = 0
        self._stall_ticks = 0
        self._low_rms_ticks = 0

        # UI
        self.header = QLabel("Perfil: LYKENOX Voice | Voicebank: LYKENOX Spanish Lite")
        self.header.setStyleSheet("font-weight: bold;")
        self.progress_label = QLabel("Progreso: 0 / 0")
        self.layer_selector = QComboBox()
        self.layer_selector.addItems(["Low", "Mid", "High"])
        self.layer_selector.currentIndexChanged.connect(self._on_layer_changed)
        self.target_note_label = QLabel("Guía registro: --")
        self.target_freq_label = QLabel("Referencia: -- Hz")
        self.layer_instruction = QLabel("Graba con emisión natural.")
        self.layer_instruction.setWordWrap(True)
        self.btn_guide_tone = QPushButton("Tono guía")

        self.mic_selector = QComboBox()
        self.mic_info = QLabel("Cargando información de dispositivo...")
        self.mic_info.setWordWrap(True)
        self.mic_info.setStyleSheet("color: #666; font-size: 11px;")
        self._populate_mics()

        self.rec_indicator = QLabel("○ STANDBY")
        self.rec_indicator.setStyleSheet("font-size: 18px; color: gray;")

        self.current_prompt = QLabel("Fonema actual: ---")
        self.current_prompt.setStyleSheet(
            "font-size: 24px; color: #0078d4; font-weight: bold;"
        )

        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet("font-family: monospace; font-size: 16px;")

        self.level_meter = QLabel("Nivel: ----------")
        self.live_f0_label = QLabel("F0 actual: -- Hz")
        self.status = QLabel("Listo.")

        self.aliases_list = QListWidget()
        self.aliases_list.currentRowChanged.connect(self._on_alias_selected)

        # Buttons
        self.btn_record = QPushButton("Grabar")
        self.btn_stop = QPushButton("Detener grabación")
        self.btn_test_mic = QPushButton("Probar micrófono")
        self.btn_calibrate = QPushButton("Calibrar voz")
        self.btn_listen = QPushButton("Escuchar")
        self.btn_save = QPushButton("Guardar")
        self.btn_delete = QPushButton("Borrar")
        self.btn_next = QPushButton("Siguiente")
        self.btn_prepare = QPushButton("Preparar voicebank")
        self.btn_validate = QPushButton("Validar voicebank")

        self._setup_ui()

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self._refresh_alias_list()
        self._update_ui_state()

    # ── UI setup ────────────────────────────────────────────────

    def _populate_mics(self) -> None:
        if QMediaDevices is None:
            return
        self.mic_selector.clear()
        devices = QMediaDevices.audioInputs()
        logger.info(f"Detectados {len(devices)} dispositivos de entrada:")

        default_device = QMediaDevices.defaultAudioInput()

        for i, device in enumerate(devices):
            desc = device.description()
            is_default = (device == default_device)
            self.mic_selector.addItem(desc, device)

            # Log detailed info
            mode = "PREDETERMINADO" if is_default else "DISPONIBLE"
            rates = device.minimumSampleRate(), device.maximumSampleRate()
            channels = device.minimumChannelCount(), device.maximumChannelCount()

            logger.info(
                f"[{i}] {desc} | {mode}\n"
                f"    ID: {device.id().data().decode(errors='replace') if hasattr(device.id(), 'data') else device.id()}\n"
                f"    Rates: {rates[0]}-{rates[1]} Hz | Channels: {channels[0]}-{channels[1]}"
            )

            if is_default:
                self.mic_selector.setCurrentIndex(i)

        self.mic_selector.currentIndexChanged.connect(self._on_mic_changed)
        self._on_mic_changed()

    def _on_mic_changed(self) -> None:
        device = self.mic_selector.currentData()
        if not device or device.isNull():
            self.mic_info.setText("Sin dispositivo seleccionado.")
            return

        rates = device.minimumSampleRate(), device.maximumSampleRate()
        channels = device.minimumChannelCount(), device.maximumChannelCount()

        info_text = (
            f"ID: {device.id().data().decode(errors='replace') if hasattr(device.id(), 'data') else str(device.id())}\n"
            f"Frecuencias: {rates[0]} - {rates[1]} Hz | Canales: {channels[0]} - {channels[1]}"
        )
        self.mic_info.setText(info_text)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        # Top bar
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.header)
        top_bar.addStretch()
        top_bar.addWidget(self.progress_label)
        layout.addLayout(top_bar)

        # Mic selector
        mic_layout = QVBoxLayout()
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Micrófono:"))
        sel_row.addWidget(self.mic_selector)
        mic_layout.addLayout(sel_row)
        mic_layout.addWidget(self.mic_info)
        layout.addLayout(mic_layout)

        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Capa:"))
        layer_row.addWidget(self.layer_selector)
        layer_row.addWidget(self.target_note_label)
        layer_row.addWidget(self.target_freq_label)
        layer_row.addWidget(self.btn_guide_tone)
        layer_row.addStretch()
        layout.addLayout(layer_row)
        layout.addWidget(self.layer_instruction)

        # Center recording area
        center = QVBoxLayout()
        center.setAlignment(Qt.AlignCenter)
        center.addWidget(self.rec_indicator, alignment=Qt.AlignCenter)
        center.addWidget(self.current_prompt, alignment=Qt.AlignCenter)
        center.addWidget(self.timer_label, alignment=Qt.AlignCenter)
        center.addWidget(self.level_meter, alignment=Qt.AlignCenter)
        center.addWidget(self.live_f0_label, alignment=Qt.AlignCenter)
        layout.addLayout(center)

        layout.addWidget(self.aliases_list)

        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_record)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_test_mic)
        btn_layout.addWidget(self.btn_calibrate)
        btn_layout.addWidget(self.btn_listen)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addWidget(self.btn_next)
        btn_layout.addWidget(self.btn_prepare)
        btn_layout.addWidget(self.btn_validate)
        layout.addLayout(btn_layout)

        layout.addWidget(self.status)
        self.setLayout(layout)

        # Connections
        self.btn_record.clicked.connect(self._start_recording)
        self.btn_stop.clicked.connect(self._stop_recording)
        self.btn_test_mic.clicked.connect(self._test_microphone)
        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        self.btn_listen.clicked.connect(self._listen_current)
        self.btn_save.clicked.connect(self._save_current)
        self.btn_delete.clicked.connect(self._delete_current)
        self.btn_next.clicked.connect(self._next_phoneme)
        self.btn_prepare.clicked.connect(self._prepare_voicebank)
        self.btn_validate.clicked.connect(self._validate_voicebank)
        self.btn_guide_tone.clicked.connect(self._play_guide_tone)

        # Styles
        self.btn_record.setStyleSheet(
            "background-color: #d83b01; color: white; font-weight: bold; min-height: 40px;"
        )
        self.btn_stop.setStyleSheet(
            "background-color: #a80000; color: white; font-weight: bold; min-height: 40px;"
        )
        self.btn_test_mic.setStyleSheet(
            "background-color: #0078d4; color: white; font-weight: bold; min-height: 40px;"
        )
        self.btn_calibrate.setStyleSheet(
            "background-color: #68217a; color: white; font-weight: bold; min-height: 40px;"
        )
        self.btn_prepare.setStyleSheet(
            "background-color: #107c10; color: white; font-weight: bold; min-height: 40px;"
        )
        self.btn_validate.setStyleSheet(
            "background-color: #0078d4; color: white; font-weight: bold; min-height: 40px;"
        )

    # ── State management ────────────────────────────────────────

    def _on_alias_selected(self, row: int) -> None:
        if row >= 0 and not self.recording and not self.stopping:
            self.current_index = row
            self._update_ui_state()

    def _on_layer_changed(self) -> None:
        if not self.recording and not self.stopping:
            self._refresh_alias_list()
            self._update_ui_state()

    def _test_microphone(self) -> None:
        self.is_testing = True
        self.test_log = []
        self._start_recording()

    def _prepare_voicebank(self) -> None:
        """Process saved WAVs and build the final voicebank structure."""
        try:
            res = self.manager.build_voicebank()
            msg = (
                f"Voicebank preparado con éxito.\n\n"
                f"Muestras procesadas: {res['accepted']}\n"
                f"Muestras rechazadas: {len(res['rejected'])}\n"
                f"Archivo OTO generado: {Path(res['oto']).name}"
            )
            if res["rejected"]:
                msg += "\n\nAlgunas muestras fallaron la validación de calidad inicial."
            QMessageBox.information(self, "Preparación Completa", msg)
            self.status.setText(f"Voicebank preparado. {res['accepted']} muestras.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al preparar voicebank: {e}")

    def _validate_voicebank(self) -> None:
        """Run full validation on the prepared voicebank."""
        try:
            report = self.manager.validate_voicebank()
            status_str = "VÁLIDO" if report["voicebank_available"] else "INCOMPLETO / INVÁLIDO"

            summary = (
                f"Voicebank: LYKENOX Spanish Lite\n"
                f"Estado: {status_str}\n"
                f"Cobertura: {report['voicebank_coverage']}%\n\n"
                f"Muestras encontradas: {report['available_count']} / {report['reclist_count']}\n"
                f"Entradas OTO: {report['oto_entries']}\n"
                f"Muestras inválidas: {len(report['invalid_wav'])}\n"
            )

            if report["missing_aliases"]:
                summary += f"\nAliases faltantes: {', '.join(report['missing_aliases'][:10])}"
                if len(report["missing_aliases"]) > 10:
                    summary += " ..."

            if not report["voicebank_available"]:
                summary += "\n\n⚠ Revise las grabaciones marcadas como inválidas o faltantes."
            else:
                summary += "\n\n✓ ¡Listo para cantar!"

            QMessageBox.information(self, "Validación de Voicebank", summary)
            self.status.setText(f"Validación terminada: {status_str} ({report['voicebank_coverage']}%)")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al validar voicebank: {e}")

    def _show_test_results(self) -> None:
        if not self.test_log:
            return

        report = "RESULTADOS DEL TEST DE MICRÓFONO\n"
        report += "Segundos | RMS | Peak\n"
        report += "---------------------\n"

        drops = 0
        for t, r, p in self.test_log:
            flag = ""
            if r < 100:  # Noise floor threshold
                flag = " [!] BAJO"
                drops += 1
            report += f"{t:8.1f} | {r:5d} | {p:5d}{flag}\n"

        if drops > 5:
            report += "\nDIAGNÓSTICO: Se detecta caída de señal significativa.\n"
            report += "Es probable que Windows o el driver Intel estén suprimiendo su voz.\n"
            report += "Intente desactivar 'Audio Enhancements' en el Panel de Control."
        else:
            report += "\nDIAGNÓSTICO: Captura estable."

        QMessageBox.information(self, "Resultado del Test", report)
        logger.info(report)

    def _update_ui_state(self) -> None:
        has_alias = self.current_index < len(self.reclist.aliases)

        if has_alias:
            alias = self.reclist.aliases[self.current_index]
            self.current_prompt.setText(f"Fonema actual: {alias}")
        else:
            self.current_prompt.setText("¡Reclist completado!")
        layer = self._current_layer()
        note, hz = self._target_for_layer(layer)
        self.target_note_label.setText(f"Guía registro: {note}")
        self.target_freq_label.setText(f"Referencia: {hz:.2f} Hz")
        self.layer_instruction.setText(self._instruction_for_layer(layer))

        has_wav = self.temp_wav is not None and self.temp_wav.exists()

        recorded = self.manager.recorded_aliases_for_layer(layer)
        total = len(self.reclist.aliases)
        is_complete = len(recorded) >= total

        if is_complete:
            self.progress_label.setText(f"{layer}: {len(recorded)} / {total}")
            self.progress_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.progress_label.setText(f"{layer}: {len(recorded)} / {total}")
            self.progress_label.setStyleSheet("")

        busy = self.recording or self.stopping
        self.btn_record.setEnabled(not busy and not is_complete)
        self.btn_stop.setEnabled(self.recording and not self.stopping)
        self.btn_test_mic.setEnabled(not busy)
        self.btn_calibrate.setEnabled(not busy)
        self.btn_listen.setEnabled((not busy) and has_wav)
        self.btn_save.setEnabled((not busy) and has_wav)
        self.btn_delete.setEnabled((not busy) and has_wav)
        self.btn_next.setEnabled(not busy and not is_complete)
        self.btn_prepare.setVisible(is_complete)
        self.btn_validate.setVisible(is_complete)
        self.btn_prepare.setEnabled(not busy)
        self.btn_validate.setEnabled(not busy)
        self.mic_selector.setEnabled(not busy)
        self.layer_selector.setEnabled(not busy)
        self.btn_guide_tone.setEnabled(not busy and has_alias)
        self.aliases_list.setEnabled(not busy)

    # ── Recording: start ────────────────────────────────────────

    def _start_recording(self) -> None:
        if self.recording or self.stopping:
            return

        if not self.is_testing and self.current_index >= len(self.reclist.aliases):
            self.status.setText("No hay más fonemas para grabar.")
            return

        # 1. Kill any previous stream (guarantee single stream)
        self._cleanup_audio_source()

        # 2. Reset session completely — nothing reutilized
        self.current_session_id += 1
        self.audio_buffer = bytearray()
        self.record_time_ms = 0
        self.frames_received = 0
        self._prev_frames = 0
        self._stall_ticks = 0
        self._low_rms_ticks = 0

        # 3. Remove stale temp file
        if self.temp_wav and self.temp_wav.exists():
            try:
                self.temp_wav.unlink()
            except OSError:
                pass
            self.temp_wav = None

        # 4. Open new audio session
        try:
            device = self.mic_selector.currentData()
            if not device or device.isNull():
                raise RuntimeError("No se detecta micrófono.")

            self.audio_format = device.preferredFormat()
            self.audio_format.setChannelCount(1)
            self.audio_format.setSampleFormat(QAudioFormat.Int16)

            # Prefer 48 kHz if hardware supports it natively
            test_48k = QAudioFormat()
            test_48k.setSampleRate(48000)
            test_48k.setChannelCount(1)
            test_48k.setSampleFormat(QAudioFormat.Int16)
            if device.isFormatSupported(test_48k):
                self.audio_format = test_48k

            self.native_sample_rate = self.audio_format.sampleRate()

            self.audio_source = QAudioSource(device, self.audio_format, self)
            self.audio_source.stateChanged.connect(self._on_state_changed)

            # 10-second ring buffer to prevent overflow cuts
            buf_size = self.native_sample_rate * 2 * 10
            self.audio_source.setBufferSize(buf_size)

            self.io_device = self.audio_source.start()
            if not self.io_device:
                raise RuntimeError("El hardware de audio no respondió.")

            self.recording = True
            self.timer.start(50)

            # UI update
            if self.is_testing:
                self.rec_indicator.setText("● TEST MIC (10s) | Diga 'aaaaaaaa'...")
                self.rec_indicator.setStyleSheet(
                    "font-size: 18px; color: #0078d4; font-weight: bold;"
                )
            else:
                alias = self.reclist.aliases[self.current_index]
                layer = self._current_layer()
                self.rec_indicator.setText(
                    f"● GRABANDO | Capa: {layer} | Fonema: {alias} | Tiempo: 00:00 | Frames: 0"
                )
                self.rec_indicator.setStyleSheet(
                    "font-size: 18px; color: red; font-weight: bold;"
                )
            self.timer_label.setText("00:00")
            self.level_meter.setText("Nivel: ----------")

            msg = (
                f"Sesión {self.current_session_id} | "
                f"{device.description()} | "
                f"{self.native_sample_rate}Hz mono"
            )
            if self.native_sample_rate != 48000:
                msg += f" → se convertirá a 48kHz"
            self.status.setText(msg)
            self._update_ui_state()

            logger.info(
                f"Session {self.current_session_id} started. "
                f"Rate: {self.native_sample_rate}Hz | Initial frames: 0"
            )

        except Exception as e:
            self.status.setText(f"Error al iniciar grabación: {e}")
            self._cleanup_audio_source()
            self.recording = False
            self._update_ui_state()

    # ── Recording: hardware callbacks ───────────────────────────

    def _on_state_changed(self, state: int) -> None:
        """Handle hardware interruptions."""
        if self.stopping or not self.recording:
            return
        if state == QAudio.StoppedState:
            err = self.audio_source.error() if self.audio_source else QAudio.NoError
            if err != QAudio.NoError:
                logger.error(f"Hardware error: {err}")
                self._stop_recording(error=True)

    def _cleanup_audio_source(self) -> None:
        """Full teardown of QAudioSource."""
        if self.audio_source:
            try:
                self.audio_source.stop()
            except Exception:
                pass
            self.audio_source.deleteLater()
            self.audio_source = None
        self.io_device = None

    def _read_audio_data(self) -> None:
        """Pull all pending bytes from the io device (non-blocking)."""
        if self.io_device and self.io_device.isOpen():
            data = self.io_device.readAll()
            if data.size() > 0:
                chunk = data.data()
                self.audio_buffer.extend(chunk)
                self.frames_received += len(chunk) // 2

    # ── Recording: timer tick (50 ms) ──────────────────────────

    def _on_timer_tick(self) -> None:
        if not self.recording or self.stopping:
            return

        self._read_audio_data()

        self.record_time_ms += 50

        if self.is_testing and self.record_time_ms >= 10000:
            self._stop_recording()
            return

        sec = (self.record_time_ms // 1000) % 60
        mn = self.record_time_ms // 60000
        time_str = f"{mn:02d}:{sec:02d}"

        # Update indicator with live data
        alias = self.reclist.aliases[self.current_index]
        layer = self._current_layer()
        self.rec_indicator.setText(
            f"● GRABANDO | Capa: {layer} | Fonema: {alias} | Tiempo: {time_str}"
            f" | Frames: {self.frames_received}"
        )
        self.timer_label.setText(time_str)

        # Level meter
        level = 0
        p = 0
        if len(self.audio_buffer) >= 2000:
            last_chunk = self.audio_buffer[-2000:]
            level = rms(last_chunk, 2)
            p = peak(last_chunk, 2)
            v = min(15, level // 400)
            meter = "█" * v + "-" * (15 - v)
            self.level_meter.setText(f"Nivel: {meter}")
            self.level_meter.setStyleSheet(
                "color: #00ff00;" if level > 150 else "color: gray;"
            )
            live_f0 = self._estimate_live_f0(last_chunk)
            self.live_f0_label.setText(
                f"F0 actual: {live_f0:.0f} Hz" if live_f0 > 0 else "F0 actual: -- Hz"
            )

        # Test logging
        if self.is_testing and self.record_time_ms % 500 < 50:
            self.test_log.append((self.record_time_ms / 1000.0, level, p))

        # Stall detection — no new frames for ~2 s
        if self.frames_received == self._prev_frames:
            self._stall_ticks += 1
            if self._stall_ticks == 40:
                self.status.setText(
                    f"⚠ Sin datos del micrófono desde hace ~2s. "
                    f"Frames: {self.frames_received}"
                )
        else:
            self._stall_ticks = 0
        self._prev_frames = self.frames_received

        # Silence detection — low RMS for ~3 s
        if level < 50 and self.frames_received > 0:
            self._low_rms_ticks += 1
            if self._low_rms_ticks == 60:
                self.status.setText(
                    "⚠ Nivel muy bajo desde hace ~3s. ¿Micrófono activo?"
                )
        else:
            self._low_rms_ticks = 0

        # Per-second diagnostic log
        if self.record_time_ms % 1000 < 50:
            st = self.audio_source.state() if self.audio_source else "?"
            logger.info(
                f"REC | Session:{self.current_session_id} | "
                f"T:{self.record_time_ms / 1000:.1f}s | "
                f"Frames:{self.frames_received} | "
                f"RMS:{level} | Peak:{p} | QtState:{st}"
            )

    # ── Recording: stop ─────────────────────────────────────────

    def _stop_recording(self, error: bool = False) -> None:
        if not self.recording and not self.stopping:
            return

        # 1. Enter stopping state
        was_recording = self.recording
        self.stopping = True
        self.recording = False

        # 2. Stop timer (no more periodic reads)
        self.timer.stop()

        # 3. Drain pending Qt events (flush any buffered callbacks)
        QApplication.processEvents()

        # 4. Final data pull
        self._read_audio_data()

        # 5. Stop hardware
        if self.audio_source:
            self.audio_source.stop()

        # 6. Process events again (handle StoppedState signal)
        QApplication.processEvents()

        # 7. Full cleanup
        self._cleanup_audio_source()

        if self.is_testing:
            self._show_test_results()
            self.is_testing = False

        # 8. Write WAV to temp file
        success = False
        if not error and was_recording and len(self.audio_buffer) > 0:
            alias = self.reclist.aliases[self.current_index]
            self.temp_wav = (
                self.manager.raw_dir
                / f"temp_{alias}_{self.current_session_id}.wav"
            )
            self.manager.raw_dir.mkdir(parents=True, exist_ok=True)
            success = self._write_wav(self.temp_wav)

        # 9. Convert to 48 kHz if mic recorded at native rate
        if (
            success
            and self.native_sample_rate != 48000
            and self.temp_wav is not None
        ):
            alias = self.reclist.aliases[self.current_index]
            converted = (
                self.manager.raw_dir
                / f"temp_{alias}_{self.current_session_id}_48k.wav"
            )
            if self._convert_to_48k(self.temp_wav, converted):
                try:
                    self.temp_wav.unlink()
                except OSError:
                    pass
                self.temp_wav = converted
                logger.info(
                    f"Converted {self.native_sample_rate}Hz → 48000Hz"
                )

        # 10. Validate WAV (RIFF + wave.open + FFmpeg)
        valid = False
        if success and self.temp_wav is not None and self.temp_wav.exists():
            valid = self._validate_wav(self.temp_wav)

        # 11. Update UI — only now
        self.stopping = False
        self._prev_frames = 0
        self._stall_ticks = 0
        self._low_rms_ticks = 0

        if error:
            self.rec_indicator.setText("● ERROR DE GRABACIÓN")
            self.rec_indicator.setStyleSheet(
                "font-size: 18px; color: red; font-weight: bold;"
            )
            self.temp_wav = None
            self.timer_label.setText("00:00")
            self.level_meter.setText("Nivel: ----------")
            self.live_f0_label.setText("F0 actual: -- Hz")
            self.status.setText("Error en la grabación. Inténtalo de nuevo.")
        elif not success or not valid:
            self.rec_indicator.setText("○ STANDBY")
            self.rec_indicator.setStyleSheet(
                "font-size: 18px; color: gray;"
            )
            self.temp_wav = None
            self.timer_label.setText("00:00")
            self.level_meter.setText("Nivel: ----------")
            self.live_f0_label.setText("F0 actual: -- Hz")
            self.status.setText("Grabación inválida o vacía.")
        else:
            dur_wav = self._get_wav_duration(self.temp_wav)
            dur_ui = self.record_time_ms / 1000.0
            self.rec_indicator.setText("● GRABACIÓN TERMINADA")
            self.rec_indicator.setStyleSheet(
                "font-size: 18px; color: green; font-weight: bold;"
            )
            self.timer_label.setText(
                f"Tiempo UI: {dur_ui:.1f}s | WAV: {dur_wav:.1f}s"
            )
            self.status.setText(
                f"Grabación terminada. Duración real: {dur_wav:.2f} s"
            )
            logger.info(
                f"Session {self.current_session_id} stopped. "
                f"UI:{dur_ui:.2f}s | WAV:{dur_wav:.2f}s | "
                f"Frames:{self.frames_received}"
            )

        self._update_ui_state()

    # ── WAV writing / validation / conversion ───────────────────

    def _write_wav(self, path: Path) -> bool:
        if not self.audio_buffer:
            return False
        try:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.native_sample_rate)
                w.writeframes(bytes(self.audio_buffer))
            logger.info(
                f"WAV written: {path.name} | "
                f"{self.native_sample_rate}Hz | "
                f"{self.frames_received} frames | "
                f"{path.stat().st_size} bytes"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to write WAV: {e}")
            return False

    def _validate_wav(self, path: Path) -> bool:
        """Validate RIFF header + wave.open + FFmpeg."""
        if path is None or not path.exists():
            return False

        # 1. RIFF header check
        try:
            with open(path, "rb") as f:
                header = f.read(12)
                if len(header) < 12:
                    return False
                if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                    logger.error(f"Bad RIFF header: {header[:12]}")
                    return False
        except Exception:
            return False

        # 2. wave.open structural check
        try:
            with wave.open(str(path), "rb") as r:
                n_frames = r.getnframes()
                rate = r.getframerate()
                channels = r.getnchannels()
                width = r.getsampwidth()
                dur = n_frames / rate if rate else 0.0
                if n_frames == 0:
                    return False
                if channels not in (1, 2):
                    return False
                if width != 2:
                    return False
                logger.info(
                    f"wave.open OK: {n_frames}f {rate}Hz "
                    f"{channels}ch {width * 8}bit {dur:.2f}s"
                )
        except Exception as e:
            logger.error(f"wave.open FAILED: {e}")
            return False

        # 3. FFmpeg technical validation
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                capture_output=True,
                check=True,
                timeout=15,
            )
            logger.info("FFmpeg validation: OK")
            return True
        except FileNotFoundError:
            logger.warning("FFmpeg not found — skipping FFmpeg check")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg FAILED: {e.stderr.decode(errors='replace')}")
            return False
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
            return False

    def _convert_to_48k(self, input_path: Path, output_path: Path) -> bool:
        """Convert WAV to 48 kHz mono 16-bit PCM via FFmpeg."""
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-ar", "48000", "-ac", "1",
                    "-sample_fmt", "s16", str(output_path),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
            return output_path.exists() and output_path.stat().st_size > 44
        except Exception as e:
            logger.error(f"FFmpeg conversion failed: {e}")
            return False

    def _get_wav_duration(self, path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as r:
                return r.getnframes() / float(r.getframerate())
        except Exception:
            return 0.0

    # ── Button: Escuchar ────────────────────────────────────────

    def _listen_current(self) -> None:
        if not self.temp_wav or not self.temp_wav.exists():
            self.status.setText("Nada que escuchar.")
            return

        if QMediaPlayer is None:
            self.status.setText("Error: QMediaPlayer no disponible.")
            return

        if not hasattr(self, "_player"):
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)
            self._player.playbackStateChanged.connect(
                self._on_playback_state_changed
            )

        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(self.temp_wav)))
        self._audio_out.setVolume(1.0)

        dur = self._get_wav_duration(self.temp_wav)
        self._player.play()
        self.status.setText(f"Reproduciendo {self.temp_wav.name} ({dur:.1f}s)")

    def _on_playback_state_changed(self, state: int) -> None:
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.status.setText("Reproducción finalizada.")

    # ── Button: Guardar ─────────────────────────────────────────

    def _save_current(self) -> None:
        if not self.temp_wav or not self.temp_wav.exists():
            self.status.setText("No hay grabación para guardar.")
            return

        if not self._validate_wav(self.temp_wav):
            self.status.setText("Error: La grabación no es válida.")
            return

        alias = self.reclist.aliases[self.current_index]
        layer = self._current_layer()
        final_path = self.manager.raw_path_for_layer(alias, layer)

        if final_path.exists():
            ans = QMessageBox.question(
                self, "Sobrescribir",
                f"Ya existe '{alias}.wav' en {layer}. ¿Sobrescribir?",
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                final_path.unlink()
            shutil.copy2(self.temp_wav, final_path)
            self.status.setText(f"Guardado: {layer} / {alias}")
            self._refresh_alias_list()
            self._next_phoneme(force=True)
        except Exception as e:
            self.status.setText(f"Error al guardar: {e}")

    # ── Button: Borrar ──────────────────────────────────────────

    def _delete_current(self) -> None:
        if self.temp_wav and self.temp_wav.exists():
            try:
                self.temp_wav.unlink()
            except OSError:
                pass
            self.temp_wav = None

        self.timer_label.setText("00:00")
        self.level_meter.setText("Nivel: ----------")
        self.live_f0_label.setText("F0 actual: -- Hz")
        self.rec_indicator.setText("○ STANDBY")
        self.rec_indicator.setStyleSheet("font-size: 18px; color: gray;")
        self.status.setText("Grabación borrada.")
        self._update_ui_state()

    # ── Button: Siguiente ───────────────────────────────────────

    def _next_phoneme(self, force: bool = False) -> None:
        if not force and self.temp_wav and self.temp_wav.exists():
            ans = QMessageBox.question(
                self, "Siguiente",
                "Hay una grabación sin guardar.\n"
                "¿Quieres pasar al siguiente fonema?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        # Discard temp
        if self.temp_wav and self.temp_wav.exists():
            try:
                self.temp_wav.unlink()
            except OSError:
                pass
            self.temp_wav = None

        self.current_index += 1
        if self.current_index >= len(self.reclist.aliases):
            self.current_index = len(self.reclist.aliases) - 1
            self.status.setText("¡Has llegado al final de la lista!")

        self.timer_label.setText("00:00")
        self.level_meter.setText("Nivel: ----------")
        self.live_f0_label.setText("F0 actual: -- Hz")
        self.rec_indicator.setText("○ STANDBY")
        self.rec_indicator.setStyleSheet("font-size: 18px; color: gray;")
        self.aliases_list.setCurrentRow(self.current_index)
        self._update_ui_state()

    # ── Alias list ──────────────────────────────────────────────

    def _refresh_alias_list(self) -> None:
        self.aliases_list.clear()
        recorded = self.manager.recorded_aliases_for_layer(self._current_layer())
        for alias in self.reclist.aliases:
            marker = "✔" if alias.lower() in recorded else "·"
            self.aliases_list.addItem(f"{marker}  {alias}")
        if self.current_index < len(self.reclist.aliases):
            self.aliases_list.setCurrentRow(self.current_index)

    def _current_layer(self) -> str:
        return self.layer_selector.currentText() or "Low"

    def _target_for_layer(self, layer: str) -> tuple[str, float]:
        plan_path = self.manager.voicebank_dir / "multipitch_microtest_plan.json"
        # Low: ~123Hz (B2), Mid: ~145Hz (D3), High: Por calibrar
        fallback = {"Low": ("B2", 123.0), "Mid": ("D3", 145.0), "High": ("Por calibrar", 0.0)}
        if not plan_path.exists():
            return fallback.get(layer, fallback["Low"])
        try:
            data = json.loads(plan_path.read_text(encoding="utf-8"))
            for center in data.get("pitch_report", {}).get("centers", []):
                if center.get("layer") == layer:
                    hz = float(center["hz"])
                    note = str(center.get("note", midi_to_note(int(round(hz_to_midi(hz))))))
                    return note, hz
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return fallback.get(layer, fallback["Low"])

    def _on_calibrate_clicked(self) -> None:
        """Open the calibration dialog to measure vocal range."""
        dialog = CalibrationDialog(self.root, self)
        if dialog.exec():
            results = dialog.results
            if results:
                self._apply_calibration(results)

    def _apply_calibration(self, results: dict[str, float]) -> None:
        """Save calibrated frequencies to the multipitch plan with coverage logic."""
        try:
            plan_path = self.manager.voicebank_dir / "multipitch_microtest_plan.json"
            if plan_path.exists():
                data = json.loads(plan_path.read_text(encoding="utf-8"))
            else:
                data = {
                    "microtest_aliases": ["a", "bai", "la", "con", "mi", "go"],
                    "pitch_report": {}
                }

            low_hz = results.get("Low", 123.0)
            mid_hz = results.get("Mid", 145.0)
            high_hz = results.get("High", 185.0)

            low_midi = int(round(hz_to_midi(low_hz)))
            mid_midi = int(round(hz_to_midi(mid_hz)))
            high_midi = int(round(hz_to_midi(high_hz)))

            # Define ranges to cover the full MIDI spectrum (0-127)
            # Thresholds are midpoints between centers
            threshold_low_mid = (low_midi + mid_midi) // 2
            threshold_mid_high = (mid_midi + high_midi) // 2

            centers = [
                {
                    "layer": "Low",
                    "midi": low_midi,
                    "note": midi_to_note(low_midi),
                    "hz": round(low_hz, 2),
                    "range_min_midi": -99,
                    "range_max_midi": threshold_low_mid
                },
                {
                    "layer": "Mid",
                    "midi": mid_midi,
                    "note": midi_to_note(mid_midi),
                    "hz": round(mid_hz, 2),
                    "range_min_midi": threshold_low_mid + 1,
                    "range_max_midi": threshold_mid_high
                },
                {
                    "layer": "High",
                    "midi": high_midi,
                    "note": midi_to_note(high_midi),
                    "hz": round(high_hz, 2),
                    "range_min_midi": threshold_mid_high + 1,
                    "range_max_midi": 127
                }
            ]

            data["pitch_report"]["centers"] = centers
            plan_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            QMessageBox.information(self, "Calibración", "Calibración guardada. Los centros de pitch han sido actualizados.")
            self._update_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar la calibración: {e}")

    def _play_guide_tone(self) -> None:
        if QMediaPlayer is None:
            self.status.setText("Error: QMediaPlayer no disponible.")
            return
        note, hz = self._target_for_layer(self._current_layer())
        guide_path = self.manager.raw_dir / "guide_tone.wav"
        self.manager.raw_dir.mkdir(parents=True, exist_ok=True)
        self._write_guide_tone(guide_path, hz)
        if not hasattr(self, "_player"):
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)
            self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(guide_path)))
        self._audio_out.setVolume(0.6)
        self._player.play()
        self.status.setText(
            f"Tono guía {note} ({hz:.2f} Hz). Es referencia, no requisito; "
            "espere a que termine antes de grabar."
        )

    def _write_guide_tone(self, path: Path, hz: float) -> None:
        sample_rate = 48000
        duration_sec = 1.2
        total = int(sample_rate * duration_sec)
        frames = bytearray()
        for index in range(total):
            t = index / sample_rate
            fade = min(1.0, index / 2400, (total - index) / 2400)
            value = int(0.22 * fade * 32767 * math.sin(2.0 * math.pi * hz * t))
            frames.extend(struct.pack("<h", value))
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(bytes(frames))

    def _instruction_for_layer(self, layer: str) -> str:
        instructions = {
            "Low": "Graba este alias en tu registro grave cómodo. No persigas una nota exacta.",
            "Mid": "Graba este alias en tu registro medio cómodo. Mantén tu timbre natural.",
            "High": "Graba este alias en tu registro agudo cómodo. No imites otra voz ni fuerces afinación.",
        }
        return instructions.get(layer, "Graba con emisión natural.")

    def _estimate_live_f0(self, chunk: bytes) -> float:
        if self.native_sample_rate <= 0 or len(chunk) < 800:
            return 0.0
        samples = [
            struct.unpack_from("<h", chunk, index)[0]
            for index in range(0, len(chunk) - 1, 2)
        ]
        if not samples:
            return 0.0
        if max(abs(value) for value in samples) < 500:
            return 0.0
        crossings = 0
        prev = samples[0]
        for sample in samples[1:]:
            if (prev <= 0 < sample) or (prev >= 0 > sample):
                crossings += 1
            prev = sample
        duration = len(samples) / float(self.native_sample_rate)
        estimated = crossings / (2.0 * duration) if duration > 0 else 0.0
        return estimated if 70.0 <= estimated <= 500.0 else 0.0

    # ── Close ───────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self.recording or self.stopping:
            self._stop_recording()
        self._cleanup_audio_source()
        event.accept()


class CalibrationDialog(QDialog):
    """Advanced dialog to record and measure vocal range centers with technical feedback."""

    def __init__(self, root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.root = root
        self.setWindowTitle("Calibración Avanzada de Rango Vocal")
        self.setMinimumWidth(600)

        self.results = {
            "Low": {"hz": 0.0, "note": "--", "midi": 0, "frames": 0, "confidence": 0.0, "status": "Pendiente"},
            "Mid": {"hz": 0.0, "note": "--", "midi": 0, "frames": 0, "confidence": 0.0, "status": "Pendiente"},
            "High": {"hz": 0.0, "note": "--", "midi": 0, "frames": 0, "confidence": 0.0, "status": "Pendiente"},
        }
        self.current_layer = "Low"
        self.recording = False
        self.audio_source = None
        self.io_device = None
        self.audio_buffer = bytearray()
        self.last_audio_buffer = bytearray() # To play back the last take
        self.record_time_ms = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        if QMediaDevices is None:
            QMessageBox.critical(self, "Error", "QtMultimedia no está disponible en este sistema.")
            self.reject()
            return

        # Check engine for F0 measurement
        engine = OpenUtauWorldlineEngine(self.root)
        if not engine.health_check()["available"]:
            QMessageBox.warning(
                self, "Advertencia",
                "WORLDLINE-R no disponible. La medición de F0 fallará.\n"
                "Verifique que tools/renderers/worldline_r/worldline.dll exista."
            )

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        self.info = QLabel(
            "Cante una nota cómoda y sostenida para cada capa.\n"
            "El sistema medirá el tono exacto para configurar su voicebank."
        )
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        # Result Table
        self.table = QTableWidget(3, 6)
        self.table.setHorizontalHeaderLabels(["Capa", "Hz", "Nota", "MIDI", "Confianza", "Estado"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)

        for i, layer in enumerate(["Low", "Mid", "High"]):
            self.table.setItem(i, 0, QTableWidgetItem(layer))
            for j in range(1, 6):
                self.table.setItem(i, j, QTableWidgetItem("--"))

        layout.addWidget(self.table)

        # Technical Details Area
        self.details = QLabel("Detalles técnicos de la última toma: ---")
        self.details.setStyleSheet("font-family: monospace; font-size: 11px; color: #444; border: 1px solid #ccc; padding: 5px;")
        layout.addWidget(self.details)

        # Recording area
        rec_layout = QVBoxLayout()
        self.layer_info = QLabel("Paso 1: Grabe una nota GRAVE cómoda.")
        self.layer_info.setStyleSheet("font-weight: bold; font-size: 14px; color: #0078d4;")
        rec_layout.addWidget(self.layer_info, alignment=Qt.AlignCenter)

        self.timer_label = QLabel("00:00")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-family: monospace; font-size: 24px; color: #d83b01;")
        rec_layout.addWidget(self.timer_label)

        self.level_meter = QLabel("Nivel: ----------")
        self.level_meter.setAlignment(Qt.AlignCenter)
        rec_layout.addWidget(self.level_meter)

        layout.addLayout(rec_layout)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_rec = QPushButton("Grabar toma")
        self.btn_rec.setStyleSheet("background-color: #d83b01; color: white; min-height: 40px; font-weight: bold;")
        self.btn_rec.clicked.connect(self._toggle_recording)

        self.btn_listen = QPushButton("Escuchar toma")
        self.btn_listen.setEnabled(False)
        self.btn_listen.clicked.connect(self._play_last_take)

        self.btn_retry = QPushButton("Repetir toma")
        self.btn_retry.setEnabled(False)
        self.btn_retry.clicked.connect(self._retry_take)

        btn_row.addWidget(self.btn_rec)
        btn_row.addWidget(self.btn_listen)
        btn_row.addWidget(self.btn_retry)
        layout.addLayout(btn_row)

        self.status = QLabel("Listo para empezar.")
        layout.addWidget(self.status)

        # Dialog control
        self.btn_done = QPushButton("Aceptar calibración")
        self.btn_done.setEnabled(False)
        self.btn_done.clicked.connect(self.accept)
        layout.addWidget(self.btn_done)

        self.setLayout(layout)
        self._update_table()

    def _toggle_recording(self) -> None:
        if not self.recording:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        try:
            device = QMediaDevices.defaultAudioInput()
            if device.isNull():
                raise RuntimeError("No se detecta micrófono predeterminado.")

            fmt = device.preferredFormat()
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.Int16)

            self.audio_source = QAudioSource(device, fmt, self)
            self.audio_buffer = bytearray()
            self.record_time_ms = 0
            self.io_device = self.audio_source.start()
            if not self.io_device:
                raise RuntimeError("Error al abrir dispositivo de audio.")

            self.recording = True
            self.btn_rec.setText("Detener")
            self.btn_rec.setStyleSheet("background-color: #a80000; color: white; min-height: 40px;")
            self.btn_listen.setEnabled(False)
            self.btn_retry.setEnabled(False)
            self.timer.start(50)
            self.status.setText(f"Grabando nota {self.current_layer}...")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _stop(self) -> None:
        self.timer.stop()
        if self.audio_source:
            self.last_rate = self.audio_source.format().sampleRate()
            self.audio_source.stop()
            data = self.io_device.readAll()
            self.audio_buffer.extend(data.data())
            self.audio_source.deleteLater()
            self.audio_source = None

        self.recording = False
        self.btn_rec.setText("Grabar toma")
        self.btn_rec.setStyleSheet("background-color: #d83b01; color: white; min-height: 40px;")

        self.last_audio_buffer = bytearray(self.audio_buffer)
        self.btn_listen.setEnabled(True)
        self.btn_retry.setEnabled(True)

        self._process_recording()

    def _on_timer_tick(self) -> None:
        if not self.recording:
            return
        data = self.io_device.readAll()
        self.audio_buffer.extend(data.data())
        self.record_time_ms += 50
        sec = (self.record_time_ms // 1000) % 60
        ms = (self.record_time_ms % 1000) // 10
        self.timer_label.setText(f"{sec:02d}:{ms:02d}")

        if len(self.audio_buffer) >= 2000:
            v = min(15, rms(self.audio_buffer[-2000:], 2) // 400)
            self.level_meter.setText(f"Nivel: {'█' * v}{'-' * (15 - v)}")

    def _process_recording(self) -> None:
        if not self.audio_buffer:
            self.status.setText("Grabación vacía.")
            return

        try:
            samples = []
            for i in range(0, len(self.audio_buffer), 2):
                if i + 1 < len(self.audio_buffer):
                    val = struct.unpack("<h", self.audio_buffer[i:i+2])[0]
                    samples.append(val / 32768.0)

            engine = OpenUtauWorldlineEngine(self.root)
            stats = engine.analyze_f0(samples)

            hz = stats["mean_f0_hz"]
            confidence = stats["confidence"]

            status_text = "OK"
            if hz <= 0:
                status_text = "ERROR (Sin F0)"
            elif confidence < 30:
                status_text = "REPETIR (Inestable)"

            midi = int(round(hz_to_midi(hz))) if hz > 0 else 0
            note = midi_to_note(midi) if hz > 0 else "--"

            self.results[self.current_layer] = {
                "hz": hz,
                "note": note,
                "midi": midi,
                "frames": stats["voiced_frames"],
                "confidence": confidence,
                "status": status_text
            }

            self.details.setText(
                f"Toma {self.current_layer}: {hz:.2f} Hz | "
                f"Confianza: {confidence}% | "
                f"Frames voz: {stats['voiced_frames']} / {stats['total_frames']} | "
                f"Duración: {stats['duration_sec']} s"
            )

            self._update_table()
            self._check_and_advance()

        except Exception as e:
            QMessageBox.critical(self, "Error al procesar", str(e))

    def _update_table(self) -> None:
        for i, layer in enumerate(["Low", "Mid", "High"]):
            res = self.results[layer]
            self.table.setItem(i, 1, QTableWidgetItem(f"{res['hz']:.2f}" if res['hz'] > 0 else "--"))
            self.table.setItem(i, 2, QTableWidgetItem(res['note']))
            self.table.setItem(i, 3, QTableWidgetItem(str(res['midi']) if res['midi'] > 0 else "--"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{res['confidence']}%" if res['hz'] > 0 else "--"))

            status_item = QTableWidgetItem(res['status'])
            if res['status'] == "OK":
                status_item.setForeground(Qt.green)
            elif "ERROR" in res['status'] or "REPETIR" in res['status']:
                status_item.setForeground(Qt.red)
            self.table.setItem(i, 5, status_item)

        self._validate_range()

    def _validate_range(self) -> None:
        low = self.results["Low"]["hz"]
        mid = self.results["Mid"]["hz"]
        high = self.results["High"]["hz"]

        valid_order = True

        # Reset colors
        for i in range(3):
            for j in range(6):
                self.table.item(i, j).setBackground(Qt.transparent)

        if low > 0 and mid > 0 and mid <= low:
            self.table.item(1, 1).setBackground(Qt.red)
            valid_order = False
        if mid > 0 and high > 0 and high <= mid:
            self.table.item(2, 1).setBackground(Qt.red)
            valid_order = False

        all_ok = all(self.results[l]["status"] == "OK" for l in ["Low", "Mid", "High"])
        self.btn_done.setEnabled(all_ok and valid_order)
        if all_ok and not valid_order:
            self.status.setText("⚠ Orden de pitch inválido (Grave < Media < Alta).")
        elif all_ok:
            self.status.setText("✓ Calibración válida. Puede aceptar.")

    def _check_and_advance(self) -> None:
        if self.results[self.current_layer]["status"] == "OK":
            if self.current_layer == "Low":
                self.current_layer = "Mid"
                self.layer_info.setText("Paso 2: Grabe una nota MEDIA cómoda.")
            elif self.current_layer == "Mid":
                self.current_layer = "High"
                self.layer_info.setText("Paso 3: Grabe una nota ALTA cómoda.")
            else:
                self.layer_info.setText("Calibración terminada. Revise la tabla.")
                self.btn_rec.setEnabled(False)

    def _retry_take(self) -> None:
        """Reset current layer status and allow recording again."""
        self.results[self.current_layer] = {"hz": 0.0, "note": "--", "midi": 0, "frames": 0, "confidence": 0.0, "status": "Pendiente"}
        self.btn_rec.setEnabled(True)
        self.status.setText(f"Repitiendo toma {self.current_layer}...")
        self._update_table()

    def _play_last_take(self) -> None:
        if not self.last_audio_buffer:
            return

        # Simple playback of memory buffer
        temp_path = self.root / "logs" / "last_calibration_take.wav"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        # We need the sample rate used during recording
        rate = 44100
        if self.audio_source:
            rate = self.audio_source.format().sampleRate()
        elif hasattr(self, "last_rate"):
            rate = self.last_rate

        with wave.open(str(temp_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(self.last_audio_buffer))

        if not hasattr(self, "_player"):
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)

        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(temp_path)))
        self._audio_out.setVolume(1.0)
        self._player.play()

    def accept(self) -> None:
        # Convert results to the format expected by VoicebankPage._apply_calibration
        final_results = {k: v["hz"] for k, v in self.results.items()}
        self.results = final_results
        super().accept()
