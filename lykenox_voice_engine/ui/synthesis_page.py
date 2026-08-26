"""Singing synthesis request page."""

from __future__ import annotations

import json
from pathlib import Path

import os
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QUrl
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except ImportError:
    QAudioOutput = None
    QMediaPlayer = None

from lykenox_voice_engine.engines.utau_engine import UtauSampleEngine
from lykenox_voice_engine.core.score import load_score, score_to_json
from lykenox_voice_engine.models.notes import NoteEvent


class SynthesisPage(QWidget):
    """Collect lyrics and note data for direct sample-based singing synthesis."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.engine = UtauSampleEngine(root)
        self.output_wav = None

        # Player
        if QMediaPlayer:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
        else:
            self.player = None

        self.status = QLabel("Salida: pendiente | Backend: UTAU sample-based")
        self.worldline_health = self.engine.worldline_health()
        self.scores_dir = self.root / "scores"

        self.renderer_sel = QComboBox()
        self.renderer_sel.addItem("Motor Interno (Básico)", "internal")
        self.renderer_sel.addItem("LYKENOX UTAU Bridge", "worldline")
        self.renderer_sel.addItem("WORLDLINE-R real", "worldline_real")
        if not self.worldline_health["available"]:
            self.renderer_sel.model().item(2).setEnabled(False)

        self.btn_compile = QPushButton("Compilar LYKENOX UTAU Bridge")
        self.btn_compile.clicked.connect(self._compile_renderer)

        profile = QComboBox()
        profile.addItem("lykenox")
        voicebank = QComboBox()
        voicebank.addItem("LYKENOX Spanish Lite")
        lyrics = QPlainTextEdit()
        tempo = QSpinBox()
        tempo.setRange(40, 240)
        tempo.setValue(120)
        melody = QPlainTextEdit()
        score_list = QListWidget()
        score_list.setMaximumHeight(90)
        self._load_score_list(score_list)
        score_list.currentTextChanged.connect(
            lambda name: self._load_score_into_editor(name, profile, lyrics, tempo, melody)
        )
        generate = QPushButton("Generar canto")
        generate.clicked.connect(
            lambda: self._generate(profile.currentText(), lyrics.toPlainText(), melody.toPlainText(), tempo.value())
        )

        # Player controls
        player_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_folder = QPushButton("Abrir carpeta")
        self.btn_save_as = QPushButton("Guardar como...")

        for btn in [self.btn_play, self.btn_pause, self.btn_stop, self.btn_folder, self.btn_save_as]:
            btn.setEnabled(False)
            player_layout.addWidget(btn)

        self.btn_play.clicked.connect(self._play)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_save_as.clicked.connect(self._save_as)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Motor de render:"))
        layout.addWidget(self.renderer_sel)
        layout.addWidget(self.btn_compile)
        layout.addWidget(QLabel("Score:"))
        layout.addWidget(score_list)
        layout.addWidget(QLabel("Letra:"))
        layout.addWidget(lyrics)
        layout.addWidget(QLabel("Tempo:"))
        layout.addWidget(tempo)
        layout.addWidget(QLabel("Melodía (JSON):"))
        layout.addWidget(melody)
        layout.addLayout(player_layout)
        layout.addWidget(generate)
        layout.addWidget(self.status)
        self.setLayout(layout)
        if score_list.count():
            score_list.setCurrentRow(0)

    def _generate(self, profile: str, lyrics: str, melody_json: str, tempo: int) -> None:
        """Validate coverage and render vocal.wav."""

        try:
            payload = json.loads(melody_json)
            notes = [NoteEvent(**item) for item in payload.get("notes", [])]
            coverage = self.engine.coverage_for(profile, lyrics, notes)
            if coverage["missing"]:
                self.status.setText("Voicebank incompleto. Faltan aliases: " + ", ".join(coverage["missing"]))
                return

            renderer = self.renderer_sel.currentData()
            # The bridge is UTAU CLI-compatible, not OpenUtau WORLDLINE-R.
            actual_renderer = "classic" if renderer == "worldline" else renderer

            out_path = self.root / "outputs" / "vocal.wav"
            self.output_wav = self.engine.synthesize_to_path(profile, lyrics, notes, tempo, out_path, renderer=actual_renderer)

            self.status.setText(f"WAV generado: {self.output_wav.name} ({renderer})")
            for btn in [self.btn_play, self.btn_pause, self.btn_stop, self.btn_folder, self.btn_save_as]:
                btn.setEnabled(True)
        except (json.JSONDecodeError, TypeError, RuntimeError, ValueError) as exc:
            self.status.setText(f"No se pudo generar: {exc}")

    def _play(self) -> None:
        if self.player and self.output_wav and self.output_wav.exists():
            self.player.setSource(QUrl.fromLocalFile(str(self.output_wav)))
            self.audio_output.setVolume(1.0)
            self.player.play()

    def _pause(self) -> None:
        if self.player:
            self.player.pause()

    def _stop(self) -> None:
        if self.player:
            self.player.stop()

    def _open_folder(self) -> None:
        if self.output_wav:
            os.startfile(self.output_wav.parent)

    def _save_as(self) -> None:
        if self.output_wav and self.output_wav.exists():
            path, _ = QFileDialog.getSaveFileName(self, "Guardar WAV como", "", "Audio Files (*.wav)")
            if path:
                import shutil
                shutil.copy2(self.output_wav, path)

    def _compile_renderer(self) -> None:
        """Trigger compilation of the local UTAU bridge."""
        self.status.setText("Compilando LYKENOX UTAU Bridge...")
        QApplication.processEvents()

        success = self.engine.compile_worldline()
        if success:
            self.status.setText("LYKENOX UTAU Bridge compilado con éxito.")
            self.renderer_sel.setCurrentIndex(1) # Select bridge
        else:
            self.status.setText("Error al compilar. Asegúrate de tener .NET Framework instalado.")

    def _load_score_list(self, score_list: QListWidget) -> None:
        """Load available score files into the panel."""

        self.scores_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.scores_dir.glob("*.json")):
            score_list.addItem(path.name)

    def _load_score_into_editor(
        self,
        name: str,
        profile: QComboBox,
        lyrics: QPlainTextEdit,
        tempo: QSpinBox,
        melody: QPlainTextEdit,
    ) -> None:
        """Load a saved score file into the editable controls."""

        if not name:
            return
        try:
            score = load_score(self.scores_dir / name)
            index = profile.findText(score.profile)
            if index >= 0:
                profile.setCurrentIndex(index)
            lyrics.setPlainText(score.lyrics)
            tempo.setValue(score.tempo)
            melody.setPlainText(score_to_json(score))
            self.status.setText(f"Score cargado: {name}")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.status.setText(f"No se pudo cargar score: {exc}")
