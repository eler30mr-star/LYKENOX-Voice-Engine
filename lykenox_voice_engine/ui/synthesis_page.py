"""Singing synthesis request page."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QLabel, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget


class SynthesisPage(QWidget):
    """Collect lyrics and note data for direct singing synthesis."""

    def __init__(self) -> None:
        super().__init__()
        status = QLabel("Salida: pendiente | Backend: NNSVS gate")
        profile = QComboBox()
        profile.addItem("lykenox")
        lyrics = QPlainTextEdit("baila conmigo")
        tempo = QSpinBox()
        tempo.setRange(40, 240)
        tempo.setValue(120)
        pitch = QSpinBox()
        pitch.setRange(-24, 24)
        melody = QPlainTextEdit(
            '{"tempo":120,"notes":[{"lyric":"bai","midi":60,"start":0.0,"duration":0.5},'
            '{"lyric":"la","midi":62,"start":0.5,"duration":0.5},'
            '{"lyric":"con","midi":64,"start":1.0,"duration":0.5},'
            '{"lyric":"mi","midi":62,"start":1.5,"duration":0.5},'
            '{"lyric":"go","midi":60,"start":2.0,"duration":0.75}]}'
        )
        generate = QPushButton("Generar canto")
        generate.clicked.connect(
            lambda: status.setText("Bloqueado: falta checkpoint NNSVS real; no se genera audio falso.")
        )
        layout = QVBoxLayout()
        for widget in [profile, lyrics, tempo, melody, pitch, generate, status]:
            layout.addWidget(widget)
        self.setLayout(layout)
