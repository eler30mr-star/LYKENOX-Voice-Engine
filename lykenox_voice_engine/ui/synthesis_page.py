"""Singing synthesis request page."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QLabel, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget


class SynthesisPage(QWidget):
    """Collect lyrics and note data for direct singing synthesis."""

    def __init__(self) -> None:
        super().__init__()
        status = QLabel("Salida: pendiente | Backend: no instalado")
        profile = QComboBox()
        profile.addItem("lykenox")
        lyrics = QPlainTextEdit("la la la")
        tempo = QSpinBox()
        tempo.setRange(40, 240)
        tempo.setValue(120)
        pitch = QSpinBox()
        pitch.setRange(-24, 24)
        melody = QPlainTextEdit(
            '{"tempo": 120, "notes": [{"lyric": "la", "midi": 60, "start": 0.0, "duration": 1.0}]}'
        )
        generate = QPushButton("Generar canto")
        generate.clicked.connect(lambda: status.setText("Backend no instalado; no se genero audio falso."))
        layout = QVBoxLayout()
        for widget in [profile, lyrics, tempo, melody, pitch, generate, status]:
            layout.addWidget(widget)
        self.setLayout(layout)
