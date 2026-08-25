"""Singing synthesis page."""

from PySide6.QtWidgets import QFormLayout, QLabel, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

from lykenox_voice_engine.core.synthesis_service import simple_test_melody


class SynthesisPage(QWidget):
    """Collect lyrics and melody for direct singing synthesis."""

    def __init__(self) -> None:
        """Create synthesis controls."""
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.profile = QLabel("LYKENOX Voice")
        self.lyrics = QPlainTextEdit("Bailando te conocí,\nla noche brillaba por ti,\nven quédate junto a mí,\nbaila esta noche para mí.")
        self.tempo = QSpinBox()
        self.tempo.setRange(40, 300)
        self.tempo.setValue(140)
        self.melody = QPlainTextEdit(str([note.to_dict() for note in simple_test_melody()]))
        self.status = QLabel("Sin motor SVS seleccionado")
        button = QPushButton("Generar canto")
        button.clicked.connect(self.generate)
        form.addRow("Perfil", self.profile)
        form.addRow("Letra", self.lyrics)
        form.addRow("Tempo", self.tempo)
        form.addRow("Melodía JSON", self.melody)
        layout.addLayout(form)
        layout.addWidget(button)
        layout.addWidget(self.status)
        layout.addStretch()

    def generate(self) -> None:
        """Refuse fake synthesis until a real backend is selected."""
        self.status.setText("No se genera audio falso: falta backend SVS real.")
