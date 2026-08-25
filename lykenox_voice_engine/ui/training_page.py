"""Training readiness page."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from lykenox_voice_engine.core.training_service import TrainingService


class TrainingPage(QWidget):
    """Show safe training controls without launching Phase 1 training."""

    def __init__(self) -> None:
        super().__init__()
        self._service = TrainingService()
        self._status = QLabel("Perfil: lykenox | Engine: no seleccionado | Device: CPU | RAM: pendiente")
        layout = QVBoxLayout()
        layout.addWidget(self._status)
        for text, handler, enabled in [
            ("Comprobar", self._check, True),
            ("Microtest", self._microtest, True),
            ("Entrenar", None, False),
            ("Continuar", None, False),
            ("Detener", None, False),
        ]:
            button = QPushButton(text)
            button.setEnabled(enabled)
            if handler:
                button.clicked.connect(handler)
            layout.addWidget(button)
        self.setLayout(layout)

    def _check(self) -> None:
        """Display backend readiness."""

        result = self._service.check()
        self._status.setText(f"Comprobar: {result['status']} | {result['reason']}")

    def _microtest(self) -> None:
        """Display microtest status."""

        result = self._service.microtest()
        self._status.setText(f"Microtest: {result['status']} | {result['reason']}")
