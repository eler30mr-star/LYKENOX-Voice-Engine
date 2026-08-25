"""Training readiness page."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from lykenox_voice_engine.core.training_service import TrainingService


class TrainingPage(QWidget):
    """Show NNSVS microtest controls without enabling full training."""

    def __init__(self) -> None:
        super().__init__()
        self._service = TrainingService()
        self._status = QLabel("NNSVS: pendiente | Device: CPU | Entrenamiento completo desactivado")
        layout = QVBoxLayout()
        layout.addWidget(self._status)
        for text, handler, enabled in [
            ("Comprobar NNSVS", self._check, True),
            ("Preparar microtest", self._prepare, True),
            ("Ejecutar microtest", self._microtest, True),
            ("Detener", self._stop, True),
            ("Entrenamiento completo", None, False),
        ]:
            button = QPushButton(text)
            button.setEnabled(enabled)
            if handler:
                button.clicked.connect(handler)
            layout.addWidget(button)
        self.setLayout(layout)

    def _check(self) -> None:
        """Display NNSVS backend readiness."""

        result = self._service.check()
        self._status.setText(f"NNSVS disponible: {result.get('available')} | CPU | {result.get('reason', '')}")

    def _prepare(self) -> None:
        """Prepare the microtest dataset and show status."""

        result = self._service.prepare_microtest()
        self._status.setText(f"Preparacion: {result.get('ok')} | {result.get('status')}")

    def _microtest(self) -> None:
        """Run the safe NNSVS microtraining gate."""

        result = self._service.microtest()
        self._status.setText(f"Microtest: {result.get('ok')} | {result.get('reason')}")

    def _stop(self) -> None:
        """Cancel backend jobs."""

        result = self._service.stop()
        self._status.setText(f"Stop: {result['status']}")
