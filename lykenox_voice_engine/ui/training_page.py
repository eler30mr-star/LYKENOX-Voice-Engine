"""Local readiness page for the LYKENOX voicebank route."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from lykenox_voice_engine.core.training_service import TrainingService


class TrainingPage(QWidget):
    """Show local identity/voicebank readiness without fake neural training."""

    def __init__(self) -> None:
        super().__init__()
        self._service = TrainingService()
        self._status = QLabel("Ruta local: voicebank LYKENOX | CPU | entrenamiento neural desactivado")
        layout = QVBoxLayout()
        layout.addWidget(self._status)
        for text, handler, enabled in [
            ("Comprobar ruta local", self._check, True),
            ("Preparar carpetas", self._prepare, True),
            ("Microtest neural", self._microtest, False),
            ("Detener", self._stop, True),
        ]:
            button = QPushButton(text)
            button.setEnabled(enabled)
            if handler:
                button.clicked.connect(handler)
            layout.addWidget(button)
        self.setLayout(layout)

    def _check(self) -> None:
        """Display local backend readiness."""

        result = self._service.check()
        voicebank = result.get("voicebank", {})
        self._status.setText(
            f"Ruta local OK | cobertura voicebank: {voicebank.get('voicebank_coverage', 0)}% | "
            f"{result.get('reason', '')}"
        )

    def _prepare(self) -> None:
        """Prepare the microtest dataset and show status."""

        result = self._service.prepare_microtest()
        self._status.setText(f"Preparacion: {result.get('ok')} | {result.get('status')}")

    def _microtest(self) -> None:
        """Show why neural microtraining is disabled."""

        result = self._service.microtest()
        self._status.setText(f"Microtest: {result.get('ok')} | {result.get('reason')}")

    def _stop(self) -> None:
        """Return current idle state."""

        result = self._service.stop()
        self._status.setText(f"Stop: {result['status']}")
