"""Training page."""

from PySide6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from lykenox_voice_engine.engines.audit_only import AuditOnlyEngine


class TrainingPage(QWidget):
    """Training controls with mandatory microtest gate."""

    def __init__(self) -> None:
        """Create training page."""
        super().__init__()
        self.engine = AuditOnlyEngine()
        layout = QVBoxLayout(self)
        layout.addWidget(self._summary())
        row = QHBoxLayout()
        for text, handler in [("Comprobar", self.check), ("Microtest", self.microtest), ("Entrenar", self.train), ("Continuar", self.resume), ("Detener", self.stop)]:
            button = QPushButton(text)
            button.clicked.connect(handler)
            row.addWidget(button)
        row.addStretch()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addLayout(row)
        layout.addWidget(self.log, 1)

    def _summary(self) -> QGroupBox:
        """Create training summary."""
        box = QGroupBox("Entrenar")
        grid = QGridLayout(box)
        for row, (key, value) in enumerate([("Perfil", "LYKENOX Voice"), ("Motor", "pendiente"), ("Device", "cpu"), ("Checkpoint", "-")]):
            grid.addWidget(QLabel(key), row, 0)
            grid.addWidget(QLabel(value), row, 1)
        return box

    def check(self) -> None:
        """Show backend availability."""
        self.log.append(str(self.engine.check_available()))

    def microtest(self) -> None:
        """Require backend selection before real microtraining."""
        self.log.append("Microtest pendiente: primero elegir backend SVS real.")

    def train(self) -> None:
        """Block long training in scaffold phase."""
        self.log.append("No se entrena: falta auditoría y microtest.")

    def resume(self) -> None:
        """Show resume placeholder."""
        self.log.append("Continuar requiere checkpoint real del backend elegido.")

    def stop(self) -> None:
        """Stop placeholder."""
        self.log.append("No hay proceso activo.")
