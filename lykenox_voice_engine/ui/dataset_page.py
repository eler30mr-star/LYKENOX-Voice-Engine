"""Dataset page."""

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from lykenox_voice_engine.core.dataset_service import DatasetService


class DatasetPage(QWidget):
    """Import, inspect, and prepare singing datasets."""

    def __init__(self) -> None:
        """Create dataset controls."""
        super().__init__()
        self.service = DatasetService()
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        for text, handler in [("Importar", self.import_files), ("Preparar", self.prepare), ("Validar", self.refresh)]:
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch()
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["archivo", "duración", "sample rate", "canales", "peak", "estado"])
        layout.addLayout(buttons)
        layout.addWidget(self.table)
        self.refresh()

    def import_files(self) -> None:
        """Import authorized voice files."""
        files, _ = QFileDialog.getOpenFileNames(self, "Importar voz", "", "Audio (*.wav *.m4a *.mp3 *.flac *.ogg *.opus)")
        self.service.import_files(files)
        self.refresh()

    def prepare(self) -> None:
        """Placeholder for backend-specific non-destructive preparation."""
        self.refresh()

    def refresh(self) -> None:
        """Reload dataset table."""
        items = self.service.list_items()
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            values = [item.path.name, item.duration, item.sample_rate, item.channels, item.peak, item.status]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
