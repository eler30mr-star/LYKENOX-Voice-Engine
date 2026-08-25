"""Dataset management page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from lykenox_voice_engine.config.settings import load_settings
from lykenox_voice_engine.core.dataset_service import DatasetService


class DatasetPage(QWidget):
    """Import, inspect, and prepare vocal recordings."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._service = DatasetService(load_settings(root).datasets_dir)
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["Archivo", "Duracion", "Hz", "Canales", "Peak", "Estado"])
        layout = QVBoxLayout()
        for text, handler in [("Importar", self._import_files), ("Validar", self._refresh)]:
            button = QPushButton(text)
            button.clicked.connect(handler)
            layout.addWidget(button)
        for text in ["Escuchar", "Eliminar", "Preparar"]:
            button = QPushButton(text)
            button.setEnabled(False)
            layout.addWidget(button)
        layout.addWidget(self._table)
        self.setLayout(layout)
        self._refresh()

    def _import_files(self) -> None:
        """Import selected audio files into the raw dataset folder."""

        names, _ = QFileDialog.getOpenFileNames(self, "Importar audios")
        self._service.import_files("lykenox", [Path(name) for name in names])
        self._refresh()

    def _refresh(self) -> None:
        """Reload dataset metadata in the table."""

        rows = self._service.list_raw("lykenox")
        self._table.setRowCount(len(rows))
        for row, info in enumerate(rows):
            values = [
                info.path.name,
                "" if info.duration is None else f"{info.duration:.1f}s",
                "" if info.sample_rate is None else str(info.sample_rate),
                "" if info.channels is None else str(info.channels),
                "" if info.peak is None else f"{info.peak:.3f}",
                info.status,
            ]
            for column, value in enumerate(values):
                self._table.setItem(row, column, QTableWidgetItem(value))
