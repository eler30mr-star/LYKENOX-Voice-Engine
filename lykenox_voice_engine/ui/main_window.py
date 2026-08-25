"""Main PySide6 window."""

import sys

from PySide6.QtWidgets import QApplication, QListWidget, QListWidgetItem, QMainWindow, QStackedWidget, QHBoxLayout, QWidget

from lykenox_voice_engine.ui.dataset_page import DatasetPage
from lykenox_voice_engine.ui.synthesis_page import SynthesisPage
from lykenox_voice_engine.ui.training_page import TrainingPage


class MainWindow(QMainWindow):
    """Desktop shell for LYKENOX Voice Engine."""

    def __init__(self) -> None:
        """Create app navigation and pages."""
        super().__init__()
        self.setWindowTitle("LYKENOX Voice Engine")
        central = QWidget()
        layout = QHBoxLayout(central)
        self.nav = QListWidget()
        self.nav.setFixedWidth(180)
        for name in ["Dataset", "Entrenar", "Cantar"]:
            self.nav.addItem(QListWidgetItem(name))
        self.pages = QStackedWidget()
        self.pages.addWidget(DatasetPage())
        self.pages.addWidget(TrainingPage())
        self.pages.addWidget(SynthesisPage())
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        layout.addWidget(self.nav)
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.nav.setCurrentRow(0)


def main() -> None:
    """Run the desktop application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 720)
    window.show()
    sys.exit(app.exec())
