"""Native PySide6 desktop UI for LYKENOX Voice Engine."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QLabel, QListWidget, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from lykenox_voice_engine.ui.dataset_page import DatasetPage
from lykenox_voice_engine.ui.synthesis_page import SynthesisPage
from lykenox_voice_engine.ui.training_page import TrainingPage


class MainWindow(QMainWindow):
    """Main native desktop window."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.setWindowTitle("LYKENOX Voice Engine")
        self.resize(1180, 760)
        nav = QListWidget()
        stack = QStackedWidget()
        nav.addItems(["Perfil", "Dataset", "Entrenamiento", "Sintesis"])
        stack.addWidget(QLabel("Perfil: LYKENOX Voice | Estado: scaffold sin backend IA"))
        stack.addWidget(DatasetPage(root))
        stack.addWidget(TrainingPage())
        stack.addWidget(SynthesisPage())
        nav.currentRowChanged.connect(stack.setCurrentIndex)
        nav.setCurrentRow(0)
        layout = QVBoxLayout()
        layout.addWidget(nav)
        layout.addWidget(stack)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


def run_app(root: Path) -> int:
    """Run the native Windows desktop application."""

    app = QApplication([])
    window = MainWindow(root)
    window.show()
    return app.exec()
