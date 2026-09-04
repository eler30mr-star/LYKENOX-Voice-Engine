"""Native PySide6 desktop UI for LYKENOX Voice Engine."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from lykenox_voice_engine.ui.identity_dataset_page import IdentityDatasetPage
from lykenox_voice_engine.ui.synthesis_page import SynthesisPage
from lykenox_voice_engine.ui.voicebank_page import VoicebankPage


class MainWindow(QMainWindow):
    """Main native desktop window.

    Audio capture is intentionally not implemented here. RECORDING_V2 capture lives in the
    separate RecVoice application; this repository keeps only the LYKENOX dataset/training
    contracts and consumers of the resulting authorized WAV artifacts.
    """

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.setWindowTitle("LYKENOX Voice Engine")
        self.resize(1180, 760)

        nav = QListWidget()
        stack = QStackedWidget()
        nav.addItems(
            [
                "Perfil",
                "Grabar Identidad Legacy",
                "Cantar Legacy",
                "Voicebank Legacy",
            ]
        )
        stack.addWidget(
            QLabel(
                "Perfil: LYKENOX Identity Voice | Objetivo: speech + singing directo con mi voz | "
                "Captura RECORDING_V2: aplicación separada RecVoice"
            )
        )
        stack.addWidget(IdentityDatasetPage(root))
        stack.addWidget(SynthesisPage(root))
        stack.addWidget(VoicebankPage(root))
        nav.currentRowChanged.connect(stack.setCurrentIndex)
        nav.setCurrentRow(0)
        layout = QHBoxLayout()
        layout.addWidget(nav, 1)
        layout.addWidget(stack, 4)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


def run_app(root: Path) -> int:
    """Run the native Windows desktop application."""

    app = QApplication([])
    window = MainWindow(root)
    window.show()
    return app.exec()
