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
from lykenox_voice_engine.ui.recording_v2_bootstrap import ensure_recording_v2_pilot
from lykenox_voice_engine.ui.recording_v2_page import RecordingV2Page
from lykenox_voice_engine.ui.synthesis_page import SynthesisPage
from lykenox_voice_engine.ui.voicebank_page import VoicebankPage


class MainWindow(QMainWindow):
    """Main native desktop window."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.setWindowTitle("LYKENOX Voice Engine")
        self.resize(1180, 760)

        # Desktop-first contract: metadata needed by the active RECORDING_V2 gate is prepared
        # automatically. This never modifies audio and removes the need to run PowerShell helpers.
        ensure_recording_v2_pilot(root)

        nav = QListWidget()
        stack = QStackedWidget()
        nav.addItems(
            [
                "Perfil",
                "Grabar RECORDING_V2",
                "Grabar Identidad Legacy",
                "Cantar Legacy",
                "Voicebank Legacy",
            ]
        )
        stack.addWidget(
            QLabel("Perfil: LYKENOX Identity Voice | Objetivo: speech + singing directo con mi voz")
        )
        stack.addWidget(RecordingV2Page(root))
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
