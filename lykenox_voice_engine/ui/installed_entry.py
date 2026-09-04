"""Installed desktop entry point for LYKENOX Voice Engine.

The installed executable contains application code only. User-owned datasets, manifests, models and
outputs remain in a writable LYKENOX workspace selected on first launch and remembered under the
current user's LocalAppData directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from lykenox_voice_engine.ui.main_window import MainWindow

APP_NAME = "LYKENOX Voice Engine"
STATE_DIR_NAME = "LYKENOX Voice Engine"
WORKSPACE_STATE_FILE = "workspace.json"


def _state_path() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else (Path.home() / "AppData" / "Local")
    return base / STATE_DIR_NAME / WORKSPACE_STATE_FILE


def _is_workspace(path: Path) -> bool:
    path = Path(path)
    return (
        path.is_dir()
        and (path / "datasets" / "lykenox" / "identity_voice").is_dir()
        and (path / "LYKENOX_IDENTITY_DISTRIBUTION_POLICY.md").is_file()
    )


def _load_saved_workspace() -> Path | None:
    state = _state_path()
    if not state.exists():
        return None
    try:
        payload = json.loads(state.read_text(encoding="utf-8"))
        path = Path(str(payload.get("workspace", ""))).expanduser().resolve()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return path if _is_workspace(path) else None


def _save_workspace(path: Path) -> None:
    state = _state_path()
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps({"workspace": str(path.resolve())}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _workspace_from_argv() -> Path | None:
    if "--workspace" not in sys.argv:
        return None
    index = sys.argv.index("--workspace")
    if index + 1 >= len(sys.argv):
        return None
    candidate = Path(sys.argv[index + 1]).expanduser().resolve()
    return candidate if _is_workspace(candidate) else None


def _select_workspace() -> Path | None:
    selected = QFileDialog.getExistingDirectory(
        None,
        "Selecciona la carpeta de trabajo de LYKENOX",
        str(Path.home()),
    )
    if not selected:
        return None
    path = Path(selected).resolve()
    if not _is_workspace(path):
        QMessageBox.critical(
            None,
            "Carpeta no válida",
            "Selecciona la raíz de LYKENOX-Voice-Engine. Debe contener datasets\\lykenox\\identity_voice "
            "y LYKENOX_IDENTITY_DISTRIBUTION_POLICY.md.",
        )
        return None
    return path


def resolve_workspace() -> Path | None:
    explicit = _workspace_from_argv()
    if explicit is not None:
        _save_workspace(explicit)
        return explicit
    saved = _load_saved_workspace()
    if saved is not None:
        return saved
    return _select_workspace()


def main() -> int:
    app = QApplication([])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("LYKENOX")

    workspace = resolve_workspace()
    if workspace is None:
        QMessageBox.information(
            None,
            APP_NAME,
            "No se seleccionó un workspace. LYKENOX no modificó ningún dato.",
        )
        return 2

    _save_workspace(workspace)
    os.chdir(workspace)

    try:
        window = MainWindow(workspace)
        window.setWindowTitle(APP_NAME)
        window.show()
        return app.exec()
    except Exception as exc:  # GUI boundary: surface startup failure instead of silent exit.
        QMessageBox.critical(None, "LYKENOX no pudo iniciar", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
