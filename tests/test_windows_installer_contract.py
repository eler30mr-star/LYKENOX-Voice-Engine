from __future__ import annotations

from pathlib import Path


def test_installed_entry_uses_persistent_writable_workspace() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "lykenox_voice_engine" / "ui" / "installed_entry.py").read_text(encoding="utf-8")
    assert "LOCALAPPDATA" in text
    assert 'WORKSPACE_STATE_FILE = "workspace.json"' in text
    assert "QFileDialog.getExistingDirectory" in text
    assert "datasets" in text and "identity_voice" in text
    assert "os.chdir(workspace)" in text
    assert "MainWindow(workspace)" in text


def test_windows_installer_creates_shortcuts_and_launches_app() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "packaging" / "windows" / "LYKENOX.iss").read_text(encoding="utf-8")
    assert "OutputBaseFilename=LYKENOX-Setup" in text
    assert "{localappdata}\\Programs\\LYKENOX Voice Engine" in text
    assert "{autodesktop}" in text
    assert "PrivilegesRequired=lowest" in text
    assert 'Filename: "{app}\\{#MyAppExeName}"' in text


def test_ci_builds_and_uploads_setup_exe() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / ".github" / "workflows" / "build-windows-installer.yml").read_text(encoding="utf-8")
    assert "windows-latest" in text
    assert "pyinstaller --noconfirm --clean packaging/windows/LYKENOX.spec" in text
    assert "Inno Setup 6\\ISCC.exe" in text
    assert "packaging/windows/output/LYKENOX-Setup.exe" in text
    assert "actions/upload-artifact@v4" in text
    assert "LYKENOX-Windows-Installer" in text
