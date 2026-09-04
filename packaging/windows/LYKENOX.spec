# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parents[1]

soundfile_datas, soundfile_binaries, soundfile_hidden = collect_all("soundfile")
try:
    sfdata_datas, sfdata_binaries, sfdata_hidden = collect_all("_soundfile_data")
except Exception:
    sfdata_datas, sfdata_binaries, sfdata_hidden = [], [], []

hiddenimports = [
    "PySide6.QtMultimedia",
    "PySide6.QtNetwork",
    *soundfile_hidden,
    *sfdata_hidden,
]

a = Analysis(
    [str(ROOT / "lykenox_voice_engine" / "ui" / "installed_entry.py")],
    pathex=[str(ROOT)],
    binaries=[*soundfile_binaries, *sfdata_binaries],
    datas=[*soundfile_datas, *sfdata_datas],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "faster_whisper",
        "librosa",
        "numba",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LYKENOX",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LYKENOX",
)
