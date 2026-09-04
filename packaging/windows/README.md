# LYKENOX Windows Installer

The Windows desktop installer is built by GitHub Actions as `LYKENOX-Setup.exe`.

## User flow

1. Download the `LYKENOX-Windows-Installer` artifact from the **Build Windows Installer** workflow.
2. Run `LYKENOX-Setup.exe`.
3. The installer creates Desktop and Start Menu shortcuts.
4. On the first app launch only, select the writable LYKENOX workspace root, for example `D:\Proyectos\LYKENOX-Voice-Engine`.
5. The selected workspace is remembered in `%LOCALAPPDATA%\LYKENOX Voice Engine\workspace.json`.

The installed program code lives under `%LOCALAPPDATA%\Programs\LYKENOX Voice Engine`. User datasets, manifests, models and recordings stay in the selected workspace and are not copied into the install directory.

The installer contains no external learned voice model and does not change the RECORDING_V2 raw-audio policy.
