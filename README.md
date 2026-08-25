# LYKENOX Voice Engine

Local desktop/API scaffold for singing voice synthesis research and integration.

This project is separate from ACE-Step and does not use RVC or voice conversion as its main architecture.

## Run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m lykenox_voice_engine
```

## API

Default host is `127.0.0.1`, port `8765`.

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.api.server
```
