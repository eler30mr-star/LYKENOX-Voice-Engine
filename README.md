# LYKENOX Voice Engine

Native local singing voice synthesis scaffold for the LYKENOX workflow.

This project is intentionally separate from ACE-Step. Phase 1 provides the desktop UI,
local API contract, profile storage, dataset manager, backend interfaces, and technical
audit. It does not download large AI models and does not train a singing model yet.

## Run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_app.py
```

API only:

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py
```

Default API: `http://127.0.0.1:8765`.
