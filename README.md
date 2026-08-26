# LYKENOX Voice Engine

Personal LYKENOX identity voice engine for direct Spanish speech and singing synthesis.

Final objective:

```text
text -> LYKENOX identity model -> speech.wav
lyrics + melody -> LYKENOX identity model -> singing.wav
```

The main architecture is not RVC, SVC, or post-conversion of another voice. The target is
an original voice model trained/adapted from the owner's recordings. WORLDLINE-R and the
UTAU-style voicebank are kept as local baselines and fallback research tools, not as the
final neural model.

See `docs/LYKENOX_IDENTITY_VOICE_ARCHITECTURE.md`.

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

## Identity Dataset

Start here for the real model objective:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_identity_dataset.py
.\.venv\Scripts\python.exe scripts\run_app.py
```

Open `Grabar Identidad` and record full Spanish speech and singing phrases. These
recordings are the source dataset for the future LYKENOX personal speech/singing model.

Target API:

- `POST /speak`: direct text-to-speech with the trained LYKENOX identity model.
- `POST /sing`: direct text-to-singing with lyrics and notes.
- `POST /synthesize`: legacy WORLDLINE/voicebank compatibility endpoint.
