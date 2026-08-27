# LYKENOX Voice Engine

Standalone LYKENOX identity voice product for direct Spanish speech and singing synthesis.

Final objective:

```text
text -> LYKENOX speech identity model -> speech.wav
lyrics + melody/score -> LYKENOX singing identity model -> singing.wav
```

LYKENOX is the product. The final application must not require Piper, Coqui, OpenUtau,
RVC, SVC, a cloud API, a source speaker/singer, or a reference WAV at inference time.
Published architectures and redistributable infrastructure libraries may be used or
implemented internally, but the shipped runtime must load self-contained LYKENOX model
artifacts directly.

WORLDLINE-R and the UTAU-style voicebank remain local baselines/fallback research tools,
not the final neural identity model.

Architecture documents:

- `docs/LYKENOX_IDENTITY_VOICE_ARCHITECTURE.md`
- `docs/PRODUCT_INDEPENDENCE.md`
- `docs/LYKENOX_SPEECH_V0.md`
- `docs/LYKENOX_SPEECH_ALIGNMENT_PIPELINE.md`

## Product boundary

LYKENOX owns and keeps stable:

- master identity dataset and metadata
- Spanish frontend contracts
- model manifest/artifact layout
- training orchestration interfaces
- speech and singing runtime interfaces
- local API and desktop UI

Training recipes and neural architecture families are replaceable implementation details.
They must not become mandatory third-party product runtimes.

## Current neural speech milestone

The repository contains the first LYKENOX-owned speech stack under active validation:

```text
Spanish text
  -> LYKENOX es-phoneme-v1 frontend
  -> LYKENOX duration/alignment supervision
  -> LYKENOX compact acoustic model
  -> mel spectrogram
  -> future LYKENOX vocoder
  -> speech.wav
```

Validated locally on CPU so far:

- synthetic acoustic forward/backward gate
- real WAV -> mel feature cache
- real-data acoustic update smoke
- phoneme CTC/Viterbi forced-alignment smoke

The next controlled gate trains a persistent LYKENOX aligner against train/validation data,
saves the best versioned checkpoint with early stopping, and only after validation passes
generates checkpoint-bound duration caches plus an outlier audit:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_alignment_pipeline --epochs 20 --patience 4
```

Do not start long acoustic-model training until the duration audit is reviewed and the
aligned acoustic smoke gate passes.

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

The engine-neutral master dataset is the source of truth:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_identity_dataset.py
.\.venv\Scripts\python.exe scripts\run_app.py
```

Open `Grabar Identidad` and record full Spanish speech and singing phrases. Trainer-specific
metadata must be generated from this master dataset; it must never replace it.

Target API:

- `POST /speak`: direct text-to-speech with the persistent LYKENOX speech model.
- `POST /sing`: direct text-to-singing with lyrics and notes using the persistent LYKENOX singing model.
- `POST /synthesize`: legacy WORLDLINE/voicebank compatibility endpoint.

Until neural artifacts actually exist, `/speak` and `/sing` must fail honestly rather than
fall back to another voice or a placeholder.
