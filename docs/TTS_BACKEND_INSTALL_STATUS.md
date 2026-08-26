# LYKENOX TTS Backend Install Status

Date: 2026-08-26

## Goal

Install the local CPU runtime needed for a first direct LYKENOX speech-model
microtraining path. This is not RVC, SVC, or voice conversion.

## Installed Runtime: App Environment

- Python: 3.12.14 in `.venv`
- PyTorch: `torch 2.13.0+cpu`
- torchaudio: `2.11.0+cpu`
- librosa: `1.0.0`
- soundfile: `0.14.0`
- CUDA available: no

## Isolated Coqui TTS Environment

Coqui TTS does not install in the app's Python 3.12 environment, but it installs
successfully in an isolated Python 3.11 environment:

- Environment: `tools/tts_env/.venv`
- Python: `3.11.16`
- Coqui TTS: `0.22.0`
- PyTorch: `2.13.0+cpu`
- CUDA available: no
- Spanish frontend package: `gruut_lang_es` installed through Coqui TTS

Recreate with:

```powershell
tools\tts_env\setup_tts_env.ps1
```

Spanish Coqui models listed without downloading checkpoints:

- `tts_models/es/mai/tacotron2-DDC`
- `tts_models/es/css10/vits`

These are third-party Spanish voices, not the LYKENOX identity. They must not be
presented as the user's trained voice.

## Backend Candidates Checked

- Coqui `TTS`: installable with isolated Python 3.11; not installable from pip
  for the app's Python 3.12 environment.
- Piper phonemizer pip package: not available from pip in this environment.
- PyTorch CPU runtime: installed and verified.

## Current Status

The machine can now run CPU neural/audio preflight and feed a local microtraining
script from:

`datasets/lykenox/identity_voice/prepared/speech/train.auto.csv`

This is not yet a trained voice model. `/speak` must continue to fail honestly
until a real LYKENOX checkpoint exists.

## Next Engineering Step

Implement the smallest real CPU trainable speech model path, then train a trial
checkpoint and test whether it can overfit short LYKENOX Spanish utterances.
