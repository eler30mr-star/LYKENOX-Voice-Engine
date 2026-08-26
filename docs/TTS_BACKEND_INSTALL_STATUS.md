# LYKENOX TTS Backend Install Status

Date: 2026-08-26

## Goal

Install the local CPU runtime needed for a first direct LYKENOX speech-model
microtraining path. This is not RVC, SVC, or voice conversion.

## Installed Runtime

- Python: 3.12.14 in `.venv`
- PyTorch: `torch 2.13.0+cpu`
- torchaudio: `2.11.0+cpu`
- librosa: `1.0.0`
- soundfile: `0.14.0`
- CUDA available: no

## Backend Candidates Checked

- Coqui `TTS`: not installable from pip for this Python 3.12 environment.
- Piper phonemizer pip package: not available from pip in this environment.
- PyTorch CPU runtime: installed and verified.

## Current Status

The machine can now run CPU neural/audio preflight and feed a local microtraining
script from:

`datasets/lykenox/identity_voice/prepared/speech/train.auto.csv`

This is not yet a trained voice model. `/speak` must continue to fail honestly
until a real checkpoint exists.

## Next Engineering Step

Implement the smallest real CPU trainable speech model path, then train a trial
checkpoint and test whether it can overfit short LYKENOX Spanish utterances.
