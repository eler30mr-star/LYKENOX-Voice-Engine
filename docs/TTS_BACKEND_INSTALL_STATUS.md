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
- PyTorch: `2.5.1+cpu`
- torchaudio: `2.5.1+cpu`
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

After filtering for direct TTS training quality:

- Total prepared rows: 33
- Directly usable rows: 6
- Directly usable duration: 1.05 minutes
- Blocked rows: 27
- Blocked by generic metadata: 5
- Blocked by duplicate text over the safe limit: 9
- Blocked because long files need segmentation first: 13
- Recoverable long-audio duration if segmented: 24.6 minutes

Filtered files:

- `datasets/lykenox/identity_voice/prepared/speech/manifest.filtered.jsonl`
- `datasets/lykenox/identity_voice/prepared/speech/train.filtered.csv`
- `datasets/lykenox/identity_voice/prepared/speech/val.filtered.csv`
- `datasets/lykenox/identity_voice/prepared/speech/blocked.filtered.csv`

After segmenting the 13 long recordings:

- Segmented clips: 132
- Train rows: 118
- Validation rows: 14
- Segmented duration: 23.84 minutes
- Missing WAVs: 0
- CPU preflight: passed

Coqui smoke training:

- Command: `python -m TTS.bin.train_tts --config_path models/lykenox_identity/coqui_smoke/config.json --small_run 8`
- Result: completed one small-run epoch on CPU
- Best checkpoint: `models/lykenox_identity/coqui_smoke/lykenox_coqui_vits_smoke-August-26-2026_04+23PM-5733fd3/best_model_7.pth`
- Smoke WAV: `outputs/identity_smoke/speech_coqui_smoke.wav`
- Smoke WAV format: 48000 Hz mono PCM16, 1.20 seconds

This smoke checkpoint is not a production LYKENOX voice. It proves that local
training and inference work end-to-end on CPU.

Coqui one-epoch CPU training:

- Command: `python -m TTS.bin.train_tts --config_path models/lykenox_identity/coqui_smoke/config.json`
- Result: completed 1 epoch on CPU
- Training rows loaded: 118
- Training rows used after length filtering: 98
- Validation rows loaded: 14
- Validation rows used after length filtering: 11
- Best checkpoint: `models/lykenox_identity/coqui_smoke/lykenox_coqui_vits_smoke-August-26-2026_04+32PM-0c34b36/best_model_98.pth`
- Epoch-1 WAV: `outputs/identity_epoch1/speech_epoch1.wav`
- Epoch-1 WAV format: 48000 Hz mono PCM16, 2.13 seconds

Known data/config limitations:

- Training metadata was normalized to ASCII for Coqui's default character set.
- The Spanish-specific characters are preserved in source manifests, but not in
  the current Coqui metadata used for this run.
- Some samples were discarded because `max_audio_len` excludes clips longer than
  about 18 seconds.
- One validation row still contained an inverted question mark transformed into
  an unsupported character by normalization; this should be cleaned before a
  longer run.

## Next Engineering Step

Do not train the final voice from the unfiltered dataset. Either segment the
13 long recordings into short sentence-level clips, or record new short prompted
sentences with exact text. The current 6 usable clips are enough only for a
runtime smoke test, not for a meaningful identity model.
