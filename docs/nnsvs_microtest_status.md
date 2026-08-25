# NNSVS CPU Microtest Status

## Runtime

- Isolated env: `tools/nnsvs_env/.venv`
- Python: 3.12.14
- Official PyPI candidate: `nnsvs==0.1.1`
- Install attempt: timed out after 244 seconds; `nnsvs` was not installed.

## NNSVS format requirements

NNSVS recipes require score data and aligned audio. The documented stage 0 converts
MusicXML or UST to HTS-style full-context labels, segments singing data, and splits
train/dev/test. Stage 1 extracts acoustic, duration, time-lag, and pitch-related
features from aligned labels plus WAV.

For Spanish, no supported Spanish frontend is confirmed in this environment. The app
therefore records the required phoneme inventory and micro-score, but does not invent
HTS labels.

## Decision

The backend integration is implemented as a hard gate. Training and synthesis remain
blocked until NNSVS imports successfully and a real MusicXML/UST-to-HTS Spanish path is
validated.
