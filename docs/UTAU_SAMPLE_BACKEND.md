# LYKENOX Spanish Lite Sample Backend

This backend is the primary local route for LYKENOX Voice Engine. It is a UTAU-style,
sample-based renderer for Spanish voicebanks.

## Renderer

The first renderer is `internal_concat_pcm`: a local CPU renderer that reads validated
48 kHz mono 16-bit PCM WAV samples from the selected voicebank and writes
`outputs/<job_id>/vocal.wav` from score note durations.

It does not use RVC, SVC, neural conversion, source singer audio, CUDA, XPU, cloud, or
OpenUtau as an external runtime. OpenUtau/UTAU only informs the voicebank format:
`wav/`, `oto.ini`, `character.txt`, `prefix.map`, `phonemes.json`, `reclist.txt`, and
`config.json`.

## Spanish Lite Voicebank

`profiles/lykenox/voicebank/reclist.txt` contains the initial Spanish Lite aliases.
The target format for accepted recordings is:

- WAV PCM
- 48 kHz
- mono
- 16-bit

Raw recordings are kept in `datasets/lykenox/voicebank_raw/`. Accepted samples are
copied into `profiles/lykenox/voicebank/wav/` by the voicebank builder. Personal WAV
files are ignored by Git.

## OTO

`oto.ini` stores:

`wav=alias,offset,consonant,cutoff,preutterance,overlap`

Initial timing uses an energy-based estimator. It is deliberately lightweight and local.
A full waveform editor with draggable markers is still a UI follow-up.

## Spanish Phonemizer

`lykenox_voice_engine/core/spanish_phonemizer.py` implements practical first-pass rules
for b/v, c/qu/k, g/gu, j/g before e/i, ñ, ch, ll/y, r/rr, silent h, x, and common
diphthongs. It targets voicebank coverage rather than perfect linguistic analysis.

Example:

`baila conmigo` -> `bai`, `la`, `con`, `mi`, `go`

## API

`GET /health` reports:

- `backend: identity_voice_target`
- `legacy_backend: utau_worldline_fallback`
- `voicebank_available`
- `voicebank_coverage`
- `renderer_available`

`POST /synthesize` receives `profile`, `lyrics`, `notes`, and `tempo`. It refuses to
render when required aliases are missing.

## Future Music Studio Flow

LYKENOX Music Studio should later compute or import lyrics, melody, durations, and tempo,
then call this API directly and receive `vocal.wav` for mixing with the instrumental.
There is no source singer and no voice conversion in this route.
