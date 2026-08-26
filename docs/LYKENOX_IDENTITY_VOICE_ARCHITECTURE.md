# LYKENOX Identity Voice Architecture

Date: 2026-08-26
Project: D:\Proyectos\LYKENOX-Voice-Engine

## Objective

LYKENOX Voice Engine is a personal identity voice model project.

The final product must generate audio directly from text, lyrics, and melody using the
owner's original voice identity:

```text
speech text -> LYKENOX identity model -> speech.wav
lyrics + melody/prosody -> LYKENOX identity model -> singing.wav
```

This is not RVC, SVC, or voice conversion. The model must not require a source singer
or another speaker recording at inference time. The generated voice must be the trained
LYKENOX identity itself.

## Non-Goals

- Do not convert a third-party or synthetic singer into LYKENOX.
- Do not use RVC as the main architecture.
- Do not use SVC as the main architecture.
- Do not present WORLDLINE-R, UTAU, or multipitch resampling as the final neural model.
- Do not fake readiness by generating WAVs from a placeholder.

## Modes

### Speech

API target:

```json
POST /speak
{
  "profile": "lykenox",
  "text": "Hola, este es mi modelo de voz.",
  "language": "es"
}
```

Output:

```text
speech.wav
```

### Singing

API target:

```json
POST /sing
{
  "profile": "lykenox",
  "lyrics": "baila conmigo",
  "tempo": 120,
  "notes": [
    {"lyric": "bai", "midi": 60, "start": 0.0, "duration": 0.5},
    {"lyric": "la", "midi": 62, "start": 0.5, "duration": 0.5}
  ],
  "language": "es"
}
```

Output:

```text
singing.wav
```

## System Architecture

```text
Text/Lyrics
  -> Spanish text normalization
  -> phoneme/grapheme frontend
  -> duration/prosody planner
  -> acoustic identity model
  -> vocoder
  -> wav

Lyrics + melody
  -> Spanish singing frontend
  -> note/phoneme alignment
  -> pitch/duration conditioning
  -> acoustic identity model
  -> vocoder
  -> singing.wav
```

The same identity should be shared by speech and singing. Speech and singing may use
separate acoustic heads or adapters, but they must represent the same trained personal
voice.

## Dataset Strategy

The current 92-alias WORLDLINE voicebank is useful evidence, but it is not enough for a
professional TTS/SVS identity model.

Required dataset families:

- Clean spoken Spanish sentences for text-to-speech.
- Sustained vowels and phoneme coverage for timbre stability.
- Sung phrases with known lyrics and pitch/duration alignment.
- Calibration takes across comfortable low, mid, and high registers.
- Metadata per recording: text, lyrics, mode, language, sample rate, duration, F0 stats,
  loudness, clipping, noise, and approval state.

The data capture workflow should validate quality, not force exact pitch targets.

## Runtime Status

The current repository contains:

- WORLDLINE-R official integration for direct sample-based singing.
- Adaptive multipitch voicebank selection based on measured F0.
- API compatibility endpoints for legacy `/synthesize` and `/synthesize-midi`.
- New target endpoints `/speak` and `/sing`.

The current repository does not yet contain:

- A trained LYKENOX neural speech model.
- A trained LYKENOX neural singing model.
- A vocoder trained or adapted for the LYKENOX identity.
- A complete Spanish neural frontend for arbitrary text and singing alignment.

Therefore `/speak` and `/sing` must fail honestly until the model exists.

## Engine Roles

| Component | Role | Final? |
| --- | --- | --- |
| WORLDLINE-R | Sample-based fallback and research baseline | No |
| Adaptive multipitch | Better baseline for the current voicebank | No |
| Neural identity model | Direct original LYKENOX speech and singing | Yes |
| API | Contract used by other apps to request speech/singing | Yes |

## Training Direction

The professional path is to train or adapt a neural model using the owner's own data.
The model must learn the identity directly. If external pretrained components are used,
they may provide general language/acoustic priors, but inference must not depend on a
source singer or post-hoc voice conversion.

Minimum viable milestones:

1. Capture and validate a spoken Spanish corpus.
2. Capture and validate a small sung aligned corpus.
3. Build a unified metadata catalog.
4. Train or adapt a speech model for LYKENOX identity.
5. Train or adapt a singing model using lyrics + note conditioning.
6. Expose `/speak` and `/sing` through the same local API.
7. Keep WORLDLINE-R only as a comparison fallback.

## Success Criteria

- `/speak` reads unseen Spanish text with the LYKENOX voice.
- `/sing` sings unseen Spanish lyrics from a melody/score with the LYKENOX voice.
- No source singer is required at inference time.
- No RVC/SVC conversion is part of the main path.
- The other app can call this API and receive final vocal audio directly.
