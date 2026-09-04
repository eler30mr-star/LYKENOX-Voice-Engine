# LYKENOX Identity Voice — RECORDING_V2 decision

Date: 2026-09-04  
Policy: LYX-POL-001 v1.1

## Decision

The existing 132-item speech corpus will no longer be treated as the preferred source for the next persistent vocoder/acoustic retraining cycle.

Auditory cleanup evidence showed:

- FFmpeg AFFTDN preserved voice identity but did not remove the relevant contamination and added an enclosed/muffled ambience; it is rejected for the 132-item batch.
- External DeepFilterNet calibration isolated background noise well and preserved the voice, but some exterior events still escaped in several utterances.
- Continuing to optimize cleanup of contaminated recordings is lower value and higher risk than replacing the source capture with cleaner recordings.

Therefore the active dataset-quality gate is **RECORDING_V2 clean recapture at source**.

## Status of the old 132-item corpus

The previous corpus and all RAW/prepared artifacts remain immutable and preserved as historical/forensic evidence.

They are not deleted and may still be used for controlled forensic comparisons, but they are **not authorized as the primary source for new persistent training** while RECORDING_V2 is pending.

No 132-item denoise batch is authorized as a substitute for RECORDING_V2.

## RECORDING_V2 capture contract

1. Record lossless WAV, mono, with no lossy codec.
2. Preferred capture format: 48 kHz / 24-bit PCM or float32. Engine-rate conversion, when required, occurs later in the owned preprocessing pipeline.
3. Disable microphone/OS noise suppression, AGC, voice enhancement, echo cancellation, compressor, limiter, EQ and dereverb during capture.
4. Keep microphone, room, speaking distance and input gain as stable as practical across the session.
5. Aim for healthy unclipped level. Peaks should normally remain below about -6 dBFS; clipping is an automatic retake.
6. Leave roughly 0.5–1.0 s of room tone before and after the spoken content where practical.
7. Speak naturally. Do not exaggerate diction or force a different vocal placement merely to make the dataset cleaner.
8. Any rooster/chicken call, motor/tool noise, foreign voice, knock, wind burst, strong traffic event, phone alert or similar event overlapping speech is a **retake**, not a denoise target.
9. Breath, consonant noise, mouth transients and natural vocal attacks are part of the identity and must not be removed merely because they are broadband.
10. RAW RECORDING_V2 takes are immutable once imported. Acceptance/rejection is represented by metadata, not destructive editing of RAW.

## Acceptance hierarchy

Technical checks can reject a take for clipping, invalid geometry, missing samples or obvious corruption. They cannot accept perceptual quality.

Human listening remains the final authority for:

- clean environment,
- unchanged identity/timbre,
- natural consonants/formants,
- natural breathing and attacks,
- absence of external learnable events,
- absence of room/processing artifacts that dominate the voice.

A contaminated take is retaken rather than aggressively repaired when practical.

## Dataset flow

```text
RECORDING_V2 RAW CAPTURE
        ↓
technical audit
        ↓
human auditory ACCEPT / RETAKE
        ↓
RECORDING_V2 ACCEPTED
        ↓
owned segmentation / manifest generation
        ↓
regenerate ALL mel / F0 / periodicity / residual / cepstrum / caches
        ↓
rerun real-residual + identity GOLD oracles
        ↓
only then reconsider persistent training
```

## Training freeze

Until RECORDING_V2 is captured, audited, accepted, derived targets are regenerated, and the GOLD oracle controls are rerun:

- further source training: **BLOCKED**
- further source architecture changes: **BLOCKED**
- predicted-duration changes: **BLOCKED**
- post-vocoder denoise/EQ/gain patches: **BLOCKED**

The fixed minimum-phase renderer remains unchanged.

## Reuse of text/splits

To minimize schedule cost, RECORDING_V2 reuses the existing 132 segmented prompt texts and train/validation split assignment unless a prompt is explicitly replaced for linguistic reasons. The acoustic recordings are new; the text design does not need to be reinvented.
