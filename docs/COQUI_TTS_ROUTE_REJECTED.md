# Coqui TTS Route Rejected

Date: 2026-08-26

## Decision

The Coqui VITS-from-scratch route is rejected for the LYKENOX identity voice
engine.

It must not be presented as the main engine, as a successful voice model, or as
progress toward production-quality LYKENOX identity synthesis.

## User Objective

The required objective is direct generation:

```text
text -> LYKENOX identity model -> speech.wav
lyrics + melody/prosody -> LYKENOX identity model -> singing.wav
```

The generated voice must be the user's own trained identity.

## Not Allowed

- RVC
- SVC as the main architecture
- Source singer audio at inference time
- Generating with another voice and converting it afterward
- Presenting a third-party voice as LYKENOX
- Continuing a backend after it produces unusable output

## What Happened

The Coqui setup proved only that Python, dataset loading, CPU training, checkpoint
writing, and command-line inference can execute locally.

The produced audio was not usable:

- `outputs/identity_epoch1/speech_epoch1.wav`
- measured as silence/invalid output
- peak level: `-inf dB`
- RMS level: `-inf dB`
- min/max sample level: `0`

This is not an acceptable voice result.

## Root Cause

Training Coqui VITS from scratch on the current local CPU setup and the current
dataset is not a professional path to the user's objective.

The model is too large for this small, imperfect dataset and short CPU run. It
does not establish a usable speech identity model.

## Status

Rejected.

The scripts and local artifacts may remain only as evidence and tooling history.
They must not be wired into `/speak`, `/sing`, the UI, or any production path.

## Required Next Rule

Before any new backend is installed or trained, it must be checked against the
actual objective:

- direct text-to-speech with trained LYKENOX identity
- future direct text/melody-to-singing with trained LYKENOX identity
- no voice conversion
- no source singer requirement
- local Windows/CPU feasibility must be honestly evaluated before work starts
