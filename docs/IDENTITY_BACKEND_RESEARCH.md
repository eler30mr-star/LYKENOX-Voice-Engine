# LYKENOX Identity Backend Research

Date: 2026-08-26

## Fixed Objective

LYKENOX must be a direct identity voice engine:

```text
text -> LYKENOX identity model -> speech.wav
lyrics + melody/prosody -> LYKENOX identity model -> singing.wav
```

Rejected by definition:

- source speaker/singer audio at inference
- RVC
- SVC as the main architecture
- voice conversion presented as LYKENOX
- random backend installs before feasibility is proven

## Current Hardware Boundary

- Windows laptop
- Intel i7-1255U class CPU
- 12 GB RAM class
- No CUDA
- No functional local GPU path

This machine can run some inference slowly, but it is not a professional target
for training modern neural TTS/SVS models from scratch.

## Investigated Families

### Piper

Official training flow: prepare dataset, train/fine-tune VITS, export ONNX.

Evidence from the official Piper training guide:

- Piper supports training or fine-tuning a voice.
- The guide says most cases should fine-tune from an existing model.
- The example training path uses GPU acceleration.
- Piper is strongest after a model already exists because ONNX inference is
  CPU-friendly.

Fit:

- Direct TTS: yes, after a model exists.
- Own voice: yes, if trained/fine-tuned correctly.
- Spanish: possible through Spanish phonemization/model path.
- Singing: no. Piper is speech TTS, not score-controlled singing.
- CPU training on this laptop: not a good professional route.

Verdict: possible future speech backend if training happens on suitable hardware.
Not the next local CPU training step.

### Mimic 3

Official positioning: local neural TTS designed for CPU/low-end inference.

Evidence from Mycroft/Mimic documentation:

- Mimic 3 is local/offline neural TTS.
- It can run on low-end hardware.
- Creating a custom voice is described as a significant time investment and has
  historically required many hours of consistent audio.

Fit:

- Direct TTS: yes, after a model exists.
- Own voice: only through a proper custom voice process.
- Spanish: depends on model/language resources.
- Singing: no score-controlled singing path.
- CPU training on this laptop: not a quick or validated route.

Verdict: useful class for CPU inference, not a proven local custom-voice training
route for this project.

### Coqui TTS / VITS

Local attempt already failed.

Evidence:

- Training ran locally.
- Generated output `outputs/identity_epoch1/speech_epoch1.wav` was measured as
  silence/invalid audio.
- The result must not be used as progress or connected to the app.

Fit:

- Direct TTS: theoretically yes.
- Own voice: theoretically yes with enough clean data and training.
- Singing: no score-controlled singing path in the tested route.
- CPU training on this laptop: rejected by local evidence.

Verdict: rejected for this project unless a completely different, proven recipe
is selected before installing/running anything.

### F5-TTS / Spanish-F5

Official usage pattern: zero-shot/reference-based TTS.

Evidence:

- Spanish-F5 exposes inference using reference audio plus reference text.
- It has Spanish model work available.
- It is TTS, not melody-controlled singing.
- Pretrained model license is non-commercial for the referenced model release.

Fit:

- Direct TTS: yes.
- Own timbre without full training: possible through user's own reference audio.
- Requires source speaker at inference: yes, a reference voice prompt is part of
  the normal interface.
- Singing: no proper lyrics+melody SVS path.
- CPU: possible only as a slow inference test; not assumed until measured.

Verdict: candidate only for speech microtest if the user accepts reference-prompt
TTS. It does not satisfy the strict "trained model with no reference prompt" goal.

### IndexTTS

Official usage pattern: zero-shot voice cloning from reference audio.

Evidence:

- Current IndexTTS documentation describes Spanish support in IndexTTS 2.5.
- It uses speaker reference audio for cloning.
- It exposes Python/API-style inference.
- Documentation emphasizes BF16/FP16, DeepSpeed, CUDA kernels, or vLLM options
  for performance paths.

Fit:

- Direct TTS: yes.
- Spanish: yes according to current docs.
- Own timbre without training: yes, through the user's own reference audio.
- Requires source speaker at inference: yes, a reference prompt.
- Singing: no score-controlled singing path.
- CPU Windows 12 GB: uncertain and likely slow/heavy; must be proven before use.

Verdict: candidate only for speech research/microtest if reference-prompt TTS is
accepted. It is not the strict trained-identity backend yet.

### Amphion Vevo / Vevo1.5 / Vevo2

Official positioning: research toolkit for speech/singing generation, VC/SVC,
TTS/SVS, style/timbre control, and melody/prosody control.

Evidence:

- Amphion lists Vevo2 and Vevo1.5 under singing voice conversion and unified
  speech/singing generation.
- Vevo1.5 inference includes AR+FM for text, prosody, and style control.
- The documented language metadata for Vevo1.5 training lists `en`, `zh`, `ja`,
  `ko`, `fr`, and `de`, not Spanish.
- Environment setup is conda/Linux-style and the project provides Docker/NVIDIA
  workflows.

Fit:

- Singing/SVS: conceptually yes.
- Melody/prosody control: yes.
- Timbre reference: yes.
- Spanish: not supported in the documented Vevo1.5 language list.
- CPU Windows 12 GB: not a practical target.
- Source singer/conversion risk: high because many Vevo paths are VC/SVC.

Verdict: not viable for this laptop/project as the next route.

## Decision

There is no currently validated open-source backend that satisfies all of these
at once on this laptop:

- direct speech TTS in the user's identity
- direct singing with lyrics + melody in the user's identity
- no source speaker/singer prompt at inference
- no voice conversion
- Spanish
- local CPU training on 12 GB RAM

## Correct Engineering Plan

### Track A: Honest Local Singing Now

Use the existing WORLDLINE-R/sample-based path only as sample-based singing.

This can preserve the user's real recorded identity because it uses the user's
actual recorded phoneme samples, but it is not neural TTS and must not be
presented as one.

Next professional step:

- finish the multipitch microtest
- compare monopitch vs multipitch
- continue only if the user's perceived identity improves

### Track B: Neural Speech Identity

Do not continue CPU-from-scratch training locally.

Professional options:

1. Train/fine-tune a direct TTS model off-machine with suitable GPU hardware.
2. Use a reference-prompt TTS model only if the product definition allows a
   reference WAV prompt from the user's own voice at inference.
3. Keep `/speak` unavailable until one of the above is proven.

### Track C: Neural Singing Identity

Do not start local SVS training on this CPU laptop.

Professional options:

1. Keep singing sample-based with WORLDLINE-R multipitch.
2. For true neural SVS, investigate/train off-machine with a real SVS model and
   enough aligned singing data.

## No-Go Rules

Do not install or train another backend unless it passes this checklist first:

- exact repository identified
- license identified
- input contract identified
- Spanish support verified
- CPU inference feasibility estimated
- training/adaptation requirement identified
- source-speaker/source-singer dependency clearly absent or explicitly accepted
- expected checkpoint size identified
- expected RAM/VRAM identified
- test success criteria defined before running

## Immediate Next Step

The next practical step on this machine is not another neural install.

It is:

```text
complete and verify WORLDLINE-R multipitch microtest
```

That keeps the work aligned with real recordings, real pitch control, and the
current CPU hardware.
