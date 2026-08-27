# Post-Multipitch Neural TTS Research

Date: 2026-08-27

## User Perceptual Result

The user compared:

- `outputs/comparison/vocal_monopitch.wav`
- `outputs/comparison/vocal_adaptive_multipitch.wav`

Result:

- Monopitch is more understandable and better pronounced.
- Neither output sounds sufficiently like the user's real voice.
- The main failure is identity/formants/naturalness, not only pitch.
- The sound is perceived as autotune-like.

Decision:

Do not expand the full Low/Mid/High voicebank yet. The microtest did not prove a
clear identity improvement, so recording the full 276-sample multipitch set is
not justified now.

## Fixed Objective

The target remains:

```text
text -> LYKENOX identity model inside the engine -> speech.wav
lyrics + melody -> LYKENOX identity model inside the engine -> singing.wav
```

Not allowed:

- source speaker or source singer audio at generation time
- RVC
- SVC as the main architecture
- post-conversion from another voice
- random installs before feasibility is proven

All work must run on the user's local Windows CPU machine.

## Current Dataset State

Current neural TTS dataset readiness report:

- `reports/tts_dataset_readiness.json`
- train rows: 118
- validation rows: 14
- segmented rows: 132
- segmented duration: 23.84 minutes
- status: dataset prepared, no neural model trained
- backend selected: no

This is enough for a tightly controlled microtest, not enough to promise a final
voice.

## Investigated Routes

### Piper / Piper1 GPL

Official training flow:

- CSV metadata with `wav|text`
- audio directory
- espeak-ng phonemization
- VITS training/fine-tuning
- ONNX export for CPU-friendly inference

Important official points:

- Fine-tuning from an existing checkpoint is recommended.
- Training examples use GPU-oriented setups.
- Officially reported voice training hardware includes large RAM and NVIDIA GPUs.
- Exported ONNX inference is the strongest local CPU part.

Fit for LYKENOX:

- Direct TTS: yes, after a LYKENOX voice model exists.
- Identity integrated in model: yes, if fine-tuned/trained as a LYKENOX voice.
- Spanish: yes in principle through espeak-ng/Piper Spanish paths.
- No source speaker at inference: yes for a normal exported single-speaker voice.
- Singing: no. Piper is speech TTS, not score-controlled singing.
- CPU training on this laptop: uncertain/high-risk; must be microtested with a
  tiny run before any full training attempt.

Verdict:

Best candidate for local neural speech TTS microtest, but only under strict
limits and only after checking dependency/install footprint. Do not promise full
quality from 23.84 minutes.

### Piper Plus / Zero-Shot Variants

Some newer Piper-family work exposes speaker embedding or reference-audio style
flows.

Fit for LYKENOX:

- Direct TTS: yes.
- Spanish: appears supported in multilingual examples.
- Uses reference/speaker embedding: yes in zero-shot modes.
- Identity integrated without reference prompt: only if converted into a stored
  single-speaker voice or fixed speaker embedding path.

Verdict:

Research-only for now. Do not use zero-shot/reference mode as the main product
because the user wants identity inside the engine, not a reference prompt per
request.

### MaryTTS Unit Selection

Official MaryTTS voicebuilding supports building unit-selection voices with:

- Java
- SoX
- Edinburgh Speech Tools
- audio/text/label files
- packaged voice component

Fit for LYKENOX:

- Direct local TTS: yes.
- Identity integrated in voice package: yes.
- CPU local: yes.
- Neural: no.
- Spanish custom frontend/voicebuilding: would require practical integration
  work and labels.

Verdict:

Fallback if neural CPU training is not viable. It is not neural, but it matches
the "voice is inside the engine" requirement better than voice conversion.

### ESPnet TTS

Official ESPnet TTS recipes include:

- data preparation
- statistics
- TTS training
- decoding/evaluation
- many recipe stages
- CPU-only install mode is possible, but training workflows are research-grade
  and heavy.

Fit for LYKENOX:

- Direct TTS: yes.
- Identity integrated: yes if trained.
- Spanish: possible, not plug-and-play for this project.
- CPU local training: too heavy for this laptop as next step.

Verdict:

Not the next local route. Keep as research reference only.

### Coqui TTS

Already tested locally and rejected.

Evidence:

- Training ran.
- Output was silence/noise.
- Artifacts and environment were removed.

Verdict:

Do not reopen this route unless a completely different proven recipe is selected
first. Current Coqui route is rejected.

## Professional Decision

For the next TTS step, the only route worth a controlled investigation is:

```text
Piper single-speaker Spanish fine-tune/export microtest
```

But with hard gates:

1. No install until exact package/runtime footprint is checked.
2. No full training first.
3. Convert current dataset to Piper metadata format only.
4. Validate espeak-ng Spanish phonemization availability.
5. Validate whether training can start on CPU with a tiny subset.
6. Stop immediately if dependency footprint is too large or CPU step is
   impractical.

If Piper CPU microtest fails, the next local route is not another neural stack.
It is a local integrated voice package/concatenative TTS route, with MaryTTS-like
voicebuilding or a custom lightweight unit-selection engine.

## Proposed Next Microtest

Name:

```text
LYKENOX Piper CPU feasibility microtest
```

Inputs:

- `datasets/lykenox/identity_voice/prepared/speech_segmented/train.segmented.csv`
- `datasets/lykenox/identity_voice/prepared/speech_segmented/val.segmented.csv`
- 10-20 short WAV/text pairs only for the first boot test

Expected output of the microtest:

- no final voice promise
- one dependency report
- one converted Piper metadata folder
- one CPU training-start report
- if possible, one tiny checkpoint or explicit failure reason

Success criteria:

- Training command starts without CUDA-only failure.
- RAM stays within machine limits.
- No source speaker is required for inference.
- Export path can produce a single-speaker model artifact in principle.
- Spanish text path is validated.

Failure criteria:

- Requires CUDA.
- Requires WSL/Linux-only pieces that are impractical here.
- Requires more RAM than available.
- Cannot use Spanish text cleanly.
- Depends on reference voice prompt at inference.

## What Not To Do

- Do not continue expanding multipitch until identity improvement is heard.
- Do not reinstall Coqui.
- Do not install ESPnet as the next step.
- Do not use zero-shot reference-prompt models as the main objective.
- Do not call any WAV success unless intelligibility and identity are evaluated.

## Sources Checked

- Piper training guide:
  `https://github.com/rhasspy/piper/blob/master/TRAINING.md`
- Piper1 GPL training guide:
  `https://github.com/OHF-voice/piper1-gpl/blob/main/docs/TRAINING.md`
- MaryTTS voicebuilding plugin:
  `https://github.com/marytts/gradle-marytts-voicebuilding-plugin`
- MaryTTS:
  `https://github.com/marytts/marytts`
- ESPnet TTS docs:
  `https://espnet.github.io/espnet/recipe/tts1.html`
- Coqui TTS:
  `https://github.com/coqui-ai/TTS`
