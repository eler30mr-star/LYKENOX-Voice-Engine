# LYKENOX Speech — persistent acoustic prosody training

## Gate status before this stage

The following prerequisites are already closed:

- `alignment-v3` teacher durations are clean;
- `mel-v1` acoustic targets are cached;
- `speech-pitch-cache-v1` contains exact frame-aligned F0/voicing targets for all 132 utterances;
- the bounded acoustic prosody-head smoke passed on CPU, with total, mel, duration, F0 and voicing losses all decreasing.

This stage does not change the accepted v4.1 vocoder architecture.

## Training objective

The persistent acoustic model learns jointly:

```text
text tokens
  -> duration prediction
  -> teacher-duration frame regulation during training
  -> mel prediction
  -> F0 prediction
  -> voicing prediction
```

The losses are:

```text
mel L1
+ 0.10 * log-duration Smooth-L1
+ 0.25 * voiced-frame log-F0 Smooth-L1
+ 0.25 * voicing BCEWithLogits
```

F0 targets are read only from the completed persistent pitch cache. Waveform pitch extraction is not run inside acoustic training.

## Exactly resumable contract

Trainer contract:

```text
acoustic-prosody-bounded-resumable-v1
```

Default real training:

```text
batch size:               2
max epochs:               36
early-stop patience:      6
learning rate:            2e-4
weight decay:             1e-4
gradient clip:            5.0
checkpoint every updates: 16
wall-clock budget:        70 s
checkpoint reserve:       8 s
```

`last.pt` stores the exact current epoch/item offset, model state, optimizer state, torch RNG state, partial-epoch metric accumulator, validation history, best state metadata, run configuration and hashed data provenance.

Execution-only limits such as wall-clock budget are intentionally excluded from the run identity, so rerunning the same command continues the same experiment.

Persistent output:

```text
models/lykenox_identity/training/acoustic_prosody_v1/
  last.pt
  best.pt
  training_progress.json
  training_report.json
```

## Mandatory resume-equivalence gate

Before starting real persistent training, run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_acoustic_prosody_resume_smoke
```

The smoke compares 10 uninterrupted updates against 3 updates plus an exact 7-update resume. A pass requires exact equality for:

- model state;
- optimizer state;
- torch RNG state;
- epoch/item/global-step position;
- partial training metadata/history;
- run configuration;
- persistent data provenance;
- fixed held-out model outputs.

Expected next gate:

```text
start_bounded_resumable_acoustic_prosody_training
```

## Real training command

Only after the resume smoke passes:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_acoustic_prosody_train
```

If it returns:

```text
status: incomplete
next_gate: rerun_same_command_to_resume
```

rerun the exact same command. Do not delete checkpoints or alter parameters mid-run.

## Completion meaning

A completed `status: pass` closes persistent supervised acoustic training only. It does not yet mean product inference is complete.

The next end-to-end stage must still:

1. audit held-out predicted mel/F0/voicing quality from `best.pt`;
2. fix predicted-duration inference semantics so structural tokens can remain zero-duration and real duration ranges are not clipped by the old fixed `80` frame ceiling;
3. synthesize unseen text using predicted rather than waveform-derived F0/voicing;
4. evaluate intelligibility, identity and v4.1 waveform quality;
5. only then proceed toward export/runtime integration.
