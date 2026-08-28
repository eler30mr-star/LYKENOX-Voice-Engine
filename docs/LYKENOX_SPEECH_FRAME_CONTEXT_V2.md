# LYKENOX Speech — post-regulation frame context v2

## Why v1 persistent acoustics stopped

The first persistent acoustic/prosody run completed correctly, but its held-out audit exposed a structural limitation:

```text
intra_token_mel_delta_l1_target:      0.29826996
intra_token_mel_delta_l1_predicted:   0.0
intra_token_f0_delta_cents_target:    92.1301
intra_token_f0_delta_cents_predicted: 0.0
```

All frame contracts were exact and the supervised losses improved, but the model was mathematically unable to vary mel or F0 inside one token duration. Training longer cannot fix a representation that contains no frame coordinate.

The v1 checkpoint remains a valid record of that experiment but is rejected for end-to-end product inference.

## Root cause

The original acoustic path was:

```text
encoded token
  -> repeat the same token vector for N duration frames
  -> frame-wise mel head
  -> frame-wise F0/voicing head
```

Every repeated frame inside a token had exactly the same vector. Because the mel and prosody heads were pointwise, every interior frame necessarily produced the same output.

A temporal convolution alone is not a complete fix: in a sufficiently long constant token interior, a translation-invariant convolution can still see the same local neighborhood. The model needs an explicit frame coordinate as well as temporal context.

## New architecture

Configuration:

```text
frame_context_version: token-progress-conv-v1
frame_context_layers: 3
frame_context_kernel_size: 5
```

Historical configurations default to:

```text
frame_context_version: none
```

That default is intentional. Old checkpoints that were saved before this field existed reconstruct their original architecture rather than silently loading into a different network.

## Frame coordinates

For every regulated frame the model derives, using tensor-only duration arithmetic:

1. centered progress inside the owning token;
2. log duration of the owning token;
3. normalized progress through the full utterance.

These three features are projected into the acoustic hidden dimension and added to the repeated token representation. This explicitly breaks the within-token symmetry that caused the held-out failure.

## Temporal context

After positional injection, three residual depthwise-separable convolution blocks operate over the regulated frame sequence with dilations:

```text
1, 2, 4
```

The blocks are masked so padded frames do not become acoustic content. They provide local cross-frame and cross-token coarticulation while remaining compact enough for the CPU-only target machine.

The mel and F0/voicing heads consume this contextualized frame representation.

## Contracts preserved

This fix does not change:

- `alignment-v3` teacher durations;
- `mel-v1` targets;
- `speech-pitch-cache-v1` F0/voicing targets;
- the Spanish frontend;
- the accepted v4.1 vocoder conditioning contract;
- the requirement for no reference WAV during product inference.

Teacher durations may still contain zero-duration structural tokens. The new position features derive from the exact regulated durations and therefore preserve that behavior during training.

The old predicted-duration inference rule (`min=1`, `max=80`) remains separate technical debt and must be corrected before unseen-text product inference.

## Bounded architecture smoke — passed

The mandatory CPU smoke passed with:

```text
status: pass
frame_context_version: token-progress-conv-v1
exact_duration_to_frame_contract: true
frame_context_gradient_seen: true
probe_decreased.total: true
probe_decreased.acoustic: true
probe_decreased.duration: true
probe_decreased.f0: true
probe_decreased.voicing: true
intra_token_mel_motion_pass: true
intra_token_f0_motion_pass: true
```

Observed predicted intra-token motion after the smoke:

```text
mel delta L1: 0.02298093   target 0.29320355
F0 delta:    24.1596 c     target 61.0776 c
```

This closes the zero-motion structural defect at the bounded architecture level. It does not yet prove persistent held-out quality.

## Persistent v2 trainer

Trainer contract:

```text
acoustic-frame-context-bounded-resumable-v2
```

Persistent output directory:

```text
models/lykenox_identity/training/acoustic_frame_context_v2/
  last.pt
  best.pt
  training_progress.json
  training_report.json
```

The trainer starts from a new random initialization and never resumes or fine-tunes the rejected v1 checkpoint. Its run identity includes the frame-context version/layers/kernel plus the existing data and loss contracts.

Default real training remains CPU-bounded and resumable:

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

## Mandatory exact-resume gate

Before starting real v2 training, run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_acoustic_frame_context_resume_smoke
```

The smoke compares ten uninterrupted updates against three updates plus an exact seven-update resume. A pass requires exact equality for:

- model state;
- optimizer state;
- Torch RNG state;
- epoch/item/global-step position;
- training metadata/history;
- run configuration;
- persistent data provenance;
- frame-context architecture identity;
- fixed held-out model outputs.

Expected pass gate:

```text
next_gate: start_bounded_resumable_acoustic_frame_context_v2_training
```

Only after this resume-equivalence gate passes should real v2 persistent training start.

## Real v2 training command

After the resume smoke passes:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_acoustic_frame_context_train
```

If the trainer returns `status: incomplete`, rerun the exact same command without deleting checkpoints or altering parameters.

A completed v2 run must still pass a fresh held-out expressivity/prosody audit before predicted-duration inference or unseen-text synthesis.