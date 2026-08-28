# LYKENOX Speech — persistent acoustic prosody training

## Gate status before this stage

The following prerequisites are already closed:

- `alignment-v3` teacher durations are clean;
- `mel-v1` acoustic targets are cached;
- `speech-pitch-cache-v1` contains exact frame-aligned F0/voicing targets for all 132 utterances;
- the bounded acoustic prosody-head smoke passed on CPU, with total, mel, duration, F0 and voicing losses all decreasing;
- exact resume equivalence passed: model state, optimizer state, Torch RNG, position, metadata, run config, provenance and fixed probe output were all byte/numerically exact after split resume.

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

## Persistent run result

The real persistent run completed with:

```text
status:       pass
stop_reason:  early_stopping
epochs:       13
global_step:  767
best_epoch:   7
```

Held-out validation improved from initialization to the selected epoch-7 checkpoint:

```text
initial total:     1.461489
best total:        0.775831

initial acoustic:  1.153041
best acoustic:     0.608616

initial duration:  0.950050
best duration:     0.136093

initial F0:        0.139214
best F0:           0.086208

initial voicing:   0.714557
best voicing:      0.528212
```

Best checkpoint:

```text
models/lykenox_identity/training/acoustic_prosody_v1/best.pt
```

The six completed epochs after epoch 7 did not improve the validation selection metric enough to reset patience, so early stopping closed the run. Do not continue training this checkpoint merely to reach the configured 36-epoch ceiling.

## Mandatory held-out expressivity audit

Persistent supervised optimization passing is not yet an end-to-end inference gate.

The current bootstrap length regulator repeats one encoded token vector for every frame assigned to that token. The mel decoder and F0/voicing heads currently operate on those repeated frame vectors without a post-regulation temporal-context block. Therefore decreasing mel/F0 losses alone does not prove that the model can reproduce real frame-to-frame motion inside a phoneme/token span.

Before changing duration inference or synthesizing unseen text, run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_acoustic_prosody_audit
```

The audit uses validation data and teacher durations intentionally, so duration prediction is held out of the equation. It measures:

- exact mel/F0/voicing frame contracts;
- held-out mel L1;
- F0 correlation and error in cents on target-voiced frames;
- voicing precision/recall/F1/balanced accuracy;
- target versus predicted mel frame-to-frame motion inside token spans;
- target versus predicted F0 motion inside token spans.

If real held-out targets have intra-token motion while predictions are effectively constant, the audit returns:

```text
status: needs_review
next_gate: fix_post_regulation_frame_context_before_end_to_end
```

That result is architectural. Training longer is not the remedy.

If frame expressivity passes, the next gate is:

```text
fix_predicted_duration_semantics_before_end_to_end
```

## Completion meaning

The completed persistent run closes persistent supervised acoustic training for the current architecture only. It does not yet mean product inference is complete.

The next end-to-end stage must still:

1. close the held-out acoustic/prosody expressivity audit from `best.pt`;
2. if required, add post-regulation frame context and retrain only after its bounded smoke passes;
3. fix predicted-duration inference semantics so structural tokens can remain zero-duration and real duration ranges are not clipped by the old fixed `80` frame ceiling;
4. synthesize unseen text using predicted rather than waveform-derived F0/voicing;
5. evaluate intelligibility, identity and v4.1 waveform quality;
6. only then proceed toward export/runtime integration.
