# LYKENOX Speech — persistent frame-context v2 held-out audit

## Persistent training result

The persistent acoustic v2 run completed successfully and stopped by early stopping rather than by execution budget.

```text
status: pass
stop_reason: early_stopping
epochs_completed: 34
best_epoch: 28
global_step: 2006
trainer_contract_version: acoustic-frame-context-bounded-resumable-v2
frame_context_version: token-progress-conv-v1
```

Validation improved from:

```text
total:    1.449685 -> 0.648488
acoustic: 1.143815 -> 0.504008
duration: 0.950050 -> 0.123692
F0:      0.139214 -> 0.060683
voicing: 0.704246 -> 0.467759
```

The selected checkpoint is:

```text
models/lykenox_identity/training/acoustic_frame_context_v2/best.pt
```

No additional training should be performed merely because the maximum epoch count was not reached. The selected checkpoint is epoch 28 and later epochs did not improve the selection objective enough to reset early stopping.

## Why another held-out audit is mandatory

The bounded frame-context smoke proved that `token-progress-conv-v1` can represent non-zero intra-token mel and F0 motion. Persistent training must now prove that this property survives on the validation split in the selected `best.pt`.

Teacher durations remain intentional for this audit. Predicted-duration inference semantics are still a separate unresolved gate; mixing them into this measurement would make it impossible to distinguish acoustic/prosody quality from duration errors.

## Audit command

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_acoustic_frame_context_audit
```

Audit version:

```text
acoustic-frame-context-heldout-audit-v2
```

The audit requires exact identity for:

```text
trainer_contract_version: acoustic-frame-context-bounded-resumable-v2
frame_context_version: token-progress-conv-v1
frame_context_layers: 3
frame_context_kernel_size: 5
```

It then measures the same held-out contract used to expose the rejected v1 model:

- mel L1;
- F0 log correlation and cents errors;
- voicing precision/recall/F1/balanced accuracy;
- exact frame-grid contract on every validation item;
- predicted versus target intra-token mel motion;
- predicted versus target intra-token F0 motion.

If the old v1 held-out report is present locally, the v2 report also records a diagnostic side-by-side comparison. That comparison is informative but not a hard monotonic gate: the mandatory acceptance condition is exact v2 identity, exact held-out frame contracts, and non-zero held-out frame expressivity.

## Pass condition

A pass requires:

```text
status: pass
architecture_identity_exact: true
all_frame_contracts_exact: true
frame_expressivity_pass: true
predicted_has_intra_token_mel_motion: true
predicted_has_intra_token_f0_motion: true
next_gate: fix_predicted_duration_semantics_before_end_to_end
```

A pass does **not** yet authorize unseen-text synthesis. It closes the persistent acoustic frame-context defect and moves the project to the duration-inference semantics gate.

The old inference rule that clamps every valid token to at least one frame and caps durations at 80 frames must be corrected before text-only end-to-end synthesis.