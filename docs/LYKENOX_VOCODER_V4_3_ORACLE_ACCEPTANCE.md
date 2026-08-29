# LYKENOX Vocoder v4.3 — full-utterance oracle acceptance

## Persistent training result

The v4.3 persistent run completed numerically under:

```text
v4-3-bounded-resumable-v1
```

Reported result:

```text
status: pass
stop_reason: max_epochs
persistent_training_complete: true
architecture: lykenox_mel_filtered_carrier_v4_3
epochs_completed: 28
global_step: 3304
best_epoch: 27
resumed_invocations: 86

initial_validation:
  reconstruction: 4.413432
  envelope: 4.188759
  spectral_balance: 1.672070
  local_spectral_contrast: 0.337628
  selection_score: 6.993354

best_validation:
  reconstruction: 1.277300
  envelope: 0.686826
  spectral_balance: 0.095586
  local_spectral_contrast: 0.295755
  selection_score: 1.703760

training_improved: true
envelope_improved: true
local_spectral_contrast_improved: true
full_utterance_perceptual_acceptance: false
```

This closes numerical training only. It does not grant product acceptance.

## Current gate

Use complete held-out utterances with:

```text
target mel + target F0 + target voicing
```

and compare v4.3 directly against the trained v4.2 corrective baseline.

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_3_full_utterance_oracle_acceptance
```

The audit writes three groups:

```text
NN_reference.wav
NN_v4_2_oracle.wav
NN_v4_3_oracle.wav
```

under:

```text
models/lykenox_identity/evaluation/vocoder_v4_3_full_utterance_oracle_v1/
```

and writes:

```text
full_utterance_oracle_report.json
```

## Objective diagnostics

For v4.2 and v4.3 the report records:

- reconstruction loss;
- direct log-mel envelope loss and components;
- target-relative broad spectral balance;
- target-relative local spectral contrast;
- RMS/peak;
- spectral centroid and broad-band fractions;
- fraction of energy above 300 Hz;
- RMS relative to the real reference.

It also counts how often v4.3 improves each target-referenced loss over v4.2.

These numbers support diagnosis only. They do not replace listening.

## Listening rule

Listen in this order for each group:

```text
reference -> v4.2 oracle -> v4.3 oracle
```

Judge:

1. whether the residual metallic/insect-like chillido is absent or materially reduced;
2. word/consonant intelligibility;
3. natural formant and mid/high-band detail;
4. apparent level without post-hoc normalization.

V4.3 is accepted for the waveform stage only if the residual v4.2 artifact is materially resolved across all three complete held-out utterances without sacrificing intelligibility or usable level.

Expected pre-listening report state:

```text
status: needs_listening
structural_gate_pass: true
persistent_training_complete: true
full_utterance_perceptual_acceptance: false
next_gate: listen_v4_3_full_utterance_oracle_pairs_and_accept_or_revise_vocoder
```

If the artifact remains, no additional training is authorized by inertia. The remaining failure must first be assigned.

If this oracle gate passes, the next independent debt is predicted-duration calibration before the final reference-free text-to-waveform perceptual gate.
