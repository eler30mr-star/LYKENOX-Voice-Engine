# LYKENOX Vocoder v4.2 — full-utterance oracle acceptance

## Persistent training result

The separate v4.2 persistent run completed successfully under trainer contract:

```text
v4-2-bounded-resumable-v1
```

Reported completion:

```text
status: pass
stop_reason: max_epochs
persistent_training_complete: true
architecture: lykenox_envelope_first_source_filter_v4_2
envelope_loss_version: vocoder-envelope-loss-v1
epochs_completed: 28
global_step: 3304
best_epoch: 25
resumed_invocations: 100

initial_validation:
  reconstruction: 7.340579
  envelope: 6.817812
  spectral_balance: 0.957880
  selection_score: 10.988955

best_validation:
  reconstruction: 1.231856
  envelope: 0.705158
  spectral_balance: 0.139506
  selection_score: 1.619312

training_improved: true
envelope_improved: true
v4_1_checkpoint_mutated: false
reference_audio_required_for_product_inference: false
full_utterance_perceptual_acceptance: false
```

Best checkpoint:

```text
models/lykenox_identity/training/vocoder_source_filter_v4_2/best.pt
```

This closes the persistent numerical training gate. It does **not** grant product or perceptual acceptance.

## Why the next gate is full utterances

V4.1 previously passed short held-out crop checks but later reproduced a persistent periodic metallic/insect-like buzz on complete oracle-conditioned utterances. Therefore v4.2 cannot be accepted from validation loss alone, and 64-frame listening crops are no longer sufficient evidence.

The next gate fixes acoustic uncertainty completely:

```text
target mel
+ target F0
+ target voicing
+ full held-out utterance
-> vocoder waveform
```

If the characteristic buzz remains here, the waveform generator is still the blocker. Predicted duration, predicted mel, and predicted prosody are intentionally excluded from this acceptance decision.

## Acceptance audit

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance
```

The audit uses three fixed held-out validation utterances. For each it writes:

```text
NN_reference.wav
NN_v4_1_oracle.wav
NN_v4_2_oracle.wav
```

Output directory:

```text
models/lykenox_identity/evaluation/vocoder_v4_2_full_utterance_oracle_v1/
```

Report:

```text
full_utterance_oracle_report.json
```

The historical v4.1 reconstruction is comparison-only; it is not restored as a product backend. The real reference is audit-only and remains outside normal product inference.

## Objective checks

For each full utterance the report records:

- exact waveform-length contract;
- finite full-utterance generation;
- waveform RMS and peak;
- spectral centroid;
- broad-band power fractions;
- RMS difference from the real reference in dB;
- multi-resolution reconstruction loss;
- direct log-mel envelope loss;
- envelope level, spectral-slope and temporal-delta components;
- target-relative spectral-balance loss;
- energy fraction above 300 Hz;
- whether v4.2 improves the three target-referenced losses over v4.1.

These measurements support diagnosis but do not replace listening.

## Listening protocol

For each numbered item listen in this order:

```text
reference -> v4.1 oracle -> v4.2 oracle
```

Judge separately:

1. the persistent periodic metallic/insect-like buzz or chillido;
2. word and consonant intelligibility;
3. natural vocal/formant detail rather than low-frequency harmonic dominance;
4. apparent output level without post-hoc normalization.

V4.2 is perceptually accepted only if the characteristic v4.1 buzz is materially resolved across all three complete held-out oracle utterances while intelligibility and level remain usable. A lower numerical loss by itself is insufficient.

Expected audit state before listening:

```text
status: needs_listening
structural_gate_pass: true
persistent_training_complete: true
full_utterance_perceptual_acceptance: false
next_gate: listen_v4_2_full_utterance_oracle_pairs_and_accept_or_revise_vocoder
```

If the buzz remains, do not reconnect predicted acoustic conditioning and do not continue training by inertia. The failure must be assigned before another architectural revision or training run is authorized.

## What remains blocked

Until this gate passes:

- no `/speak` product integration;
- no release/export acceptance;
- no reference-free end-to-end acceptance;
- no additional vocoder training by inertia.

After oracle waveform acceptance, the next independent debt is predicted-duration calibration: the previous held-out audit measured predicted total duration at about 0.649 of teacher duration. Only after that timing issue is corrected should the complete text-to-waveform reference-free perceptual gate be repeated.
