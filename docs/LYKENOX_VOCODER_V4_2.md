# LYKENOX Vocoder v4.2 — corrective architecture and persistent-training gate

## Why v4.2 exists

The first reference-free end-to-end speech gate was structurally correct but perceptually failed: speech was only partly intelligible and carried a persistent periodic metallic/insect-like buzz. A controlled held-out crossover reproduced the same defect with **target mel + target F0 + target voicing + teacher durations**, removing the acoustic model and predicted-duration path as the cause of this particular artifact.

The subsequent v4.1 harmonic-gain and source-shape ablations established that:

1. reducing harmonic gain mostly removes useful voice and loudness rather than cleanly removing the artifact;
2. disabling aperiodic noise, replacing the learned harmonic envelope with fixed `1/h`, or retaining only 4/2 harmonics does not produce a clean oracle reconstruction;
3. the exact v4.1 baseline remains the clearest generated variant, but is still audibly far from the real held-out waveform.

The failure is therefore assigned to the **neural filtering/reconstruction capacity and source-to-envelope interaction of v4.1**, not to one removable excitation channel and not to a simple output-gain problem.

## What v4.2 preserves

V4.2 remains a LYKENOX-owned source-filter design. It preserves:

- explicit F0 and voicing conditioning;
- deterministic sample-rate conditioning;
- exact `mel_frames * hop_length` waveform length;
- no transposed convolution;
- no learned temporal upsampling;
- no reference audio at product inference;
- no source speaker, voice conversion, cloud backend, external TTS product, or runtime model download;
- fixed 30 Hz linear-phase high-pass output protection.

Architecture identity:

```text
lykenox_envelope_first_source_filter_v4_2
```

The important change is **envelope first, source second**:

```text
mel [frame rate]
  -> 96-channel frame spectral-envelope encoder
  -> deterministic interpolation to sample rate
  -> 64-channel envelope path

F0 + voicing + bounded 8-harmonic excitation + aperiodic source
  -> independent 64-channel temporal source stem
  -> mel-controlled source gate

(envelope path + gated source path)
  -> 8 gated residual/skip filter blocks
     dilations: 1,2,4,8,16,32,64,128
  -> waveform projection
  -> fixed 30 Hz HPF
  -> tanh waveform
```

The source acts as a phase/pitch excitation cue inside a wider nonlinear filter rather than being granted raw waveform authority by simple concatenation.

Default candidate dimensions remain deliberately bounded for CPU use:

- hidden channels: 64;
- conditioning channels: 96;
- harmonics: 8;
- parameters: 399,049;
- sample-rate receptive field: 1,035 samples / 43.125 ms.

## Envelope-targeted training objective

V4.2 keeps waveform/multi-resolution-STFT reconstruction and target-relative broad-band spectral balance, and adds a direct differentiable **log-mel envelope reconstruction objective**:

```text
waveform prediction
  -> log-mel
  -> target-relative comparison
       + log-mel level L1
       + mel-band spectral-slope L1
       + temporal-delta L1
```

This objective directly targets the formant/consonant/envelope reconstruction deficit exposed by v4.1. It is training-only and adds no runtime dependency.

## Architecture smoke — PASSED

The bounded real-data architecture gate completed successfully:

```text
status: pass
architecture: lykenox_envelope_first_source_filter_v4_2
parameters: 399049
parameter_budget_pass: true
receptive_field_ms: 43.125
receptive_field_pass: true
exact_length_contract: true
structural_finite: true
gradients_finite: true
total_decreased: true
envelope_decreased: true
benchmark_rtf: 0.5508
persistent_training_started: false
v4_1_checkpoint_mutated: false
```

The local CPU RTF remains below 1.0 in the bounded inference benchmark, so the architecture is feasible enough to proceed to persistent training.

## Persistent trainer contract

The separate trainer lives in:

```text
lykenox_voice_engine/training/speech_vocoder_v4_2_artifact.py
lykenox_voice_engine/training/speech_vocoder_v4_2_train.py
```

Trainer contract:

```text
v4-2-bounded-resumable-v1
```

Artifact directory:

```text
models/lykenox_identity/training/vocoder_source_filter_v4_2/
```

V4.1 is never loaded or overwritten by this trainer.

The stable training identity includes dataset manifest hashes, architecture identity, mel feature-cache contract, pitch-target version, segment schedule, envelope-loss version, source-balance version and all optimization hyperparameters. Execution-only controls such as the per-invocation wall-clock budget are excluded from the training identity so the same experiment can resume across bounded CPU invocations.

Default persistent objective:

```text
reconstruction
+ 0.50 * log_mel_envelope
+ 0.25 * target_relative_spectral_balance
+ after warmup:
    0.03 * adversarial
    0.50 * feature_matching
```

Checkpoint selection uses only the stable target-referenced validation objective:

```text
validation_reconstruction
+ 0.50 * validation_envelope
+ 0.25 * validation_spectral_balance
```

Adversarial scores are intentionally excluded from checkpoint selection.

## Exact-resume gate — PASSED

The resume smoke compared four deterministic updates executed directly against `2 + checkpoint/reload + 2` and passed bit-exact equality for all persistent state:

```text
status: pass
global_step_exact: true
epoch_exact: true
next_item_offset_exact: true
generator_state_exact: true
discriminator_state_exact: true
generator_optimizer_exact: true
discriminator_optimizer_exact: true
torch_rng_state_exact: true
run_config_exact: true
v4_1_checkpoint_unchanged: true
temporary_artifacts_removed: true
persistent_v4_2_training_started: false
next_gate: start_bounded_resumable_v4_2_persistent_training
```

This closes the architecture and resumability gates. **Persistent v4.2 training is now authorized.**

## Authorized persistent run

Run the default training identity and do not vary its hyperparameters between invocations:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_2_train
```

Default persistent configuration:

```text
segment_mel_frames: 64
train_items: 118
val_items: 14
max_epochs: 28
warmup_epochs: 4
patience: 6
generator_lr: 2e-4
discriminator_lr: 1e-4
envelope_weight: 0.50
balance_weight: 0.25
adversarial_weight: 0.03
feature_matching_weight: 0.50
gradient_clip_norm: 5.0
checkpoint_every_updates: 8
time_budget_seconds: 70
```

The run writes:

```text
models/lykenox_identity/training/vocoder_source_filter_v4_2/
  last.pt
  best.pt
  training_progress.json
  training_report.json
```

Each invocation stops conservatively before the short command-runner ceiling and writes `last.pt` atomically only between complete generator/discriminator updates. If the result is:

```text
status: incomplete
next_gate: rerun_same_command_to_resume
```

rerun the **identical command**. Do not change training arguments mid-run and do not delete or replace `last.pt` to force a different experiment.

The run terminates by bounded `early_stopping` or `max_epochs`. Training is numerically successful only when it writes:

```text
persistent_training_complete: true
training_improved: true
envelope_improved: true
status: pass
```

A completed numerical run does **not** imply perceptual acceptance.

## Mandatory post-training acceptance

After persistent training completes, the next gate is:

```text
run_v4_2_full_utterance_oracle_acceptance
```

Acceptance must use **full held-out utterances** with target mel + target F0 + target voicing. Short 64-frame crops are not sufficient. The periodic metallic/insect-like buzz must be absent or materially resolved before v4.2 can be connected back to reference-free acoustic predictions.

No `/speak` integration, release export, or product-runtime acceptance is authorized before that full-utterance oracle listening gate passes.

## Separate duration debt

The held-out failure-isolation audit measured predicted acoustic durations at only about 0.649 of teacher total duration. That remains a real, separate inference problem. It is intentionally not mixed into this vocoder revision because the metallic periodic artifact was reproduced with teacher durations and fully oracle acoustic conditioning.

The order remains:

```text
clean oracle waveform stage
  -> correct predicted-duration calibration
  -> repeat full reference-free text-to-waveform perceptual gate
```
