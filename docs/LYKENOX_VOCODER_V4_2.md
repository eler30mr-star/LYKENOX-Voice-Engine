# LYKENOX Vocoder v4.2 — corrective architecture gate

## Why v4.2 exists

The first reference-free end-to-end speech gate was structurally correct but perceptually failed: speech was only partly intelligible and carried a persistent periodic metallic/insect-like buzz. A controlled held-out crossover then reproduced the same defect with **target mel + target F0 + target voicing + teacher durations**. That removes the acoustic model and predicted duration path as the cause of this particular artifact.

The subsequent v4.1 harmonic-gain and source-shape ablations established three useful facts:

1. reducing harmonic gain does not cleanly remove the defect; it mostly removes useful voice and loudness;
2. disabling aperiodic noise, replacing the learned harmonic envelope with fixed `1/h`, or retaining only 4/2 harmonics does not produce a clean oracle reconstruction;
3. the exact v4.1 baseline remains the clearest generated variant, but is still audibly far from the real held-out waveform.

Therefore the current failure is assigned to the **neural filtering/reconstruction capacity and source-to-envelope interaction of v4.1**, not to one removable excitation channel and not to a simple output-gain problem.

## What is preserved

V4.2 is a corrective revision of the accepted source-filter direction, not a return to rejected architectures.

It preserves:

- LYKENOX-owned implementation;
- explicit F0 and voicing contract;
- deterministic sample-rate conditioning;
- exact `mel_frames * hop_length` waveform length;
- no transposed convolution;
- no learned temporal upsampling;
- no reference audio at product inference;
- no source speaker, voice conversion, cloud backend, external TTS product, or runtime model download;
- fixed 30 Hz linear-phase high-pass output protection.

## Architecture change

Architecture identity:

```text
lykenox_envelope_first_source_filter_v4_2
```

The important change is **envelope first, source second**.

V4.1 interpolated raw 80-bin mel features to sample rate, concatenated them with eight harmonic channels plus voicing/F0/noise, immediately compressed that entire representation to only 32 hidden channels, and then passed it through five lightweight depthwise-separable residual blocks. The oracle listening result shows that this filter can carry pitch and speech rhythm but does not sufficiently reshape the synthetic excitation into a natural vocal waveform.

V4.2 instead uses:

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

The source is no longer granted raw direct authority by simple concatenation. It acts as a phase/pitch excitation cue inside a wider nonlinear filter whose conditioning remains explicitly tied to the mel spectral envelope.

Default candidate dimensions remain deliberately bounded for the target CPU:

- hidden channels: 64;
- conditioning channels: 96;
- harmonics: 8;
- sample-rate receptive field: about 43 ms;
- hard architecture-smoke parameter budget: 600k parameters.

## New training objective

V4.1 already used waveform/multi-resolution-STFT reconstruction and target-relative broad-band spectral balance. V4.2 adds a direct differentiable **log-mel envelope reconstruction objective**:

```text
waveform prediction
  -> log-mel
  -> compare against target waveform log-mel
       + level L1
       + mel-band spectral-slope L1
       + temporal-delta L1
```

This objective exists specifically because the current failure is perceptual envelope/formant/consonant reconstruction, not merely aggregate waveform energy. It is training-only and creates no new runtime dependency.

## Architecture smoke result

The real-data CPU architecture gate passed locally:

```text
status: pass
parameters: 399049
receptive_field_ms: 43.125
exact_length_contract: true
structural_finite: true
gradients_finite: true
total_decreased: true
envelope_decreased: true
benchmark_rtf: 0.5508
persistent_training_started: false
v4_1_checkpoint_mutated: false
```

This authorizes construction of the persistent trainer; it does not itself authorize starting the long run until exact resume semantics are validated.

## Persistent trainer contract

The separate v4.2 trainer is now implemented as:

```text
lykenox_voice_engine/training/speech_vocoder_v4_2_artifact.py
lykenox_voice_engine/training/speech_vocoder_v4_2_train.py
```

Artifact directory:

```text
models/lykenox_identity/training/vocoder_source_filter_v4_2/
```

V4.1 is never loaded or overwritten by this trainer.

The stable training identity includes dataset manifest hashes, architecture identity, mel feature-cache contract, pitch-target version, segment schedule, envelope-loss version, source-balance version and all optimization hyperparameters. Execution-only controls such as per-invocation wall-clock budget are deliberately excluded from the training identity so the identical experiment can resume across short CPU invocations.

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

The default run is bounded to 70 seconds per invocation, checkpoints every 8 complete updates, and writes `last.pt` only between complete generator/discriminator updates. `last.pt` contains generator, discriminator, both optimizer states, exact epoch/item offset and torch RNG state.

## Current gate: exact resume smoke

**Do not start the persistent run yet.** First validate that interruption/resume is bit-exact.

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_2_resume_smoke
```

The smoke runs four temporary updates once directly and once as `2 + resume + 2`, with adversarial training active. It must report:

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
persistent_v4_2_training_started: false
next_gate: start_bounded_resumable_v4_2_persistent_training
```

Only after that gate passes is the persistent run authorized. The production command will then be:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_2_train
```

If it returns `status: incomplete`, rerun the **identical command**. Do not change hyperparameters mid-run and do not delete/replace `last.pt` to force a different experiment.

## Acceptance after persistent training

A successful numerical training report is not final acceptance. The next mandatory gate is full held-out oracle utterances using target mel + target F0/voicing. Short ~64-frame crops are insufficient because they failed to expose the v4.1 product defect.

V4.2 is accepted for the waveform stage only if the full-utterance oracle outputs lose the persistent periodic metallic/insect-like buzz while preserving intelligibility, useful mid/high-band speech detail, and sane level. Only after that do we return to predicted-duration calibration and fully reference-free text-to-waveform synthesis.

## Separate duration debt

The held-out failure-isolation audit also measured predicted acoustic durations at only about 0.649 of teacher total duration. That remains a real, separate inference problem. It is intentionally not mixed into this vocoder revision because the periodic metallic artifact has already been reproduced with teacher durations and fully oracle acoustic conditioning.

The order remains: first make the oracle waveform stage clean, then correct predicted duration calibration, then repeat the fully reference-free text-to-waveform perceptual gate.
