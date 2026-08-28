# LYKENOX Vocoder v4.2 — corrective architecture gate

## Why v4.2 exists

The first reference-free end-to-end speech gate was structurally correct but perceptually failed: speech was only partly intelligible and carried a persistent periodic metallic/insect-like buzz.  A controlled held-out crossover then reproduced the same defect with **target mel + target F0 + target voicing + teacher durations**.  That removes the acoustic model and predicted duration path as the cause of this particular artifact.

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

V4.1 interpolated raw 80-bin mel features to sample rate, concatenated them with eight harmonic channels plus voicing/F0/noise, immediately compressed that entire representation to only 32 hidden channels, and then passed it through five lightweight depthwise-separable residual blocks.  The oracle listening result shows that this filter can carry pitch and speech rhythm but does not sufficiently reshape the synthetic excitation into a natural vocal waveform.

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

The source is no longer granted raw direct authority by simple concatenation.  It acts as a phase/pitch excitation cue inside a wider nonlinear filter whose conditioning remains explicitly tied to the mel spectral envelope.

Default candidate dimensions remain deliberately bounded for the target CPU:

- hidden channels: 64;
- conditioning channels: 96;
- harmonics: 8;
- sample-rate receptive field: about 43 ms;
- hard architecture-smoke parameter budget: 600k parameters.

## New training objective

V4.1 already used waveform/multi-resolution-STFT reconstruction and target-relative broad-band spectral balance.  V4.2 adds a direct differentiable **log-mel envelope reconstruction objective**:

```text
waveform prediction
  -> log-mel
  -> compare against target waveform log-mel
       + level L1
       + mel-band spectral-slope L1
       + temporal-delta L1
```

This objective exists specifically because the current failure is perceptual envelope/formant/consonant reconstruction, not merely aggregate waveform energy.  It is training-only and creates no new runtime dependency.

## Professional gate order

Persistent v4.2 training is **not authorized yet**.

The required order is:

1. run the bounded real-data v4.2 architecture smoke;
2. require exact length, finite forward/backward, finite gradients, bounded parameter count, >=40 ms receptive field, decreasing composite loss, and decreasing direct mel-envelope error;
3. inspect local CPU step time and inference RTF;
4. only if that passes, build a new exactly-resumable persistent v4.2 trainer with separate artifact identity;
5. persistent selection must include the new envelope objective and must not overwrite v4.1 artifacts;
6. final acceptance must include **full held-out utterances**, not only ~64-frame listening crops;
7. oracle full-utterance speech must lose the periodic metallic/buzz artifact before reference-free acoustic predictions are reconnected.

Run the current bounded gate:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_2_architecture_smoke
```

A pass should report:

```text
status: pass
architecture: lykenox_envelope_first_source_filter_v4_2
persistent_training_started: false
v4_1_checkpoint_mutated: false
exact_length_contract: true
structural_finite: true
gradients_finite: true
total_decreased: true
envelope_decreased: true
parameter_budget_pass: true
receptive_field_pass: true
next_gate: build_bounded_resumable_v4_2_training_candidate
```

## Separate duration debt

The held-out failure-isolation audit also measured predicted acoustic durations at only about 0.649 of teacher total duration.  That remains a real, separate inference problem.  It is intentionally not mixed into this vocoder revision because the periodic metallic artifact has already been reproduced with teacher durations and fully oracle acoustic conditioning.

The order remains: first make the oracle waveform stage clean, then correct predicted duration calibration, then repeat the fully reference-free text-to-waveform perceptual gate.
