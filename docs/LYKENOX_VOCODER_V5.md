# LYKENOX Vocoder v5 — non-sinusoidal corrective candidate

## Why v5 exists

The v4.x family isolated a persistent perceptual defect described as a radio-mistuned / metallic interference attached to the voice. Full-utterance oracle listening rejected v4.3 and v4.4 even when several target-referenced spectral metrics improved.

The decisive v4.4 path-attribution ablation showed that the explicit periodic branch preserved the defect much more strongly than the aperiodic branch. Reducing the harmonic bank from 24 to 8 helped somewhat but did not solve the problem. Therefore v5 is an architectural break rather than another v4.x tuning pass.

No further v4.x training is authorized.

## Architecture identity

```text
lykenox_stochastic_glottal_filter_v5
```

Source family:

```text
stochastic_glottal_pulse_noise
```

Explicit sinusoidal carrier:

```text
false
```

Deterministic harmonic bank:

```text
0 harmonics
```

## Corrective principle

F0 still controls timing and voicing, but it no longer creates an audible bank of sinusoidal harmonics.

Instead:

```text
F0 + voicing
  -> sample-rate phase / glottal timing controls

broadband deterministic noise
  -> stochastic glottal pulse bursts during voiced regions
  -> low voiced broadband floor
  -> independent unvoiced broadband component

mel + phase/F0 controls
  -> dynamic bias-free filter selection / gating

excitation-dependent hidden state
  -> bias-free residual filtering
  -> waveform projection
  -> fixed 30 Hz high-pass FIR
  -> waveform
```

The periodic information is used as timing/conditioning, not as a directly audible sinusoidal carrier.

## Structural invariant

Mel, F0 and phase controls are not allowed to create waveform without excitation:

```text
excitation_scale = 0
mel != 0
F0 != 0
voiced != 0
=> waveform == 0 exactly
```

This preserves the useful anti-shortcut property of the later v4.x models while removing the path that was perceptually implicated.

## CPU-bounded candidate

The initial v5 candidate intentionally reduces model cost relative to v4.4:

```text
hidden channels: 48
mel conditioning channels: 64
filter bases per block: 2
dilations: 1,2,4,8,16,32,64,128
```

Persistent training is **not authorized** yet.

## Current gate — architecture smoke only

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v5_architecture_smoke
```

The smoke must establish at minimum:

```text
status: pass
architecture: lykenox_stochastic_glottal_filter_v5
source_family: stochastic_glottal_pulse_noise
explicit_sinusoidal_carrier: false
deterministic_harmonics: 0
persistent_training_started: false
historical_checkpoints_mutated: false
exact_length_contract: true
structural_finite: true
no_conditioning_only_waveform: true
zero_excitation_max_abs: 0.0
no_sinusoidal_carrier: true
f0_changes_waveform: true
gradients_finite: true
total_decreased: true
envelope_decreased: true
parameter_budget_pass: true
receptive_field_pass: true
cpu_candidate_pass: true
next_gate: build_bounded_resumable_v5_training_candidate
```

This smoke uses a short real-data optimization probe and reports local CPU update time plus inference RTF. A pass authorizes only construction of an exact-resume trainer. It does **not** authorize persistent v5 training by itself.

## Remaining order

```text
v4.4 path attribution                 [CLOSED: periodic path implicated]
  -> v5 non-sinusoidal architecture smoke   [CURRENT]
  -> exact-resume v5 trainer gate
  -> bounded persistent v5 training
  -> full-utterance oracle listening acceptance
  -> predicted-duration calibration
  -> reference-free text-to-waveform perceptual gate
```

No `/speak`, export or product acceptance is authorized until a vocoder passes full-utterance perceptual acceptance without the radio-mistuned / metallic interference.
