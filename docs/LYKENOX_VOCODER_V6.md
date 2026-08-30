# LYKENOX Vocoder v6 — direct conditional waveform decoder

## Why v6 exists

V5 removed the explicit sinusoidal carrier but still forced every audible voiced sample through a stochastic glottal-pulse/noise source. Full-utterance oracle listening judged v5 worse: the speech was described as gangoso/nasal, still not clean, and subjectively too weak.

The v5 oracle files also show that the weak perceived level is not explained by global RMS alone. Across the three supplied reference/v5 pairs, global RMS stayed close to the reference (roughly within 0.6 dB), while the spectral centroid and the 1–3 kHz / 3–8 kHz presence bands fell materially. This is consistent with a muffled/nasal signal that can measure similar RMS while sounding quieter and less intelligible.

Therefore v6 treats two defects separately:

1. **noise/source coloration** — remove the explicit voiced source path entirely;
2. **weak/nasal perceived level** — train target-relative short-time level and target-relative spectral presence, rather than relying on a post-hoc gain boost.

No additional v5 training is authorized.

## Architecture identity

```text
lykenox_direct_conditional_waveform_v6
```

Source family:

```text
direct_conditional_waveform_decoder
```

Hard constraints:

```text
explicit_source: false
explicit_sinusoidal_carrier: false
deterministic_harmonics: 0
voiced_noise_source: false
raw_source_bypass: false
conditioning_only_waveform: true
```

Unlike v4.x/v5, v6 deliberately does **not** require `zero excitation => zero waveform`. There is no product-time excitation input to zero. Mel, F0 and voicing are conditioning variables of the waveform decoder itself.

## Signal path

```text
mel + log-F0 + voicing
  -> frame encoder / context
  -> deterministic progressive resize-refine stages (4 x 4 x 4 x 4 = hop 256)
  -> sample-rate conditioning
       - glottal phase aperture feature
       - centered phase feature
       - voicing
       - log-F0
       - unvoiced-only deterministic detail feature
  -> learned sample-rate residual decoder
  -> learned output-level degree of freedom
  -> bounded waveform
```

Important: phase and the unvoiced detail signal are **conditioning features**, not additive waveform sources. There is no raw bypass from them to the output.

The unvoiced detail feature exists only to preserve fricative/consonant capacity; v6 has no broadband voiced-noise source.

## Why this differs from old direct prototypes

Earlier direct upsampling prototypes exposed frame-rate carriers or sub-bass collapse. V6 adds two protections that those probes did not jointly have:

- no phase-indexed/transposed-convolution upsampler: all four expansion stages use deterministic linear resize followed by learned refinement;
- voiced phase information is injected at sample rate as conditioning, so high-frequency periodic detail does not have to be reconstructed from a smooth mel interpolation alone.

The sample-rate residual decoder has a lower-bound receptive field of about 64 ms before counting the additional frame/upsampling context.

## Level and clarity objective

New loss identity:

```text
vocoder-level-presence-v1
```

Implemented in:

```text
lykenox_voice_engine/training/speech_vocoder_level_presence_loss.py
```

It contains two target-relative components.

### Level

```text
global log-RMS match
+ short-time log-RMS match
```

This gives the model an explicit training signal for useful amplitude and dynamics. It does not normalize or boost audio at product inference.

### Spectral presence

The predicted/target energy distribution is compared across:

```text
80–300 Hz
300–1000 Hz
1–3 kHz
3–8 kHz
```

The 1–8 kHz bands receive greater weight because the rejected v5 files lost formant/consonant presence there. The target recording remains the authority; this is not a fixed EQ curve.

A diagnostic `presence_1k_8k_error_db` explicitly measures mismatch in the clarity/presence region.

## Current gate — architecture/optimization smoke only

Implemented:

```text
lykenox_voice_engine/training/speech_vocoder_v6_architecture_smoke.py
```

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v6_architecture_smoke
```

The smoke uses one real training segment and must prove:

```text
status: pass
architecture: lykenox_direct_conditional_waveform_v6
source_family: direct_conditional_waveform_decoder
explicit_source: false
explicit_sinusoidal_carrier: false
deterministic_harmonics: 0
voiced_noise_source: false
raw_source_bypass: false
conditioning_only_waveform: true
persistent_training_started: false
historical_checkpoints_mutated: false
exact_length_contract: true
structural_finite: true
direct_waveform_contract: true
mel_changes_waveform: true
f0_changes_waveform: true
voicing_changes_waveform: true
gradients_finite: true
total_decreased: true
envelope_decreased: true
level_decreased: true
presence_decreased: true
presence_error_decreased: true
parameter_budget_pass: true
receptive_field_pass: true
cpu_candidate_pass: true
next_gate: build_bounded_resumable_v6_training_candidate
```

The smoke also reports predicted/target RMS and all four band fractions before/after the short optimization probe, so a structural pass cannot hide another low-frequency collapse.

## Gate order

```text
v5 full-utterance oracle listening        [REJECTED: noisy/gangoso/weak]
  -> v6 direct waveform architecture smoke [CURRENT]
  -> exact-resume v6 trainer gate
  -> bounded persistent v6 training
  -> full-utterance oracle listening acceptance
  -> predicted-duration calibration
  -> reference-free text-to-waveform perceptual gate
```

Persistent v6 training is **not authorized** until the architecture smoke passes. No `/speak`, export or product acceptance is authorized until a vocoder produces clean full-utterance oracle speech with useful level and without the radio/noise/nasal artifacts.
