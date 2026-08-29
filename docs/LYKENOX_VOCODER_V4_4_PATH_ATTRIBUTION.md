# LYKENOX Vocoder v4.4 — path attribution conclusion

## Gate status

The trained v4.4 full-utterance oracle gate was perceptually rejected because the radio-mistuned / metallic interference remained. A bounded post-training path attribution audit was then executed with no training and no checkpoint mutation.

Structural result:

```text
status: needs_listening
architecture: lykenox_dynamic_filter_hybrid_v4_4
baseline_reproduction_exact: true
baseline_reproduction_max_delta: 0.0
checkpoint_unchanged: true
persistent_training_restarted: false
```

The rendered variants were:

```text
baseline.wav
periodic_only.wav
aperiodic_only.wav
no_block_aperiodic.wav
equal_filter_selector.wav
```

## Listening conclusion

Listening closes the attribution gate with the following result:

```text
periodic path carries the dominant pitched / radio-like interference: YES
aperiodic-only path reproduces the same pitched interference: NO
aperiodic path supplies useful broadband/consonant detail: YES
removing repeated block aperiodic injection solves the artifact: NO
equalizing the dynamic filter selector solves the artifact: NO
dynamic filter selector is the sole cause: NO
```

`periodic_only.wav` retains the strongest coherent periodic/buzzy character while becoming darker and losing high-frequency detail. `aperiodic_only.wav` loses the voiced body and becomes weak/noisy, but it does not retain the same strongly pitched carrier texture. `no_block_aperiodic.wav` remains strongly periodic and therefore does not implicate repeated aperiodic injection as the principal cause. `equal_filter_selector.wav` becomes weak/noisy rather than clean, so the learned time-varying filter selector is not the sole cause either.

Simple signal diagnostics agree with the listening result. On the audited utterance, normalized frame periodicity increased from about `0.439` in the baseline to about `0.553` in `periodic_only`, while `aperiodic_only` dropped to about `0.214`. The periodic-only output also concentrated about `81%` of spectral energy in 80–300 Hz and essentially removed >3 kHz energy, whereas the aperiodic-only path retained much more mid/high-frequency energy but lacked useful voiced body.

## Root-cause assignment

The remaining failure is assigned primarily to the **explicit deterministic sinusoidal voiced excitation assumption** used by the v4.x family. The periodic branch is required for pitch/body, but its coherent harmonic carrier remains perceptually exposed after learned filtering. The aperiodic branch is required for consonants and broadband detail, so simply suppressing it is not a valid correction. Likewise, freezing the dynamic filter selector does not clean the voice.

Therefore the failure is not a scalar gain problem, not an aperiodic-noise-level problem, and not a single filter-selector problem. It is a structural tradeoff in the current excitation representation:

```text
explicit coherent harmonic carrier
  -> useful pitch/body
  -> persistent synthetic radio/buzz character

aperiodic excitation
  -> useful broadband/consonant detail
  -> cannot independently provide voiced body
```

## Architecture decision

No further v4.1–v4.4 training or carrier tuning is authorized. Do not continue trying harmonic counts, source gains, voiced-noise floors, or post-hoc normalization.

The next vocoder candidate must break from the explicit sum-of-sinusoids carrier. F0 may remain a conditioning/control signal, but it must not directly enter the waveform as a fixed coherent harmonic bank.

A suitable next candidate should satisfy all of the following before any persistent training:

```text
- no deterministic sinusoidal harmonic bank as audible source
- F0/voicing retained as conditioning, not raw carrier authority
- learned or pulse/glottal-like voiced excitation with non-fixed phase structure
- independent aperiodic detail retained for consonants
- mel retains strong spectral-envelope/timbre authority
- no mel-only waveform shortcut
- exact waveform-length contract
- bounded CPU architecture smoke first
- full-utterance oracle listening remains the acceptance gate
```

This is a family-level architecture change, not a v4.x tuning exercise.

## Closed state

```text
v4.4 architecture smoke: PASS
v4.4 exact resume: PASS
v4.4 persistent training: PASS / CLOSED
v4.4 full-utterance perceptual acceptance: REJECTED
v4.4 path attribution: CLOSED
principal failure: deterministic periodic carrier remains perceptually exposed
additional v4.x training: NOT AUTHORIZED
next gate: design_and_smoke_non_sinusoidal_voiced_excitation_candidate
```
