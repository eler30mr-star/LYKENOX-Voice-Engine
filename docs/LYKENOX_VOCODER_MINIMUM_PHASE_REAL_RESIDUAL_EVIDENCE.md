# LYKENOX Minimum-Phase Vocoder — Real-Residual Resynthesis Evidence

**Policy:** `LYX-POL-001`  
**Diagnostic:** `scripts/diagnostic_real_residual_resynthesis_v1.py`  
**Diagnostic family:** minimum-phase source/filter forensic isolation  
**Device:** CPU only  
**Training executed:** no  
**Model used:** no  
**Checkpoint used:** no  
**Post-processing:** none

## Purpose

This diagnostic was created to separate the two halves of the active minimum-phase source/filter path:

1. spectral-envelope / minimum-phase filtering; and
2. synthetic excitation generation.

The diagnostic does not use `build_neutral_excitation`. Instead, it estimates the same order-64 minimum-phase spectral envelope from the owned reference waveform, removes that envelope in the STFT domain to obtain a real residual, and feeds that real residual back through the existing `render_time_varying_minimum_phase` path with the same cepstral envelope.

The test therefore asks a narrow question: **if the synthetic pulse+noise excitation is removed, can the current envelope/filter path resynthesize the owned reference cleanly?**

## Human listening result

The owner listened to the generated full-utterance `__real_residual_resynthesis.wav` output against the corresponding owned reference audio and reported that the resynthesis:

- sounds like the original voice audio;
- is clean;
- is natural;
- does not exhibit the previously dominant gangoso / rough synthetic character.

Owner report, recorded verbatim in substance: **"se escucha igual al audio original de mi voz, limpio y natural"**.

Under `LYX-POL-001`, this human listening result is the authoritative product-quality evidence for this diagnostic. Numeric metrics are not used to promote the result beyond what was actually heard.

## Interpretation

This is the first diagnostic in the current minimum-phase investigation that produces a clear perceptual improvement to the point of sounding like the owned original reference.

The result strongly isolates the dominant degradation away from the minimum-phase envelope/filter reconstruction itself and toward the **synthetic excitation path**. In particular:

- the order-64 cepstral envelope is capable of supporting clean resynthesis when paired with the real residual;
- the minimum-phase FIR conversion is capable of supporting clean resynthesis in this test;
- the time-varying renderer is capable of supporting clean resynthesis in this test;
- the synthetic pulse + aperiodic-noise source used by `build_neutral_excitation` remains the dominant unresolved source of the rough / gangoso texture.

This does **not** prove every aspect of the filter path is perfect under every future predicted envelope. It does prove that continuing to tune the predictor or loss while retaining the current synthetic excitation would confound the investigation and would ignore the strongest current perceptual evidence.

## Relationship to prior diagnostics

The preceding source/filter diagnostics established:

- **3b — no crossfade:** removing filter-output crossfade did not remove the artifact;
- **3c — Gaussian noise:** replacing hash noise with deterministic Gaussian noise did not remove the artifact;
- **3d — band-split excitation:** frequency-dependent periodic/aperiodic mixing improved the sound somewhat but did not fully solve it;
- **3e — cepstral order 32:** lowering the cepstral order did not remove the artifact;
- **3f — real residual:** replacing the synthetic excitation with the real residual produced clean, natural resynthesis close to the original reference.

Taken together, these results make synthetic excitation design the primary engineering target.

## Decision

The active minimum-phase family is **not rejected** by this evidence. The filter/envelope side has now demonstrated a clean oracle resynthesis path.

However, bounded training of the current predictor+renderer combination must not proceed as though the excitation problem were solved. The existing `build_neutral_excitation` source is not yet acceptable for product training or product-quality evaluation.

### Next authorized engineering action

Design and validate a new **LYKENOX-owned synthetic excitation model/path** that can approximate the real residual characteristics demonstrated by this diagnostic, while preserving:

- CPU-only execution;
- no third-party pretrained model or checkpoint;
- no remote inference;
- exact duration;
- no post-hoc gain normalization, EQ, denoise, or enhancement;
- deterministic/reproducible behavior where required;
- full held-out human listening as final acceptance authority.

The real-residual diagnostic remains a positive forensic reference and must not be overwritten by later experiments.
