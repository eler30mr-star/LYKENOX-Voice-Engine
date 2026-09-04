# LYKENOX vocoder — positive milestone through 2026-09-04

Policy: `LYX-POL-001`

This document freezes the positive engineering evidence reached before the CLEAN_V1 dataset pass. Human listening is the acceptance authority; objective metrics may reject or localize a defect but do not accept product quality by themselves.

## Confirmed positive evidence

1. **The fixed minimum-phase renderer is not the dominant failure.** Real Step-3f residual plus oracle cepstrum reconstructs clean/natural speech through the unchanged renderer. The identity/real-residual controls remain GOLD references.

2. **The old learned waveform-block paths are structurally closed.** The 512/256 overlapping source created a near-total 93.75 Hz grid comb; the non-overlapping 256-sample source still collapsed to a repeated deterministic template. Direct per-frame deterministic waveform regression remains rejected.

3. **The residual-statistics source removed the old hop-grid/template defect.** Its structural metrics are valid positive evidence, but the original candidate was not perceptually accepted as final quality.

4. **Residual phase/temporal coherence was isolated as the primary audible failure.** On `speech_0021_6cd35984e877_seg_001`, candidate residual magnitude combined with the real residual phase sounded correct/natural. Therefore the fixed renderer and candidate magnitude were capable of supporting a good render when the phase was correct.

5. **Magnitude-only Griffin-Lim was not sufficient.** The 64-iteration deterministic Griffin-Lim reconstruction became intelligible but retained a mild robotic texture. This is a partial improvement, not a PASS.

6. **Target temporal phase increments produced a major improvement.** `candidate magnitude + target temporal dphase + candidate initial phase anchor` sounded almost very good, with only a light wind/telephone-like residue. The zero-anchor version was thinner and more robotic. This established that temporal phase evolution is critical and that the per-frequency initial phase anchor also matters.

7. **The smooth group-delay anchor approximation was rejected.** It increased objective anchor alignment but sounded less clean than the candidate-anchor baseline. The full target-anchor ceiling sounded very good.

8. **`speech_0021` under full target phase is now a positive natural reconstruction.** The candidate-magnitude target-phase baseline was judged correct/natural. LOW/HIGH target-magnitude swaps did not provide a meaningful cleanup gain; the MID swap mainly changed the voice color to a thinner/brighter timbre. Remaining hiss/ambient noise is consistent with contamination already present in the original recording.

9. **`speech_0022_ba721f6129b9_seg_005` reference and identity roundtrip are clean.** Therefore the low intermittent grinder-like artifact is not caused by the original waveform roundtrip or the fixed renderer.

10. **The `speech_0022` grinder artifact follows candidate spectral shape, not broadband frame level.** In the level/shape decomposition under full target phase:
   - candidate spectral shape + target frame level made the artifact more obvious;
   - target spectral shape + candidate frame level sounded good and removed the characteristic grinder-like contamination.
   This localizes the remaining `speech_0022` magnitude failure primarily to the time-varying candidate spectral shape.

11. **A non-speech transient in the source data is a plausible contamination mechanism.** The original `speech_0022` contains a chicken call near the beginning. The candidate appears to distort/imitate that event as a grinder-like texture. A dedicated transient-localization diagnostic exists but is intentionally not the current blocker because the project is moving to a cleaned corpus first.

## Important unresolved boundary

The inference-time phase solution is **not yet solved**. Target residual phase/dphase are oracle controls and cannot be used as the final product mechanism. The experiments above localize what information is missing; they do not authorize copying target phase at inference.

The residual-statistics checkpoint is therefore **not accepted as the final product source** even though several controlled hybrids are now natural.

## Dataset decision: CLEAN_V1 is now the gate

All future training/reference audio must be derived from a cleaned, versioned corpus before new source training proceeds.

Rules:

- Preserve RAW originals unchanged.
- Build `CLEAN_V1` as a separate corpus.
- Remove or reject external events such as animals, tools/motors, wind, hum, excessive hiss, impacts, unrelated voices and other non-voice contamination.
- Cleaning must be conservative and must preserve the owner's timbre, consonants, formants, breath structure and natural attacks.
- If noise overlaps speech and cannot be removed without damaging the voice, reject that segment rather than over-process it.
- Regenerate mel, F0, periodicity, cepstrum, real residual targets and all caches/derived features from CLEAN_V1. Do not reuse acoustic targets computed from dirty WAVs.
- For vocoder evaluation, generated RAW output remains un-denoised; post-vocoder denoise must not be used to hide model defects.

## Current engineering freeze

Until CLEAN_V1 exists and passes auditory validation:

- no new source architecture training;
- no renderer modification;
- no predicted-duration modification;
- no product-path EQ/denoise/gain patching;
- no reopening of deterministic waveform-block regression.

The next active project step is **CLEAN_V1 corpus construction and validation**, followed by regeneration of acoustic targets and re-running the GOLD oracle controls before any new training.

## Preserved diagnostics

The transient spectral-shape localization diagnostic is preserved for forensic comparison after CLEAN_V1 but is not currently required to proceed:

- `143f2bb9bc97945987e1b81f7aaec7ee9264a073` — `diagnostic(vocoder): localize speech0022 transient spectral-shape contamination`
- `30f4160ea1592117176f1be79977fc676be7e5e8` — `test(vocoder): lock speech0022 transient spectral-shape localization gate`

This milestone should be treated as the restart point if later work regresses or loses context.
