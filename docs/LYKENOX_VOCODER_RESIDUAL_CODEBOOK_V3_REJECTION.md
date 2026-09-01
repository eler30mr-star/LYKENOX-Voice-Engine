# LYKENOX residual codebook oracle V3 — perceptual rejection

**Date:** 2026-09-01  
**Policy:** LYX-POL-001  
**Status:** rejected as a selection method; codebook representation not yet rejected

## Human listening result

The owner reports that complete held-out `__residual_codebook_oracle_v3_sequence_coherent.wav`
remains **gangoso**. The corresponding `__selected_codebook_residual_v3_sequence_coherent.wav`
sounds like noise rather than speech.

The residual sound by itself is not an acceptance criterion. Earlier identity-roundtrip evidence
already proved that the exact held-out real residual may sound noisy/non-speech-like while the same
512/256 sqrt-Hann representation plus the frozen minimum-phase renderer resynthesizes clean speech
matching the original voice audio.

Therefore the rejection applies to **V3 codeword selection**, not to the residual representation,
OLA, or frozen renderer.

## What V3 tested

V3 used the existing 6,234-codeword owned-train residual codebook and:

- prohibited per-window polarity inversion;
- kept a top-K residual-domain candidate set;
- used positive residual-domain energy gain;
- selected a complete-utterance path with residual cosine emission cost plus overlap-continuity cost;
- used no model, training, optimizer, checkpoint, third-party voice component, remote inference,
  post-hoc gain normalization, EQ, denoise, or duration modification.

This did not remove the audible gangoso quality.

## Failure localization

The identity roundtrip remains the decisive control:

`exact held-out residual -> 512/256 analysis -> same vectors -> OLA -> frozen filter -> clean voice`.

V2 and V3 instead replace those exact vectors with train codewords and remain gangoso. V3 shows that
simple neighboring-overlap continuity is insufficient. The remaining oracle must evaluate codeword
quality **after the frozen synthesis filter**, because residual-domain similarity and continuity have
now failed perceptually.

## Next gate

No selector training is authorized.

The next diagnostic is a synthesis-domain codebook oracle. Candidate train codewords are converted to
their exact local waveform contribution through the unchanged time-varying minimum-phase renderer.
The oracle gain and final candidate choice are solved in that filtered waveform-response domain and
compared against the corresponding clean held-out target-vector contribution. Held-out target data
remain diagnostic only and never enter the codebook.

Metrics may reject but cannot accept product quality. Complete held-out listening remains final.
