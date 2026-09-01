# LYKENOX residual codebook oracle — decision and policy scope

**Policy:** LYX-POL-001 v1.0  
**Status:** diagnostic capacity test authorized; production integration NOT authorized  
**Device:** CPU only  
**Data boundary:** codebook = owned `train` residual only; held-out `val` never enters the codebook

## Why this line is being opened

The minimum-phase envelope/filter path has positive full-utterance evidence: Step 3f replaced the
synthetic pulse+noise source with the owned real residual and the owner reported the result clean and
natural, matching the original voice recording. The calibrated Rosenberg + measured four-band
aperiodicity candidate subsequently remained gangoso/rough even after local calibration from 97,168
owned pitch-synchronous cycles.

This isolates the remaining quality problem to the synthetic excitation representation strongly
enough to justify, under Section 9 of LYX-POL-001, a new source representation test without changing
the validated filter path.

## Why this is CELP-style rather than a literal CELP product codec

A classical CELP encoder can search codevectors because it has the target speech/residual available.
A TTS inference path does not have the target residual for a new sentence. Therefore the current work
uses analysis-by-synthesis only as a **held-out oracle capacity test**:

1. build a codebook exclusively from owned `train` real-residual windows;
2. expose a held-out `val` residual only to the diagnostic search objective;
3. select the closest compatible train codevector and an oracle scalar gain;
4. reconstruct the excitation and pass it through the already validated minimum-phase filter;
5. listen to complete held-out output against both reference and the Step-3f real-residual ceiling.

The held-out-selected indices and gains are explicitly invalid for product inference. If the codebook
capacity test succeeds, a separate LYKENOX-owned selector/gain predictor decision and training gate
will be required. No selector training is authorized by this document.

## Codebook representation

The artifact is implemented in
`lykenox_voice_engine/training/speech_residual_codebook_v1.py`.

- source: real residual extracted by the same Step-3f inversion method;
- source split: `train` only;
- vector length: 512 samples;
- hop: 256 samples;
- analysis/synthesis: paired periodic sqrt-Hann windows;
- retention: deterministic SHA-256 bounded sampling within voicing/F0/periodicity buckets;
- artifact tensor: `models/lykenox_identity/calibration/residual_codebook_v1.pt`;
- provenance index: `models/lykenox_identity/calibration/residual_codebook_v1.json`;
- every contributing WAV is recorded with SHA-256 provenance;
- no gradient training, optimizer, external model, external codebook, remote inference, EQ, denoise,
  gain normalization or duration change is used.

The tensor is an owned data-derived DSP artifact, **not a model checkpoint**.

## Held-out oracle

`scripts/diagnostic_residual_codebook_oracle_v1.py` builds/loads the train-only codebook and runs the
held-out search. Candidate search is restricted first by compatible voicing state, F0 neighborhood
and periodicity neighborhood. For each held-out target vector it computes a bounded non-negative
least-squares gain and chooses the minimum residual-domain squared-error codevector.

This target-dependent gain is deliberately diagnostic-only. It does not make the route usable for
new text and must not be copied into production.

Outputs are written to
`models/lykenox_identity/evaluation/vocoder_minimum_phase_residual_codebook_oracle_v1/` and include:

- `<id>__residual_codebook_oracle.wav`;
- `<id>__selected_codebook_residual.wav`;
- `<id>__reference.wav`;
- `residual_codebook_oracle_report.json`.

The report also points to the Step-3f real-residual resynthesis ceiling when present.

## Acceptance / rejection rule

Metrics and nearest-neighbor error can reject capacity, but cannot accept voice quality. The only
positive outcome is human listening to all complete held-out utterances showing that the codebook
oracle approaches the clean/natural Step-3f ceiling without reintroducing the gangoso/rough texture.

- If the oracle is still gangoso/rough: reject this residual codebook representation before any
  selector is trained.
- If the oracle is consistently close to the Step-3f ceiling: the representation has demonstrated
  capacity, and only then may a separate decision consider a small LYKENOX-trained selector/gain
  predictor.

## Policy conformance

This diagnostic is compatible with LYX-POL-001 because all identity-bearing codevectors come from
owned/authorized train recordings; no third-party learned voice component is used even temporarily;
there is no remote inference; production is untouched; no persistent model training is authorized;
and complete held-out human listening remains the final quality authority.
