# LYKENOX Residual Codebook V5 Rejection Evidence

Policy: `LYX-POL-001`

## Result

The owner reports that the complete held-out V5 synthesis-domain coherent oracle retains the same dominant perceptual problem as prior codebook oracles: the final speech remains gangoso.

V5 combined the two previously isolated ideas at the same time:

- V4 exact local frozen-renderer waveform-domain candidate evaluation and non-negative least-squares gain.
- Deterministic bounded beam search with continuity measured in the filtered waveform domain over the 256-sample adjacent OLA overlap.

The added filtered-domain continuity did not produce a meaningful perceptual improvement. Therefore lack of adjacent-window continuity is not supported as the dominant remaining failure mechanism for the current 6,234-entry codebook.

## What this does not prove

This result does not by itself prove that the CELP-style codebook principle is universally invalid. The clean identity roundtrip remains positive evidence that the 512/256 sqrt-Hann representation, OLA, and frozen minimum-phase renderer can reproduce clean held-out speech when the correct residual trajectory is supplied.

The remaining unresolved question is coverage: whether the current owned-train codebook actually contains sufficiently close filtered-domain excitation vectors for held-out windows, and whether V4/V5's residual-cosine preselection hides better candidates that do exist.

## Gate

No selector training, optimizer, checkpoint, production integration, or renderer modification is authorized by this result.

The next diagnostic is `scripts/diagnostic_residual_codebook_synthesis_coverage_v1.py`, which measures per-window filtered-domain coverage for both V4's PRESELECT_K candidates and every compatible owned-train codeword. Metrics may reject coverage or the preselector, but cannot accept product quality.
