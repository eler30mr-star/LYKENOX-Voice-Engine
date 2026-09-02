# LYKENOX residual codebook V5 — synthesis-domain coherent oracle

Policy: `LYX-POL-001`

## Purpose

V5 tests the combination that previous diagnostics isolated but did not test together:

- V3 used temporal continuity but scored codewords in the raw residual domain.
- V4 moved final candidate scoring and non-negative least-squares gain into the exact local frozen-renderer waveform domain, but selected every window independently.
- V5 preserves V4 unchanged and adds bounded temporal sequence search in the same filtered waveform domain.

This is a diagnostic capacity test only. It does not authorize selector training or production integration.

## Frozen V4 components

V5 imports and reuses these V4 functions without modifying V4:

- `_preselect_residual_candidates`
- `_local_filtered_vector_response`

The V4 non-negative least-squares gain is also retained exactly in the local filtered-response domain.

## V5 sequence search

For each held-out analysis window:

1. obtain the same broad residual-cosine preselection used by V4;
2. compute the exact local frozen-renderer response for the target and every preselected codeword;
3. solve the same V4 non-negative least-squares gain in the filtered waveform domain;
4. retain every preselected candidate with its local filtered-response MSE;
5. search complete utterance paths with a deterministic bounded beam.

Default beam size: `8`.

Default continuity weight: `1.0`.

Path cost is:

`accumulated_local_filtered_response_MSE + continuity_weight * filtered_overlap_discontinuity_MSE`

The continuity term compares consecutive gain-scaled candidate responses over the exact 256-sample region corresponding to the OLA overlap shared by adjacent 512-sample / 256-hop excitation windows. The comparison remains in the filtered waveform domain.

## Ownership and policy constraints

- Codebook source remains owned/authorized `train` real residual only.
- Held-out residual and cepstrum are oracle targets only and are never written into the codebook.
- Oracle indices, gains, beam path, and continuity decisions are invalid for product inference.
- CPU only.
- No model, optimizer, gradient update, or checkpoint.
- No third-party model, voice component, pretrained weight, or remote service.
- No duration modification.
- No post-hoc output gain normalization, EQ, denoise, or enhancement.
- Production renderer remains frozen.
- Metrics may reject but cannot accept product quality.
- Complete held-out human listening remains authoritative.

## Output

V5 writes to:

`models/lykenox_identity/evaluation/vocoder_minimum_phase_residual_codebook_oracle_v5_synthesis_domain_coherent/`

For each held-out utterance it writes:

- `__residual_codebook_oracle_v5_synthesis_domain_coherent.wav`
- `__selected_codebook_residual_v5_synthesis_domain_coherent.wav`
- `__identity_roundtrip_ceiling.wav`
- `__reference.wav`

The report also records the corresponding V4 waveform path for direct comparison.

## Listening gate

Compare V5 against both V4 and the clean identity-roundtrip ceiling.

- If V5 clearly reduces gangoso/batido relative to V4, filtered-domain temporal incoherence is supported as a dominant V4 defect. Only after that listening result may beam size or continuity weight be considered for a subsequent experiment.
- If V5 remains materially the same as V4, do not continue tuning sequence search automatically. First inspect per-window filtered-response coverage, including low-cosine / high-MSE windows, to determine whether the owned codebook lacks sufficient residual transition coverage.

No further algorithm iteration is authorized by this document before the owner reports the V5 listening result.
