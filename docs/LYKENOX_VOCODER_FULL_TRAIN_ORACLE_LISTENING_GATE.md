# FULL-TRAIN residual oracle listening — final CELP/codebook gate

Policy: `LYX-POL-001`

## Evidence entering this gate

The retained-codebook line has exhausted the useful metric-only retention experiments.

The completed hash-cap sweep showed monotonic capacity improvement, but hash1024 retained 24,404
TRAIN vectors and recovered only `0.5828669827721884` of the cap128-to-full-TRAIN total-NMSE gap.
Its held-out fraction below filtered-response cosine 0.80 remained `0.2482566248256625`, versus
`0.1915388191538819` for exhaustive FULL TRAIN.

The equal-budget TRAIN-only signal-aware retention experiment also completed. It retained exactly
24,404 vectors, matching hash1024, but recovered only `0.013819832240533042` of the remaining
hash1024-to-full-TRAIN NMSE gap. It improved `0.41608554160855415` of held-out windows and worsened
`0.3895862389586239`. Mean filtered-response cosine changed only from `0.8453791398522135` to
`0.8455636371351408`.

Therefore the tested signal-aware descriptors do not materially solve retention, and another
retention-cap/descriptor sweep is not the active next action.

## Final audible gate

`scripts/diagnostic_full_train_residual_oracle_listening_v1.py`

This diagnostic uses the already-built 119,897-vector owned TRAIN retrieval bank directly. For one
held-out utterance it performs the exact synthesis-domain oracle selection already validated by the
coverage audits, checkpoints selected FULL-TRAIN index + non-negative LS gain per window, then
reconstructs the complete excitation and renders the complete held-out WAV through the unchanged
minimum-phase renderer.

The first target is held-out index 2, `speech_0024_1778f351cc1f_seg_006`, because that is the
previously reported gangoso codebook case.

Output WAVs:

- `__full_train_residual_oracle.wav` — decisive FULL-TRAIN codebook ceiling;
- `__identity_roundtrip_ceiling.wav` — known-clean exact-real-residual ceiling;
- `__reference.wav` — original held-out reference;
- `__selected_full_train_residual.wav` — excitation inspection only; it is not expected to sound like speech.

## Decision rule

- If FULL TRAIN remains dominantly gangoso versus the clean identity roundtrip, close the current
  discrete CELP/codebook retrieval line. Do not spend more CPU on retention, beam width, preselection,
  or selector training.
- If FULL TRAIN is audibly and materially better, retain the conclusion that very high TRAIN coverage
  is required and only then consider a more efficient owned representation that can reproduce that
  audible ceiling.

Human listening is the authority. Metrics cannot accept quality.

No model training, optimizer, learned checkpoint, production codebook replacement, renderer change,
post-hoc gain normalization, EQ, denoise, third-party voice component, remote inference, or held-out
insertion into TRAIN is authorized by this gate.
