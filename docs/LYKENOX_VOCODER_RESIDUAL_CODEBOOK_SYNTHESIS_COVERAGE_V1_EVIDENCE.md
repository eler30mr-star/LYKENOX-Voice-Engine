# LYKENOX Residual Codebook Synthesis Coverage V1 Evidence

Policy: `LYX-POL-001`

## Owner-reported local execution

The owner reports that `scripts/diagnostic_residual_codebook_synthesis_coverage_v1.py` completed
successfully on CPU against the existing owned TRAIN residual codebook (`6,234` retained codewords,
`58` buckets) and three held-out VAL utterances.

No model training, optimizer, checkpoint, production renderer change, third-party voice component,
or remote inference was involved.

## Global result

- Held-out windows: `2,151`
- PRESELECT_K missed the best retained compatible codeword on `1,129` windows.
- Preselector miss fraction: `0.5248721524872152` (`52.49%`).

### PRESELECT_K coverage

- filtered-response cosine mean: `0.7844196161257838`
- median: `0.8288975358009338`
- fraction below cosine `0.80`: `0.4258484425848443`
- normalized MSE p50: `0.3129289150238037`
- normalized MSE p90: `0.6780089735984802`

### Every retained compatible codeword

- filtered-response cosine mean: `0.8070508714735868`
- median: `0.8505135178565979`
- fraction below cosine `0.80`: `0.35471873547187355`
- normalized MSE p50: `0.2766266167163849`
- normalized MSE p90: `0.6378332376480103`

## Per-utterance result

### `speech_0021_6cd35984e877_seg_001`

- windows: `773`
- preselector miss fraction: `0.4864`
- PRESELECT_K cosine mean: `0.7769`
- all-retained-compatible cosine mean: `0.7981`
- PRESELECT_K fraction below `0.80`: `0.4101`
- all-retained-compatible fraction below `0.80`: `0.3480`

### `speech_0022_ba721f6129b9_seg_005`

- windows: `702`
- preselector miss fraction: `0.4630`
- PRESELECT_K cosine mean: `0.7520`
- all-retained-compatible cosine mean: `0.7703`
- PRESELECT_K fraction below `0.80`: `0.5342`
- all-retained-compatible fraction below `0.80`: `0.4744`

### `speech_0024_1778f351cc1f_seg_006`

- windows: `676`
- preselector miss fraction: `0.6331`
- PRESELECT_K cosine mean: `0.8267`
- all-retained-compatible cosine mean: `0.8554`
- PRESELECT_K fraction below `0.80`: `0.3314`
- all-retained-compatible fraction below `0.80`: `0.2382`

## Interpretation

Two failure mechanisms are supported simultaneously:

1. The residual-cosine PRESELECT_K stage is materially lossy. It excludes the best retained
   compatible synthesis-domain candidate on more than half of held-out windows.
2. Fixing only that preselection is insufficient. Even exhaustive search across every retained
   compatible codeword leaves `35.47%` of held-out windows below filtered-response cosine `0.80`,
   with normalized MSE p90 `0.6378`.

The existing `6,234`-entry artifact is not an exhaustive TRAIN bank. Its builder retains at most
`128` real residual vectors per conditioning bucket using deterministic hash sampling. Therefore the
current evidence rejects both PRESELECT_K as sufficient and the *compressed* 128-per-bucket bank as
sufficiently demonstrated, but does not yet distinguish compression loss from a structural limit of
the 512-sample / 256-hop cross-utterance codevector representation.

## Next gate

Run `scripts/diagnostic_full_train_residual_retrieval_coverage_v1.py` to compare the existing
compressed artifact directly against a diagnostic retrieval bank containing every owned TRAIN
residual vector under the same compatibility rule and exact filtered-domain scoring.

- If full-TRAIN materially closes the gap, redesign retention/compression before any selector.
- If full-TRAIN improves only marginally and substantial held-out coverage gaps remain, reject the
  current cross-utterance 512/256 codevector capacity before any selector training.

Metrics remain rejection evidence only. They cannot accept product voice quality.
