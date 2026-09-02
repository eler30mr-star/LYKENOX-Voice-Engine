# Full-TRAIN residual retrieval coverage V2 — execution evidence

Policy: `LYX-POL-001`

## Status

The resumable full-TRAIN retrieval coverage diagnostic completed locally on the owner CPU.

This is rejection/diagnostic evidence only. It does not authorize production integration, selector
training, an optimizer, a learned checkpoint, or replacement of the current residual codebook.

## Completed local result

Held-out windows evaluated: `2151`

Retained sampled codebook:

- retained codewords: `6234`
- original owned TRAIN candidate windows before retention: `119897`
- retention cap: `128` real residual vectors per conditioning bucket

Full TRAIN diagnostic retrieval bank:

- owned TRAIN residual windows: `119897`
- held-out windows admitted to bank: `false`
- training executed: `false`
- optimizer created: `false`
- checkpoint written: `false`
- production codebook replaced: `false`

Full TRAIN improved the best filtered-domain normalized MSE on `2046 / 2151` held-out windows:

- improvement count: `2046`
- improvement fraction: `0.9511854951185496`

## Global coverage comparison

### Retained 6,234-codeword artifact

- filtered-response cosine mean: `0.8070508714735868`
- filtered-response cosine median: `0.8505135178565979`
- fraction below cosine `0.80`: `0.35471873547187355`
- normalized MSE p50: `0.2766266167163849`
- normalized MSE p90: `0.6378332376480103`

### Full owned TRAIN retrieval bank

- filtered-response cosine mean: `0.8719168362064397`
- filtered-response cosine median: `0.9133571982383728`
- fraction below cosine `0.80`: `0.1915388191538819`
- normalized MSE p50: `0.1657785028219223`
- normalized MSE p90: `0.4952743649482727`

## Interpretation

The 512-sample / 256-hop residual representation is **not rejected as useless** by this evidence.
Searching all owned TRAIN residual windows materially improves local filtered-domain coverage over the
6,234-codeword artifact on 95.1% of held-out windows.

The dominant demonstrated bottleneck in the current artifact is therefore the aggressive retention
step (`max_per_bucket = 128`), not merely the downstream selector. The current PRESELECT_K issue is
also real, but selector/preselector changes cannot recover codewords that were discarded during
retention.

This evidence does **not** prove that the full-TRAIN bank is sufficient for clean product audio. Even
with every owned TRAIN residual vector available, `19.15%` of held-out windows remain below filtered-
response cosine `0.80`, and normalized MSE p90 remains `0.4953`. The clean Step-3f identity residual
roundtrip remains the perceptual ceiling.

## Next gate

Before designing a new signal-aware retention algorithm or authorizing any selector/model training,
measure the size/coverage curve of the **same existing deterministic hash-retention rule** at larger
TRAIN-only caps. Build one diagnostic nested bank at maximum cap `1024` and evaluate exact subsets at
caps `256`, `512`, and `1024` against the already-completed `128` and full-TRAIN ceilings.

This isolates how much of the full-TRAIN gain is recoverable by capacity alone without changing the
retention algorithm and without using held-out data to choose individual codewords.
