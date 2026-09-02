# Residual codebook hash-retention capacity sweep v1 — execution evidence

Policy: `LYX-POL-001`

## Owner-reported completed execution

The TRAIN-only hash-retention sweep completed locally after one timeout and one successful resumable
continuation. The final report status was:

`ready_for_hash_retention_capacity_curve_review`

No model training, optimizer, learned checkpoint, production renderer change, production codebook
replacement, held-out insertion into TRAIN, third-party voice component, or remote inference occurred.

## Fixed experimental facts

- Original owned TRAIN residual candidate windows: **119,897**.
- Existing production-inactive diagnostic codebook cap: **128 per conditioning bucket**.
- Nested hash-retention bank at cap 1024: **24,404 retained vectors**.
- The nested cap-128 subset reproduced the completed cap-128 baseline: **PASS**.
- Held-out evaluation windows: **2,151**.
- Candidate compatibility and frozen-renderer synthesis-domain scoring were unchanged.

## Global synthesis-domain coverage

| Retention | Mean cosine | Median cosine | NMSE p50 | Fraction cosine < 0.80 |
|---|---:|---:|---:|---:|
| hash cap128 | 0.8070508715 | 0.8505135179 | 0.2766266167 | 0.3547187355 |
| hash cap256 | 0.8228558514 | 0.8670662045 | 0.2481959760 | 0.3142724314 |
| hash cap512 | 0.8349809028 | 0.8795506954 | 0.2263906747 | 0.2761506276 |
| hash cap1024 | 0.8453791399 | 0.8893530965 | 0.2090510279 | 0.2482566248 |
| full TRAIN | 0.8719168362 | 0.9133571982 | 0.1657785028 | 0.1915388192 |

## Gap recovery relative to cap128 → full TRAIN

| Retention | Fraction of NMSE gap recovered | Windows improved vs cap128 |
|---|---:|---:|
| cap256 | 0.2365394787 | 0.4700139470 |
| cap512 | 0.4238224862 | 0.7252440725 |
| cap1024 | 0.5828669828 | 0.8377498838 |

## Interpretation

The capacity curve is monotonic, so insufficient retained capacity is real. However, simply increasing
the same deterministic hash sample from 128 to 1024 per bucket recovers only **58.29%** of the
cap128-to-full-TRAIN normalized-MSE gap despite retaining **24,404** vectors. Full TRAIN remains
materially better at 119,897 vectors.

Therefore:

1. The 512/256 residual representation is not rejected by this evidence.
2. The original 128-per-bucket retention is a demonstrated dominant bottleneck.
3. Blindly increasing the same hash cap has diminishing returns and is not the next preferred gate.
4. The next isolation should compare a **TRAIN-only signal-aware retention rule at the same 1024-per-
   bucket budget** against hash1024 and full TRAIN.
5. Held-out VAL may score the retained bank but must not choose retained codewords.
6. Metrics remain rejection-only; audible product acceptance still requires full held-out listening.
7. Selector/model training remains blocked.
