# Full-TRAIN residual retrieval coverage — resumable execution gate

Policy: `LYX-POL-001`

## Status

The original full-TRAIN coverage diagnostic did not complete on the owner CPU and produced no final
coverage report. This is an execution-engineering failure, not evidence that full-TRAIN retrieval
coverage is good or bad.

Owner-reported local execution on 2026-09-02:

- first run stopped around 3 minutes;
- second run stopped around 15 minutes;
- third run ran around 32 minutes and ended with code 1 without a visible traceback;
- no `full_train_residual_retrieval_coverage_v1_report.json` was produced;
- the diagnostic full-TRAIN bank *was* successfully materialized under evaluation:
  - `full_train_residual_retrieval_bank_v1.pt` (~245 MB reported locally),
  - `full_train_residual_retrieval_bank_v1.json`.

No model training, optimizer, checkpoint, production renderer change, production codebook replacement,
or held-out insertion into TRAIN occurred.

## Execution correction

`scripts/diagnostic_full_train_residual_retrieval_coverage_v2_resumable.py` preserves the same
coverage question and exact scoring contract while changing execution only:

1. Reuses the already-built diagnostic full-TRAIN bank when its metadata proves the expected version,
   policy, TRAIN source split, and held-out exclusion.
2. Reuses the completed retained-codebook exhaustive coverage CSVs instead of recomputing the 6,234
   codeword baseline for every held-out window.
3. Computes each held-out target's exact local filtered response once per window and reuses it across
   full-TRAIN candidate chunks.
4. Writes every completed held-out window to a durable partial CSV checkpoint.
5. On restart, reloads completed window indices and continues from the next unfinished window.
6. Supports selecting one held-out utterance and/or a bounded number of newly evaluated windows.
7. Produces the final V2 report only after all requested held-out windows have completed.

This does not weaken the diagnostic: candidate compatibility remains the existing voicing/F0/
periodicity rule; candidate scoring remains exact local frozen-renderer response with non-negative
least-squares gain; metrics remain rejection-only.

## Gate

Until the resumable full-TRAIN coverage report exists:

- full-TRAIN retrieval capacity is **undetermined**;
- the 128-per-bucket compression remains a live suspected bottleneck;
- no production codebook expansion is authorized;
- no selector/model training is authorized;
- no new selection-algorithm iteration is authorized.
