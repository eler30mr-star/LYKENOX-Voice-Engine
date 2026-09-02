# Signal-aware TRAIN-only retention v1 — active diagnostic gate

Policy: `LYX-POL-001`

## Evidence entering this gate

The completed hash-cap sweep reproduced the cap128 baseline and measured a monotonic capacity curve:

- cap128 mean filtered-response cosine: `0.8070508715`;
- cap256: `0.8228558514`;
- cap512: `0.8349809028`;
- cap1024: `0.8453791399`;
- full TRAIN: `0.8719168362`.

At cap1024, 24,404 TRAIN vectors recover only `0.5828669828` of the cap128-to-full-TRAIN total-NMSE
gap and still leave `0.2482566248` of held-out windows below filtered-response cosine 0.80. Full
TRAIN leaves `0.1915388192` below 0.80.

Therefore increasing the existing hash cap alone has not recovered enough of the available full-TRAIN
coverage to justify another blind capacity increase.

## Active isolation

`scripts/diagnostic_residual_codebook_signal_aware_retention_v1.py`

The diagnostic keeps the same maximum 1024 codewords per conditioning bucket and the same expected
24,404-vector total budget, but changes retention using TRAIN-only fixed DSP signal descriptors.
Held-out VAL does not select codewords.

Descriptor axes:

1. zero-crossing rate;
2. normalized spectral centroid;
3. upper-half spectral-energy fraction;
4. lag-1 normalized autocorrelation;
5. temporal energy centroid;
6. signed half-window balance.

Each axis is divided into three within-bucket rank quantiles. The six ternary coordinates create up to
729 signal-shape strata. Retention round-robins across occupied strata before taking additional members
from a stratum; a deterministic source-index hash is only a within-stratum tie/order rule.

## Comparison contract

The signal-aware 1024 bank is compared against:

- hash1024 at the same retention budget;
- exhaustive full TRAIN as the retrieval ceiling.

Scoring remains the exact local frozen-renderer waveform response with the same non-negative
least-squares gain and the existing voicing/F0/periodicity compatibility rule.

The decisive fields are:

- `coverage.hash1024`;
- `coverage.signal1024`;
- `coverage.full_train`;
- `equal_budget_comparison.fraction_of_hash1024_to_full_train_nmse_gap_recovered`;
- `equal_budget_comparison.fraction_windows_improved_vs_hash1024`.

## Gate

If signal1024 materially beats hash1024 at the same 24,404-vector budget, retention quality—not only
capacity—is confirmed as a major bottleneck and the signal-aware line may advance to an oracle
listening test. If it does not materially beat hash1024, do not blindly increase the hash cap; reassess
the TRAIN-only retention descriptor/geometry or the bank size required to approach full TRAIN.

No selector/model training, optimizer, learned checkpoint, production codebook replacement, renderer
change, held-out insertion into TRAIN, or product acceptance is authorized by this gate.
