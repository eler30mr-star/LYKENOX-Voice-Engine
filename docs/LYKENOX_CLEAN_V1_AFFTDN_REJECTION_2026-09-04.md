# CLEAN_V1 — AFFTDN listening rejection — 2026-09-04

Policy: `LYX-POL-001` v1.1

## Gate

The `FFmpeg afftdn` calibration was run only as an external/offline dataset-preparation trial. It did not write canonical `CLEAN_V1` WAVs, did not train any model, and did not mutate the owned source corpus.

Profiles auditioned:

- `SOURCE`
- `CONSERVATIVE`: 6 dB reduction target
- `MODERATE`: 10 dB reduction target

## Human auditory result

The identity voice remained recognizable/preserved in all three versions, but the cleaning objective failed:

- `SOURCE`: voice is clear, but the environmental/background noise remains fully audible.
- `CONSERVATIVE`: reduces the noise only slightly and introduces a perceptual impression of the voice being in a covered/muffled environment.
- `MODERATE`: the same covered/muffled environmental character becomes more pronounced.
- Neither processed profile meaningfully cleans the noise to the level required for `CLEAN_V1`.

Human auditory judgment is the acceptance authority. Therefore the objective signal-cleaning gate fails despite preservation of voice identity.

## Decision

`FFmpeg afftdn` is **REJECTED** as the canonical batch cleaner for the 132-item `CLEAN_V1` corpus.

The 132-item batch must **not** be processed with either AFFTDN profile.

No AFFTDN result is authorized as canonical `CLEAN_V1` audio.

## Next gate

Calibrate a qualitatively different external/offline speech-noise suppressor on the same representative material before any corpus-wide processing. The next trial may use an external pretrained tool under the explicit offline-tooling allowance of `LYX-POL-001` v1.1, provided that:

- the tool/model/checkpoint remains outside LYKENOX;
- no third-party model/weight becomes a LYKENOX training or runtime dependency;
- source WAVs remain immutable;
- trial output remains non-canonical until human listening acceptance;
- sample geometry is restored before comparison;
- no gain normalization, EQ, dereverb, or post-vocoder masking is used to manufacture a pass.

Current state remains: `CLEAN_V1` prepared, external cleaning unresolved, training unauthorized.
