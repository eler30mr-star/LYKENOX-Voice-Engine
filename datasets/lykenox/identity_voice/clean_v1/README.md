# LYKENOX Identity Voice — CLEAN_V1

`CLEAN_V1` is the mandatory cleaned identity-speech corpus for all new persistent speech/vocoder training after the 2026-09-04 data-quality gate.

Policy: `LYX-POL-001` v1.1.

## Boundary

The existing `prepared/speech_segmented` WAV files remain immutable historical source material. External offline tools may be used to clean or restore copies of those WAV files when their terms permit the resulting audio to be used by LYKENOX. The external tool, model, checkpoint or service is not integrated into LYKENOX, is not distributed with LYKENOX, is not required for inference, and may not provide initialization/distillation/weight transfer into LYKENOX models.

Cleaned WAV files live locally in `clean_v1/wav/` and are intentionally ignored by Git. Provenance manifests and gate reports are text artifacts and may be versioned after review.

## Gate sequence

1. `python scripts/prepare_identity_voice_clean_v1.py`
   - inventories the exact current train/val segmented corpus;
   - hashes every source WAV and source manifest;
   - never edits/copies over the source;
   - creates `work_manifest.csv` and the local `wav/` destination.
2. Clean the files offline, preserving each `utterance_id` filename, and write results to `clean_v1/wav/`. Files that cannot be cleaned without damaging the voice should be rejected rather than overprocessed.
3. `python scripts/validate_identity_voice_clean_v1.py --tool-name ... --tool-version ... --tool-terms-note ...`
   - rechecks source hashes;
   - checks clean WAV presence, sample-rate/channel geometry, finite samples, non-silence, clipping and excessive duration changes;
   - records the external-tool provenance boundary;
   - creates `listening_review.csv` with `PENDING` auditory decisions.
4. Listen to the clean corpus. For every technically valid item set `auditory_decision` to `ACCEPT` or `REJECT`. Metrics are never allowed to accept perceptual quality.
5. `python scripts/activate_identity_voice_clean_v1.py`
   - refuses activation while any listening decision is pending;
   - writes accepted `train.clean_v1.csv` and `val.clean_v1.csv` manifests;
   - freezes source/clean hashes and external-tool provenance;
   - switches identity-speech dataset consumers to CLEAN_V1.
6. Regenerate all clean-derived mel/F0/periodicity/cepstrum/residual targets and caches, rerun GOLD oracle controls, then explicitly authorize new persistent training.

## Voice-preservation rule

Cleaning must remove/reduce external contamination without changing the LYKENOX voice identity. Preserve timbre, formants, consonants, attacks, useful breaths and natural dynamics. Reject a segment when external noise overlaps the voice so strongly that cleaning produces metallic, watery, phasey, robotic or otherwise altered speech.

## Current training rule

CLEAN_V1 activation alone does **not** authorize source training. The active source entrypoint additionally requires:

- `all_acoustic_targets_and_caches_regenerated = true`
- `gold_oracles_rerun_after_clean_v1 = true`
- `training_authorized = true`

This prevents dirty pitch/residual/feature caches from silently surviving the corpus switch.
