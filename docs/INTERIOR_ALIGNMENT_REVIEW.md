# LYKENOX residual interior alignment gate

After `alignment-v2` moved leading/trailing CTC blanks to BOS/EOS, the target laptop audit changed from 25 long non-pause tokens in 23 utterances to 9 tokens in 8 utterances. The remaining pattern is mixed, with 5 interior and 4 residual boundary outliers.

Do not delete, clamp, or exclude those utterances automatically. A long duration can come from two mechanically different sources:

1. the CTC target state itself occupies a long acoustic region; or
2. a long interior CTC blank run is divided between neighboring phonemes by the duration policy.

The targeted forensic command reruns only the already-flagged utterances and decomposes each duration into `direct_target_frames` and `allocated_blank_frames`. It also reports their fractions, neighboring phonemes, token-relative duration statistics, and log-mel energy summaries.

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_interior_alignment_review
```

Possible diagnoses:

- `interior_blank_allocation_dominant`: fix the interior blank ownership policy before acoustic training.
- `direct_ctc_occupancy_dominant`: inspect transcript/acoustic mismatch or genuinely long spoken regions before changing duration policy.
- `mixed_interior_alignment_mechanisms`: review the small residual set individually; do not apply one global correction blindly.
- `no_interior_outliers`: proceed to the residual-boundary check or aligned acoustic smoke as appropriate.

This command performs no training and does not regenerate the 132-item duration cache.
