# LYKENOX Speech — predicted duration inference policy

## Why this gate exists

The persistent acoustic frame-context v2 checkpoint has passed held-out teacher-duration
validation, including non-zero intra-token mel and F0 motion. Product inference still cannot
use teacher durations, so the next boundary is predicted timing.

The historical bootstrap inference rule was:

```text
round(duration_prediction)
-> clamp every valid token to min=1, max=80
```

That rule is no longer acceptable:

- alignment-v3 contains structural BOS/EOS/WB tokens that may legitimately have zero frames;
- alignment-v3 also observed valid non-pause timing above 80 frames;
- explicit punctuation pauses need a different safety range from ordinary phonemes;
- teacher durations must remain completely untouched by any inference safety policy.

## Versioned product-side policy

Policy version:

```text
predicted-duration-policy-v1
```

Semantics:

```text
<pad>                 -> always 0
<bos> <eos> <wb>      -> min 0, max 160
phonemes / <unk>      -> min 1, max 160
<pau_short/long>      -> min 1, max 320
```

BOS/EOS/WB are not forced to zero: if the trained duration predictor assigns positive
acoustic timing, that timing is retained. The important correction is that they are now
allowed to round naturally to zero.

The old `LykenoxSpeechConfig.max_duration_frames = 80` remains in historical checkpoint
configuration for backward compatibility, but normal inference no longer uses it as the
product duration policy. This avoids mutating checkpoint/config provenance after training.

## Teacher-duration boundary

When explicit durations are supplied to `LykenoxSpeechAcousticModel`, they bypass the
predicted-duration policy entirely. This preserves alignment-v3 supervision exactly,
including zero-duration structural tokens and durations greater than 80 frames.

## Model output

The acoustic model now exposes:

```text
regulated_durations
```

alongside mel, F0, voicing, raw duration prediction, mel mask and mel lengths. This gives the
runtime an auditable exact timing contract:

```text
sum(regulated_durations) == mel_lengths
```

## Mandatory bounded smoke

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_predicted_duration_semantics_smoke
```

The smoke does not train. It loads the accepted persistent v2 `best.pt`, validates synthetic
edge cases for every token class, proves teacher durations are still exact, then runs several
text-only acoustic inference probes without any waveform/reference/pitch-target input.

Required pass checks include:

```text
status: pass
architecture_identity_exact: true
policy_probe_exact: true
padding_forced_zero: true
structural_zero_duration_supported: true
structural_positive_duration_supported: true
content_min_one: true
pause_min_one: true
content_above_legacy_80_preserved: true
teacher_durations_preserved_exactly: true
text_only_inference_outputs_finite: true
predicted_duration_sum_matches_mel_length: true
reference_audio_required: false
waveform_pitch_target_required: false
```

Expected next gate:

```text
build_reference_free_text_to_waveform_smoke
```

A pass does not yet prove natural speech. It only closes timing semantics so the next gate can
connect predicted mel + predicted F0 + predicted voicing to the already accepted persistent
v4.1 LYKENOX vocoder and generate the first fully reference-free waveform from text.
