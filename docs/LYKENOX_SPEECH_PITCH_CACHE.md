# LYKENOX Speech — persistent F0/voicing target cache

## Purpose

The accepted `lykenox_pitch_source_filter_v4_1` vocoder consumes:

```text
mel + F0 + voicing -> waveform
```

During isolated vocoder training, F0 and voicing were extracted from the paired owned WAV so the waveform stage could be tested independently. That waveform-derived oracle path is training-only and cannot exist in normal product inference.

The next acoustic-model stage therefore needs persistent supervised targets aligned exactly to the existing speech mel frames. This cache provides that contract.

## Versioned contract

Target extractor:

```text
lykenox-pitch-v1
```

Persistent cache:

```text
speech-pitch-cache-v1
```

Default pitch analysis:

```text
sample rate:                   24000 Hz
hop length:                    256 samples
frame length:                  1024 samples
nominal F0 request:            60-350 Hz
voiced periodicity threshold:  0.30
voiced RMS fraction:           0.08
```

Pitch-v1 searches integer autocorrelation lags. Its existing accepted implementation converts the nominal bounds with integer truncation. At 24 kHz, the nominal upper request of 350 Hz maps to lag 68, so the highest discrete F0 bin the extractor can actually emit is:

```text
24000 / 68 = 352.941176... Hz
```

The effective pitch-v1 validation interval is therefore approximately:

```text
60.000000-352.941176 Hz
```

This is not a new extractor and does not change `lykenox-pitch-v1`; it documents the exact discrete grid already used by the accepted vocoder training. Cache validation must use those realizable integer-lag bounds rather than reject a legal pitch-v1 bin merely because it is slightly above the nominal 350 Hz request.

Every target artifact stores one value per mel frame for:

```text
f0_hz
voiced
periodicity
```

Unvoiced F0 is exactly zero. `voiced` is binary. All three vectors must have exactly the same length as the corresponding cached mel sequence.

## Alignment rule

The active speech mel frontend uses centered STFT framing. For an utterance waveform after the same mono/resample/peak-normalization frontend, the expected mel frame count is:

```text
floor(waveform_samples / hop_length) + 1
```

The cache builder requires this centered waveform frame count to equal the already-cached mel frame count before accepting an utterance.

The pitch extractor keeps the existing v1 autocorrelation algorithm used by the accepted vocoder experiments. Its input contract was widened to full utterances; exact-hop vocoder crops preserve their existing first `frame_count` target values.

## Provenance and invalidation

Each per-utterance cache identity includes:

- cache version;
- pitch-target version;
- split and utterance ID;
- resolved WAV path;
- full WAV SHA256;
- mel frame count;
- split manifest SHA256;
- mel-cache version;
- speech mel configuration SHA256;
- explicit pitch-analysis configuration.

Changing owned audio, manifests, mel configuration, target extractor version, or pitch configuration therefore changes the cache identity instead of silently reusing stale targets.

After all train/val targets exist, `cache_index.json` records every artifact path and SHA256 plus dataset/config provenance. It also records the effective integer-lag F0 bounds. Future acoustic training must load targets through this completed index rather than re-running pitch extraction during training.

## Bounded/resumable builder

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_pitch_cache
```

The default command budget is 70 seconds with an 8-second checkpoint reserve. If the run returns:

```text
status: incomplete
next_gate: rerun_same_pitch_cache_command
```

rerun the exact same command. Existing valid target artifacts are reused.

A completed gate requires:

```text
status: pass
exact_centered_frame_alignment_count == total_count
indexed_reload_exact_count == total_count
all_targets_exact_mel_length == true
all_centered_frame_counts_match_mel == true
next_gate == add_acoustic_f0_voicing_heads
```

The completed cache is expected under:

```text
datasets/lykenox/identity_voice/features/speech/pitch-v1/
```

with `train/`, `val/`, `cache_index.json`, `cache_report.json`, and `cache_progress.json`.

## Product boundary

These cached targets are training data only. They do not make reference audio part of inference.

The intended product path remains:

```text
text
  -> LYKENOX Spanish frontend
  -> LYKENOX acoustic model
       -> predicted mel + predicted F0 + predicted voicing
  -> LYKENOX v4.1 vocoder
  -> waveform
```

After this cache passes locally, the next engineering gate is to add frame-level F0 and voicing heads to the acoustic model and prove with a bounded CPU smoke that their supervised losses decrease without regressing mel/duration behavior.
