# LYKENOX Speech — acoustic F0/voicing heads

## Purpose

The persistent v4.1 vocoder is accepted with the conditioning contract:

```text
mel + F0 + voicing -> waveform
```

The persistent `speech-pitch-cache-v1` target cache is also accepted on all 132 speech utterances. The next requirement is therefore inside the LYKENOX acoustic model: it must predict F0 and voicing from text-derived frame representations rather than receiving waveform-derived oracle controls at product inference.

## Model contract

`LykenoxSpeechAcousticModel` now returns, on the exact same regulated frame grid as mel:

```text
mel
f0_prediction_hz
f0_log_prediction
voicing_logits
duration_prediction
mel_mask
mel_lengths
```

The F0 head predicts in log-Hz space and exposes a positive Hz value for the vocoder-facing contract. Its final log-F0 projection is initialized to 100 Hz as a stable optimization prior; this is not a fixed pitch and is trained against cached real contours.

Voicing is represented as logits and trained as a binary voiced/unvoiced target.

Both heads operate after teacher-duration length regulation during training. Therefore:

```text
predicted mel frames == cached mel frames == cached F0 frames == cached voicing frames
```

must hold exactly.

## Loss contract

The joint bounded smoke uses:

- masked mel L1 on real mel frames;
- masked log-duration Smooth-L1 on real text tokens;
- masked log-F0 Smooth-L1 only on target-voiced real frames;
- masked voicing BCEWithLogits on all real mel frames.

Unvoiced F0 is not regressed toward zero. Voicing decides whether a frame is voiced; F0 regression learns pitch only where the owned target says pitch is defined.

## Dataset boundary

`LykenoxAlignedSpeechDataset(..., include_pitch_targets=True)` reads F0/voicing only through the completed hashed `speech-pitch-cache-v1` index. It never re-runs pitch extraction during acoustic training.

This keeps waveform-derived F0 extraction strictly on the training-target preparation side of the architecture.

## Bounded CPU gate

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_acoustic_prosody_smoke
```

Default contract:

```text
steps: 40
batch_size: 2
max_mel_frames: 900
duration_weight: 0.10
f0_weight: 0.25
voicing_weight: 0.25
```

A pass requires the fixed real-batch probe to decrease all five reported objectives:

```text
total
acoustic
duration
f0
voicing
```

and requires exact teacher-duration/mel and pitch/mel frame contracts.

Expected next gate on pass:

```text
build_bounded_resumable_acoustic_trainer_with_prosody
```

## What a pass does not prove

This smoke is intentionally an overfit/gradient contract, not final acoustic training. A pass does not yet prove:

- unseen-text intelligibility;
- identity quality;
- predicted-duration correctness at inference;
- final F0 contour accuracy;
- end-to-end vocoder quality from predicted rather than oracle F0/voicing;
- ONNX/runtime export.

Those remain later gates.
