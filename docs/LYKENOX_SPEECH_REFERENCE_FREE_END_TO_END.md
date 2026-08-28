# LYKENOX Speech — first reference-free end-to-end gate

## Closed prerequisites

The following gates are closed before this stage:

- persistent acoustic frame-context v2 training completed with `status: pass`;
- held-out v2 audit passed with exact frame contracts and non-zero intra-token mel/F0 motion;
- predicted-duration semantics passed with `predicted-duration-policy-v1`;
- the accepted persistent vocoder remains `lykenox_pitch_source_filter_v4_1`.

No additional acoustic or vocoder training is part of this gate.

## Product path under test

The first complete speech path is now:

```text
text
  -> es-phoneme-v1 frontend
  -> persistent acoustic frame-context v2
  -> predicted token durations
  -> predicted mel + F0 + voicing
  -> speech-vocoder-conditioning-v1
  -> persistent LYKENOX v4.1 vocoder
  -> WAV
```

The path explicitly requires none of the following at inference:

- reference WAV;
- waveform-derived F0 target;
- source speaker or singer;
- voice conversion;
- external TTS/SVS backend.

## Acoustic -> vocoder conditioning

Product-side conditioning is versioned as:

```text
speech-vocoder-conditioning-v1
```

The acoustic model emits a positive F0 hypothesis on every real frame plus voicing logits. The accepted v4.1 vocoder, however, was trained with binary voiced targets and F0 equal to zero on unvoiced frames.

The runtime boundary therefore:

1. thresholds acoustic voicing logits at probability `0.5`;
2. sets F0 exactly to zero on predicted-unvoiced and padded frames;
3. clamps predicted-voiced F0 to the support used by the persistent speech training pair: `60.0 .. 352.941176 Hz`;
4. preserves the exact mel frame grid;
5. sends only predicted mel/F0/voicing to the vocoder.

This is inference post-processing, not waveform analysis.

## Mandatory smoke

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_reference_free_end_to_end_smoke
```

The smoke loads:

```text
models/lykenox_identity/training/acoustic_frame_context_v2/best.pt
models/lykenox_identity/training/vocoder_source_filter_v4_1/best.pt
```

and synthesizes three fixed Spanish probe sentences from text only.

Generated WAVs and the report are written to:

```text
models/lykenox_identity/evaluation/reference_free_speech_v1/
  01_reference_free.wav
  02_reference_free.wav
  03_reference_free.wav
  reference_free_smoke_report.json
```

## Automatic pass criteria

A smoke `status: pass` requires:

```text
acoustic_identity_exact: true
vocoder_identity_exact: true
acoustic_vocoder_contract_exact: true
all_text_only_outputs_finite: true
all_duration_mel_waveform_lengths_exact: true
all_wav_headers_exact: true
all_waveforms_non_silent: true
all_probes_have_predicted_voiced_frames: true
reference_audio_required: false
waveform_pitch_target_required: false
source_speaker_or_singer_required: false
voice_conversion_required: false
```

The report also records predicted voiced fraction, predicted F0 range, fraction of voiced F0 frames clamped to training support, RMS/peak, duration, and coarse whole-waveform spectral-band fractions for each probe.

## Meaning of a pass

A pass proves that the first complete LYKENOX Speech chain executes from arbitrary text to a valid local WAV without reference audio or oracle pitch. It does **not** by itself prove intelligibility, identity similarity, naturalness, or release quality.

The mandatory next gate is:

```text
listen_and_audit_reference_free_end_to_end_wavs
```

Human/perceptual listening must evaluate the three generated WAVs before the path is accepted for runtime/export integration. If listening reveals a local conditioning problem, diagnose that integration first; do not restart acoustic or vocoder training by inertia.
