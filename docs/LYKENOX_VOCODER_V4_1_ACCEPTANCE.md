# LYKENOX Vocoder v4.1 — persistent acceptance record

> **Superseded by later full-utterance evidence.**  The record below is preserved as the historical decision made after the original short held-out gate.  Subsequent reference-free and oracle full-utterance audits reproduced a persistent periodic metallic/insect-like buzz even with target mel + target F0/voicing + teacher durations.  Harmonic-gain and source-shape ablations did not identify a safe runtime tweak.  Therefore v4.1 remains a valuable trained diagnostic checkpoint, but it is **not accepted for product runtime or final perceptual release**.  The active corrective gate is `docs/LYKENOX_VOCODER_V4_2.md`.  Do not continue v4.1 training by inertia.

## Historical decision

`lykenox_pitch_source_filter_v4_1` was accepted as the **persistent LYKENOX Speech vocoder architecture** for the next end-to-end development stage based on the evidence available at that time.

That historical gate closed the architecture/probe/checkpoint/resume/persistent-training/listening stage. It did **not** mean final product audio quality was frozen: end-to-end unseen-text synthesis still had to be audited after the acoustic model predicted F0/voicing instead of receiving waveform-derived oracle targets.

The product boundary remained:

```text
text
  -> LYKENOX Spanish frontend
  -> LYKENOX acoustic model
       -> mel + predicted F0 + predicted voicing
  -> LYKENOX v4.1 source-filter vocoder
  -> waveform
```

Normal inference must not require reference audio, a source speaker, voice conversion, an external TTS backend, cloud API, account, or remote model download.

## Persistent run result

The bounded exactly-resumable v4.1 run completed all 24 epochs / 2,832 updates and selected epoch 22 as the held-out best checkpoint.

```text
initial_validation_reconstruction:     2.624959
best_validation_reconstruction:        1.578153

initial_validation_spectral_balance:   0.849243
best_validation_spectral_balance:      0.295535

initial_validation_selection_score:    3.049581
best_validation_selection_score:       1.725920

confirmed_generated_specific_frame_locks: 0
subbass_or_silence_collapse_count:         0
upper_voice_band_missing_count:            0
automatic_artifact_gate_pass:              true
```

Best checkpoint:

```text
models/lykenox_identity/training/vocoder_source_filter_v4_1/best.pt
```

## Original held-out generated/reference review

Three validation pairs produced from the final best checkpoint were reviewed after the run.

Independent measurements on those final WAVs showed that the persistent run materially improved the earlier v4.1 probe:

- generated RMS was approximately 87.3-87.8% of the paired reference RMS across all three pairs;
- pairs 1 and 3 had strong RMS-envelope correspondence (about 0.95 and 0.93 respectively) and strong flattened log-mel correlation (about 0.87 in both cases);
- the low-energy second pair was a weaker temporal-envelope discriminator, but its centroid, flatness, RMS and broad spectral-band distribution remained close to its reference;
- spectral centroid was close to the paired reference rather than systematically several hundred hertz too high;
- spectral flatness was also much closer to the references than in the bounded probe;
- no frame-grid carrier, sub-bass/silence collapse, or missing-upper-band failure was detected by the established automatic gates.

At that point the remaining quality debt appeared local rather than architectural. The later full-utterance oracle evidence invalidated that assumption: the short listening gate did not expose the persistent buzz strongly enough.  This is precisely why full held-out utterances are now mandatory for v4.2 acceptance.

## Training-only oracle boundary

F0 and voicing used during the isolated vocoder run are extracted from the owned target WAV only to supervise/test the waveform stage. That oracle path must stop at training.

The acoustic model must predict the same F0/voicing contract from text/prosody before product inference is considered complete.

## Current engineering state

The pitch cache, acoustic F0/voicing heads, frame-context acoustic v2, and predicted-duration semantics remain separate completed gates.  The current waveform-stage blocker is the v4.1 full-utterance perceptual failure, and the active corrective architecture is the bounded v4.2 candidate.

Run only the v4.2 architecture smoke at this stage:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_2_architecture_smoke
```

Persistent v4.2 training is not authorized until that bounded gate passes.
