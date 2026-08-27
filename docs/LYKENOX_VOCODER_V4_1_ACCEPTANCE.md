# LYKENOX Vocoder v4.1 — persistent acceptance record

## Decision

`lykenox_pitch_source_filter_v4_1` is accepted as the **persistent LYKENOX Speech vocoder architecture** for the next end-to-end development stage.

This closes the architecture/probe/checkpoint/resume/persistent-training/listening gate. It does **not** mean final product audio quality is frozen: end-to-end unseen-text synthesis still has to be audited after the acoustic model predicts F0/voicing instead of receiving waveform-derived oracle targets.

The product boundary remains:

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

## Final held-out generated/reference review

Three validation pairs produced from the final best checkpoint were reviewed after the run.

Independent measurements on those final WAVs show that the persistent run materially improved the earlier v4.1 probe:

- generated RMS is approximately 87.3-87.8% of the paired reference RMS across all three pairs;
- pairs 1 and 3 have strong RMS-envelope correspondence (about 0.95 and 0.93 respectively) and strong flattened log-mel correlation (about 0.87 in both cases);
- the low-energy second pair is a weaker temporal-envelope discriminator, but its centroid, flatness, RMS and broad spectral-band distribution remain close to its reference;
- spectral centroid is now close to the paired reference rather than systematically several hundred hertz too high;
- spectral flatness is also much closer to the references than in the bounded probe;
- no frame-grid carrier, sub-bass/silence collapse, or missing-upper-band failure was detected by the established automatic gates.

The remaining quality debt is local rather than architectural. The generated signals still tend to under-represent the highest speech band (roughly above 3 kHz) relative to the references, and some segments remain spectrally smoother/noisier than the target. Those issues are appropriate targets for later end-to-end quality work; they do not justify another vocoder architecture reset.

The acceptance criterion is met: at least two of three held-out examples preserve clear vocal temporal/spectral structure without reintroducing the rejected structural artifacts, while the automatic artifact gate is clean.

## Training-only oracle boundary

F0 and voicing used during the isolated vocoder run are extracted from the owned target WAV only to supervise/test the waveform stage. That oracle path must stop at training.

The acoustic model must predict the same F0/voicing contract from text/prosody before product inference is considered complete.

## Next engineering gate

Create a versioned persistent F0/voicing target cache aligned exactly to the speech mel frames, with dataset/config/provenance checks. Then add acoustic-model F0 and voicing prediction heads and validate them with a bounded CPU smoke before any end-to-end unseen-text synthesis.
