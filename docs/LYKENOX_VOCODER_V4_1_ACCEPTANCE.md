# LYKENOX Vocoder v4.1 — candidate acceptance record

## Decision

`lykenox_pitch_source_filter_v4_1` is accepted as the **persistent vocoder candidate** for LYKENOX Speech. This is not yet a final runtime release artifact and does not authorize unbounded training by itself.

The decision is based on the bounded v4.1 probe plus held-out generated/reference analysis. The architecture remains LYKENOX-owned and local:

```text
text -> LYKENOX acoustic model -> mel + predicted F0 + predicted voicing
     -> LYKENOX source-filter vocoder -> waveform
```

Target/reference WAVs are used only during isolated training supervision. Final inference must not require reference audio, a source speaker, voice conversion, external TTS, API, account, or model download.

## Automatic gate that passed

The bounded v4.1 probe reported:

- architecture: `lykenox_pitch_source_filter_v4_1`
- trainable parameters: 12,585
- best held-out reconstruction: 1.874196
- spectral-balance validation improved
- confirmed generated-specific frame locks: 0
- sub-bass/silence collapse count: 0
- missing upper-voice-band count: 0
- automatic artifact gate: pass

Held-out pitch estimates tracked the paired references closely:

- validation 01: generated 99.58 Hz / reference 99.58 Hz
- validation 02: generated 95.80 Hz / reference 94.86 Hz
- validation 03: generated 83.91 Hz / reference 84.50 Hz

## Held-out signal review

All six v4.1 WAVs were reviewed as three exact generated/reference pairs. Objective analysis confirms that v4.1 is materially different from the rejected v0-v3 failure modes:

- generated/reference RMS-envelope correlation is approximately 0.88-0.93, showing that speech-scale temporal modulation is being followed rather than replaced by a stationary carrier;
- generated/reference log-mel cosine similarity is approximately 0.97-0.985;
- pitch measured independently with a second YIN-style analysis remains close to the references in all three pairs;
- the first two pairs preserve spectral-envelope structure more strongly than the third, so the model is a candidate rather than a finished vocoder.

Remaining quality debt is local rather than architectural:

- generated RMS is still only about 51-66% of the reference RMS;
- generated spectral flatness remains substantially above the references, indicating excess broadband/metallic or noisy energy;
- generated spectral centroid is still higher than the references;
- the third held-out pair has weaker 300-3000 Hz envelope correspondence than the first two.

These are training/conditioning/filter-quality targets. They do not justify returning to mel-only transposed/resize/polyphase vocoders.

## Next engineering gate

Before any longer run, the v4.1 candidate must have its own versioned persistent checkpoint/resume contract. The contract must preserve exact generator architecture/hyperparameters, pitch-target version, spectral-balance version, mel/segment provenance, discriminator state, both optimizer states, validation metrics, epoch/global-step position, and exact resume position.

Run the bounded contract smoke before building or starting persistent training:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_source_filter_contract_smoke
```

A pass closes the checkpoint/provenance gate only. The following gate is a bounded, resumable v4.1 trainer that can never require a command longer than the local execution budget.
