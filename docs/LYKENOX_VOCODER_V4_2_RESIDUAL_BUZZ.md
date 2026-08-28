# LYKENOX Vocoder v4.2 — residual buzz attribution

## Perceptual decision after full-utterance oracle listening

Persistent v4.2 training passed numerically and the full-utterance oracle audit improved target-referenced metrics versus v4.1, but listening still reports a residual metallic/insect-like chillido accompanying the voice.

Current decision:

```text
v4_2_training_status: pass
v4_2_full_utterance_structural_gate: pass
v4_2_perceptual_progress_vs_v4_1: improved
v4_2_full_utterance_perceptual_acceptance: false
product_runtime_acceptance: false
additional_training_authorized: false
```

The residual is smaller than v4.1, so v4.2 is a useful architectural improvement, not a failed experiment.  However, the target artifact is still audible and therefore the oracle waveform stage is not closed.

## Attachment-side objective confirmation

Direct comparison of the uploaded v4.1/v4.2 oracle pairs is consistent with the listening report: v4.2 generally reduces narrow spectral peak dominance and increases spectral flatness, while preserving similar RMS.  This supports the interpretation that the periodic character is being reduced rather than merely hidden by level changes.

The remaining problem is still not treated as a loudness-only issue and no output normalization is authorized as a substitute for waveform cleanup.

## Next causal gate

V4.2 has an explicit internal decomposition:

```text
envelope_path + source_gate(mel) * source_features
```

The next no-training audit scales only the complete transformed source contribution after the learned source stem and gate while leaving the mel-conditioned residual/skip filter unchanged.

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_2_residual_source_path_ablation
```

It writes one complete held-out utterance at:

```text
source_path_gain_1p0.wav
source_path_gain_0p75.wav
source_path_gain_0p5.wav
source_path_gain_0p25.wav
source_path_gain_0p0.wav
```

and verifies that gain `1.0` reproduces the trained v4.2 generator exactly before interpreting any listening difference.

Interpretation:

- if the residual chillido decreases with source-path gain while useful speech remains, the remaining defect enters primarily through source-branch authority;
- if the chillido remains at gain `0.0`, the downstream mel-conditioned filter/waveform projection can generate it independently of the explicit source branch;
- if reducing the source path removes both chillido and useful voice, v4.2 still depends on source leakage and a runtime gain reduction is not an acceptable product fix.

No new persistent training, `/speak` integration, release export, or reference-free acceptance is authorized until this attribution gate is closed.
