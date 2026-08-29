# LYKENOX Vocoder v4.3 — perceptual rejection and next diagnostic

Persistent v4.3 training completed numerically with `status: pass`, but full-utterance oracle listening rejected the result: the user reports that v4.3 is perceptually worse than v4.2. Therefore v4.3 is **not** accepted for product/runtime use and must not be connected to `/speak` or reference-free acoustic predictions.

The full-utterance structural audit itself passed and the objective comparison versus v4.2 was mixed:

```text
envelope_loss improved:               3/3
reconstruction_loss improved:         1/3
spectral_balance_loss improved:       1/3
local_spectral_contrast_loss improved:2/3
```

This is important because it shows that better magnitude/envelope objectives do not guarantee better perceptual fine structure. V4.3's stricter carrier contract can improve spectral envelope while still sounding worse.

## Working failure hypothesis

V4.3 intentionally removed the v4.2 additive source shortcut and forced all audible output to originate from a deterministic 24-harmonic carrier plus aperiodic excitation, transformed only through a bias-free mel-conditioned multiplicative filter.

The regression therefore raises two specific possibilities:

1. the 24-harmonic deterministic carrier is too rich / phase-coherent and remains perceptually exposed;
2. the multiplicative-only filter is too restrictive to transform that deterministic carrier into natural vocal fine structure, even though its magnitude envelope improves.

Do **not** start v4.4 training yet. First isolate those two causes on the already-trained v4.3 checkpoint.

## Authorized next gate

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_3_carrier_fine_structure_ablation
```

The diagnostic uses one complete held-out oracle-conditioned utterance and generates:

```text
baseline_24h_noise005.wav
harmonics_16_equal_rms.wav
harmonics_12_equal_rms.wav
harmonics_8_equal_rms.wav
noise_floor_0p10.wav
noise_floor_0p20.wav
carrier_fine_structure_ablation_report.json
```

No training occurs, no checkpoint is mutated, and the exact trained baseline must reproduce sample-for-sample before any variant is written.

Listen in this order:

```text
baseline_24h_noise005
-> harmonics_16_equal_rms
-> harmonics_12_equal_rms
-> harmonics_8_equal_rms
-> noise_floor_0p10
-> noise_floor_0p20
```

Interpretation:

- if equal-RMS harmonic truncation removes the regression while speech remains useful, the 24-harmonic carrier is too exposed/rich;
- if increased voiced aperiodic noise reduces the metallic periodic component while preserving intelligibility, the carrier is too phase-coherent;
- if neither family helps, the multiplicative-only filter is too restrictive and the next architecture must restore controlled learned nonperiodic/residual capacity without reintroducing an additive raw-carrier shortcut.

Until this diagnostic is closed, no additional persistent vocoder training is authorized.
