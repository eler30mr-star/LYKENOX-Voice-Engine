# LYKENOX continuous residual source — active replacement architecture

Policy: `LYX-POL-001`

## Decision

The discrete CELP/residual-codebook retrieval line is closed as an active product direction.

This is not based on one failed tuning choice. The accumulated evidence shows a structural mismatch:

- the exact Step-3f real residual through the fixed minimum-phase renderer reconstructs held-out voice cleanly;
- 512/256 sqrt-Hann analysis/OLA preserves that clean residual in the identity roundtrip;
- residual-domain selection, synthesis-domain selection, and synthesis-domain continuity all remained perceptually gangoso with the compressed codebook;
- exhaustive retrieval across all 119,897 owned TRAIN residual windows materially improves coverage, proving aggressive retention was harmful;
- however a deployable TTS system does not possess the held-out target residual needed to perform that oracle retrieval;
- hash1024 and equal-budget signal-aware1024 both remain far from exhaustive TRAIN coverage, while signal-aware1024 recovered only ~1.38% of the remaining hash1024-to-full-TRAIN NMSE gap.

Therefore the product source path must not depend on discrete cross-utterance residual retrieval.

## Active source architecture

`lykenox_voice_engine/models/vocoder/network_minimum_phase_continuous_source_v1.py`

The source is now an owned continuous autoregressive residual generator:

`mel80 + F0 + voiced + periodicity -> frame context -> recurrent residual state -> 512-sample residual analysis vector`

Properties:

- output rate is the existing 256-sample frame hop, not a transpose-convolution sample grid;
- each output is the same 512-sample sqrt-Hann residual representation whose exact OLA roundtrip already passed listening;
- the previous continuous residual vector is explicitly encoded into the recurrent state, so phase/fine-structure continuity is modeled directly;
- there is no codeword index, nearest-neighbour search, retrieval bank, selector model, or quantization bottleneck;
- inference is autoregressive at frame rate (~93.75 recurrent steps/s at 24 kHz), not sample-rate autoregression;
- CPU is the reference device.

## Training target

The only source target is the owned Step-3f real residual extracted from TRAIN recordings.

`speech_vocoder_continuous_residual_source_train_v1.py` supervises:

1. signed residual-vector shape/correlation;
2. residual-vector log RMS;
3. reconstructed continuous residual waveform;
4. final waveform after the unchanged minimum-phase renderer;
5. multi-resolution log-STFT reconstruction.

The training schedule transitions from teacher-forced previous residual vectors to fully free-running recurrent generation. Validation is always free-running.

The Step-3f oracle cepstrum is permitted inside this source-training stage only to keep the source target isolated from the already-separate envelope predictor problem. It is not a product inference input.

## Product-quality rule

Metrics may reject but cannot accept. Acceptance still requires complete held-out utterance listening.

The audible source acceptance surface is:

`scripts/render_continuous_residual_source_v1.py`

It writes, side by side:

- `__continuous_residual_source.wav` — the learned source through the unchanged renderer;
- `__identity_roundtrip_ceiling.wav` — the known-clean exact residual ceiling;
- `__reference.wav` — original held-out audio;
- `__continuous_predicted_residual.wav` — source inspection only; it is not expected to sound like speech.

## Closed paths

Do not spend further engineering time on:

- residual-codebook cap sweeps;
- hash retention redesign;
- codebook descriptor sweeps;
- Viterbi/beam continuity tuning;
- residual-cosine preselection tuning;
- codebook selector training;
- parametric Rosenberg/glottal-source variants already rejected;
- post-hoc EQ, denoise, gain normalization, or duration modification.

Historical diagnostics remain in the repository as evidence only.

## Ownership

The architecture, implementation, targets, weights, and inference path are LYKENOX-owned. No third-party pretrained model/checkpoint/service or remote TTS component is authorized.
