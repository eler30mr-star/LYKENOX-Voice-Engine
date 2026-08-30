# LYKENOX vocoder V6 — perceptual rejection

**Decision date:** 2026-08-30  
**Status:** rejected; checkpoints are forensic-only  
**Accepted comparison baseline:** v4.2

## Decision

V6 and the V6 clarity-guard continuation are rejected. No additional epoch, resume,
checkpoint selection, product integration, or reference-free reconnection is authorized.
The decisive gate was listening to three complete held-out oracle utterances after clarity
epoch 2. All three V6 outputs were materially less intelligible than v4.2 and exhibited a
stronger periodic whine plus nasal/gangoso coloration. Objective crop improvements did not
translate to speech.

The V6 training entrypoints now fail before dataset loading, optimizer construction,
artifact-directory creation, or checkpoint mutation.

## Full-utterance evidence

The oracle report itself did not grant acceptance:

- `persistent_training_complete = false`
- `full_utterance_perceptual_acceptance = false`
- `status = needs_listening`

Listening resolved that gate as a rejection. The complete-utterance metrics also contradicted
the optimistic crop interpretation:

| Utterance | Measure | v4.2 | V6 clarity |
|---|---:|---:|---:|
| 1 | reconstruction | 1.182956 | 1.589209 |
| 1 | envelope | 0.723677 | 1.091492 |
| 1 | spectral balance | 0.032817 | 0.264312 |
| 2 | reconstruction | 1.151972 | 1.522050 |
| 2 | envelope | 0.664389 | 0.955773 |
| 2 | presence 1–8 kHz error (dB) | 2.221135 | 7.029496 |
| 3 | reconstruction | 1.226940 | 1.782597 |
| 3 | envelope | 0.686155 | 1.276630 |
| 3 | local spectral contrast | 0.285444 | 0.352217 |

V6 was worse than v4.2 on reconstruction, envelope, spectral balance, and local spectral
contrast for all three utterances. Its full-utterance spectral centroids were only about
244–250 Hz, versus roughly 286–463 Hz for v4.2 and 341–542 Hz for the references. This is
consistent with the audible loss of consonants/formants and the low, periodic, gangoso
collapse.

## Root cause

The architecture was described as source-free, but that description was semantically wrong.
`network_v6.py` injects the following directly at sample rate before the waveform decoder:

1. accumulated F0 phase;
2. a narrow periodic phase aperture;
3. centered phase;
4. sample-rate voicing and log-F0;
5. deterministic noise in unvoiced regions.

Although these tensors were named conditioning controls rather than excitation, accumulated
phase and phase aperture provide a periodic carrier shortcut. The decoder can reproduce that
shortcut without learning sufficient mel-conditioned phonetic detail.

A second shortcut then amplifies the failure. `_normalized_waveform_shape` removes local mean,
divides the raw waveform by local RMS, and normalizes global RMS before applying the learned
level envelope. Consequently, an incorrect periodic shape is promoted to unit-scale structure
instead of being allowed to remain weak while the decoder learns speech detail.

The former flags `explicit_source = false` and `conditioning_only_waveform = true` captured
only the absence of a literal waveform bypass. They did not establish a genuinely source-free
waveform path. The public V6 contract now records:

- `source_free = false`
- `sample_phase_conditioning = true`
- `deterministic_unvoiced_noise_conditioning = true`
- `local_unit_rms_shape_normalization = true`
- `perceptually_rejected = true`

## Why the clarity guard failed

Band-power fractions are not intelligibility metrics. A periodic or noisy waveform can place
approximately the requested amount of energy in 1–3 kHz or 3–8 kHz while still carrying no
recognizable consonants, formants, or words. Crop-averaged presence and spectral-balance
losses therefore rewarded redistribution of wrong energy. They did not prove that the mel
content was encoded into the waveform.

The selection score also allowed improvements in weighted aggregate loss to hide regressions
in complete-utterance phonetic structure. The full-utterance oracle gate correctly exposed
that mismatch.

## Requirements for any successor

A successor is a new architecture, not a V6 continuation. Before persistent training it must
satisfy all of the following:

1. No accumulated phase, sinusoid, pulse, aperture, harmonic bank, deterministic noise, or
   other sample-rate carrier/excitation channel may enter the waveform decoder.
2. F0 and voicing may condition learned frame/latent representations, but may not be converted
   into sample-by-sample periodic phase controls.
3. No local or global unit-RMS normalization may force an arbitrary raw waveform shape to
   audible scale. Level control must not rescue an uninformative waveform.
4. The objective must directly protect mel-conditioned phonetic information, not only global
   bands, RMS, or periodicity. A differentiable analysis/re-encoding consistency term or an
   equivalent content-sensitive feature loss is required.
5. A complete held-out utterance oracle A/B against v4.2 is mandatory after the first bounded
   completed epoch. If words are not at least as intelligible as v4.2, training stops.
6. Objective metrics can reject a candidate but cannot grant perceptual acceptance.
7. Predicted duration remains unchanged. No post-hoc gain, normalization, EQ, or denoising may
   be used to conceal a waveform failure.

Until a successor passes those gates, v4.2 remains the comparison baseline and no V6 artifact
is a product candidate.
