# LYKENOX continuous residual source V2 — level-factored source architecture

Policy: `LYX-POL-001`

## Root failure being replaced

The first continuous residual source completed 600 updates and generated full held-out audio without
codebook retrieval or generation-time teacher forcing. Its held-out prediction/reference RMS ratios
were approximately `0.098`, `0.128`, and `0.153`.

That is an architecture/training failure, not a listening-volume issue. V1 asked one unconstrained
512-sample head to represent both residual fine structure and absolute residual energy. Its strongest
shape objective was largely scale-invariant and residual energy was only an auxiliary loss. This
permitted a low-energy free-running solution.

V1 is therefore rejected for the active source path. No post-hoc output gain is permitted as a fix.

## V2 source representation

V2 predicts each residual analysis vector as

`vector = unit_rms_shape * exp(log_rms)`

with two separate learned outputs:

1. **Residual shape** — 512-sample continuous vector, normalized structurally to unit RMS. Shape is
   autoregressive at frame rate so previous fine structure can inform the next frame.
2. **Residual level** — one bounded log-RMS scalar per residual vector. It is predicted explicitly
   from the current owned acoustic conditioning and recurrent shape state.

Previous absolute residual amplitude is deliberately excluded from the recurrent input. A low-level
frame therefore cannot recursively force later frames toward silence.

The bounded log-RMS parameterization is differentiable and initialized to a finite nonzero source
level; it does not use post-hoc normalization.

## Training authority

The trainer supervises:

- residual-vector directional/shape similarity;
- vector log-RMS directly;
- reconstructed residual relative error;
- reconstructed residual sequence log-RMS;
- rendered waveform relative error;
- rendered waveform sequence log-RMS;
- true log-magnitude multi-resolution STFT error.

The model is trained only from owned `train` Step-3f residual targets. Validation is fully
free-running on complete held-out utterances, not only short crops, so long-run level drift affects
checkpoint selection.

The fixed minimum-phase renderer remains unchanged. There is no codebook, external model/checkpoint,
remote service, post-hoc gain normalization, EQ, denoise, or duration modification.

## Active files

- `lykenox_voice_engine/models/vocoder/network_minimum_phase_continuous_source_v2.py`
- `lykenox_voice_engine/training/speech_vocoder_continuous_residual_source_train_v2.py`
- `scripts/render_continuous_residual_source_v2.py`
- `scripts/train_continuous_residual_source_v2.py`
- `lykenox_voice_engine/training/speech_vocoder_active_source_decision.py`

Human complete held-out listening remains the product-quality authority; metrics can reject but not
accept quality.
