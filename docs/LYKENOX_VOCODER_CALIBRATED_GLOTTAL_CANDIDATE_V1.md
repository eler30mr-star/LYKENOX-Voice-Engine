# LYKENOX calibrated glottal excitation candidate v1

Status: **implemented for calibration + oracle evaluation only; not production-active**.
Policy: `LYX-POL-001`.
Execution target: CPU only.

## Confirmed evidence that motivates this candidate

The real-residual analysis/resynthesis diagnostic established a positive perceptual reference:
when the owned real residual extracted from a held-out recording is passed back through the
existing order-64 minimum-phase envelope/filter path, the owner reports that the result sounds
clean, natural, and like the original voice recording. This evidence is recorded separately in
`LYKENOX_VOCODER_MINIMUM_PHASE_REAL_RESIDUAL_EVIDENCE.md`.

Therefore the current engineering hypothesis is intentionally narrow: preserve the demonstrated
envelope/filter path and replace only the defective generic pulse+noise excitation with a source
whose identity-bearing DSP parameters are measured from owned recordings.

Previous oracle A/B evidence also records:

- filter-output crossfade is not the dominant defect;
- hash noise versus seeded Gaussian noise is not the dominant defect;
- lowering cepstral order from 64 to 32 is not the dominant defect;
- splitting periodic/aperiodic excitation by frequency band gives a partial perceptual improvement,
  but is not sufficient on its own.

## Candidate design

The candidate uses a deterministic Rosenberg glottal pulse rather than the historical flat
band-limited impulse source. Its parameters are not textbook identity constants. They come from
owned train recordings after extracting the same real residual used by the successful diagnostic.

`lykenox_voice_engine/training/speech_glottal_calibration.py` measures pitch-synchronous cycles and
aggregates by 20 Hz F0 bins:

- an open-quotient proxy based on high-energy occupancy inside the real residual cycle;
- residual-pulse asymmetry using normalized peak position;
- residual spectral tilt in dB/octave;
- residual RMS so source amplitude is calibrated rather than arbitrary.

The artifact is written locally as:

`models/lykenox_identity/calibration/glottal_pulse_v1.json`

It records source utterance IDs, source WAV paths and SHA-256 hashes, pitch/data contract versions,
algorithm version and CPU-only/no-training/no-third-party flags.

`lykenox_voice_engine/training/speech_band_aperiodicity_calibration.py` measures real-residual
harmonic/inter-harmonic power in four bands:

- 0-1 kHz
- 1-2 kHz
- 2-4 kHz
- 4-8 kHz

and stores `noise / (harmonic + noise)` statistics by the same F0 bins in:

`models/lykenox_identity/calibration/band_aperiodicity_v1.json`

## Excitation implementation

`lykenox_voice_engine/training/speech_vocoder_minimum_phase_glottal_excitation_v1.py` implements the
candidate. It refuses to load calibration artifacts unless they match the expected owned versions,
`LYX-POL-001`, `split=train`, 24 kHz geometry, and explicit no-third-party provenance.

For voiced cycles it generates a pitch-synchronous Rosenberg pulse using interpolated measured open
quotient and asymmetry. The pulse is spectrally shaped toward the measured residual tilt and scaled
to the measured residual RMS. A deterministic seeded Gaussian source supplies the aperiodic
component.

A complementary FIR bank divides the source into 0-1, 1-2, 2-4, 4-8 and >8 kHz bands. The first
four use their directly measured aperiodicity curves; the >8 kHz remainder conservatively inherits
the 4-8 kHz calibration. The bands sum to a unit impulse, so the bank does not intentionally leave
spectral holes or duplicate a band.

This is calibrated DSP. There are no learned weights or gradient updates.

## Required oracle before any production change

`scripts/diagnostic_calibrated_glottal_oracle_v1.py` keeps the proven order-64 oracle envelope and
production minimum-phase filtering path fixed and changes only the excitation. It renders three
complete validation utterances as raw FLOAT WAV without output gain normalization, EQ, denoise or
duration changes.

Each candidate WAV must be listened to against:

1. the real reference recording;
2. the historical synthetic-excitation oracle;
3. the real-residual resynthesis, which is the demonstrated quality ceiling for the current filter.

Metrics may reject but cannot approve this candidate. Human listening of the complete held-out
utterances remains the acceptance authority.

## Integration rule

The production renderer is **not** modified by this candidate implementation. Training and new
checkpoints remain blocked while the existing synthetic excitation is known to be perceptually
defective.

Only if the calibrated glottal oracle improves clearly and consistently across the held-out
utterances may a separate production-integration decision replace `build_neutral_excitation` and
update the active renderer/excitation version. If it does not improve sufficiently, the candidate
remains forensic evidence and no production source change is authorized.
