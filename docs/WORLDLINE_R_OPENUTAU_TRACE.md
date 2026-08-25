# WORLDLINE-R OpenUtau 0.1.565 Request Trace

Upstream: `https://github.com/openutau/OpenUtau`

Tag: `0.1.565`

Commit: `a60ca5830b9064556157245d4bf8f5920d93e5f8`

## Real Pipeline

1. `OpenUtau.Core.Render.RenderPhrase`
   - Builds phrase-level timing from validated `UPhoneme` objects.
   - Builds `phrase.pitches` every 5 ticks.
   - Pitch unit is cents of MIDI tone: MIDI 60 is stored as `6000`.

2. `OpenUtau.Core.Render.RenderPhone`
   - Copies validated phoneme data into render-time values.
   - Uses milliseconds for `positionMs`, `endMs`, `leadingMs`, `preutterMs`, `overlapMs`.
   - Uses ticks for `position`, `duration`, `leading`.

3. `OpenUtau.Core.Ustx.UPhoneme`
   - `ValidateOverlap` computes `preutter`, `overlap`, `tailIntrude`, and `tailOverlap`.
   - `ValidateEnvelope` creates five envelope points:
     - `p0.x = -preutter`
     - `p1.x = p0.x + max(overlap, 5)` for overlapped phones
     - `p2.x = max(0, p1.x)`
     - `p3.x = DurationMs - tailIntrude`
     - `p4.x = p3.x + tailOverlap`

4. `OpenUtau.Classic.ResamplerItem`
   - `velocity = phone.velocity * 100`
   - `stretchRatio = pow(2, 1 - velocity * 0.01)`
   - `skipOver = phone.oto.Preutter * stretchRatio - phone.leadingMs`
   - `durRequired = phone.endMs - phone.positionMs + phone.durCorrectionMs + skipOver`
   - `durRequired = max(durRequired, phone.oto.Consonant)`
   - `durRequired = ceil(durRequired / 50 + 0.5) * 50`
   - `pitches[i] = round(sampleLerped - phone.tone * 100)`

5. `OpenUtau.Classic.WorldlineRenderer`
   - Calls `PhraseSynthAddRequest(item, posMs, skipMs, lengthMs, fadeInMs, fadeOutMs)`.
   - `posMs = item.phone.positionMs - item.phone.leadingMs - (phrase.positionMs - phrase.leadingMs)`
   - `skipMs = item.skipOver`
   - `lengthMs = item.phone.envelope[4].X - item.phone.envelope[0].X`
   - `fadeInMs = item.phone.envelope[1].X - item.phone.envelope[0].X`
   - `fadeOutMs = item.phone.envelope[4].X - item.phone.envelope[3].X`
   - Calls `PhraseSynthSetCurves` with F0 sampled every 10 ms from `phrase.pitches`.

## Cutoff Semantics

OpenUtau passes `phone.oto.Cutoff` directly to WORLDLINE-R.

In `cpp/worldline/classic/timing.cpp`:

```cpp
return request.cut_off < 0
  ? -request.cut_off
  : model.total_ms() - request.offset - request.cut_off;
```

Therefore `cutoff = -50` means WORLDLINE-R uses a fixed 50 ms input region, not "trim
50 ms from the end".

## Current Blocking Finding

The LYKENOX generated `oto.ini` uses `cutoff=-50` broadly. With exact OpenUtau
semantics, WORLDLINE-R receives about 50 ms of input for many samples. Minimal
pitch tests using the official `Resample` export and the `PhraseSynth` path both
fail to impose MIDI 60/62/64 reliably.

Changing `cutoff=-50` to tail-trim behavior improves amplitude, but that is not
OpenUtau semantics and was not kept as the adapter behavior.
