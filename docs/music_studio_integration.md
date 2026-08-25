# Future Music Studio Integration

LYKENOX Music Studio remains responsible for composition, style, instrumental, and structure.

Future flow:

1. Music Studio builds lyrics, melody, pitch, timing, tempo, and instrumental.
2. Music Studio calls LYKENOX Voice Engine on `127.0.0.1`.
3. Voice Engine synthesizes `vocal.wav` from score data and LYKENOX Voice.
4. Music Studio mixes `vocal.wav` with instrumental and masters.

No source singer audio is sent. This is synthesis, not voice conversion.
