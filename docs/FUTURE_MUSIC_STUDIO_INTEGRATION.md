# Future LYKENOX Music Studio Integration

LYKENOX Music Studio generates structure, instrumental, arrangement, and melody.
When a vocal is needed, it calls the local Voice Engine API with lyrics, tempo, notes,
and profile id. The Voice Engine returns `vocal.wav`, then Music Studio mixes that vocal.

This is direct singing synthesis. The main path does not use source singer conversion.
