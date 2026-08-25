"""Phase 1 placeholder backend that prevents accidental fake output."""

from __future__ import annotations

from pathlib import Path

from lykenox_voice_engine.engines.singing_engine import SingingEngine, SynthesisRequest


class PlaceholderSingingEngine(SingingEngine):
    """Reject synthesis until a validated SVS backend is installed."""

    def synthesize(self, request: SynthesisRequest, output_dir: Path) -> Path:
        """Raise a clear error instead of producing fake vocals."""

        raise RuntimeError("No singing synthesis backend installed. Phase 1 does not fake vocals.")
