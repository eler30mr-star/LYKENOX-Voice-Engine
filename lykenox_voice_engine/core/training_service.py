"""Training controls for the NNSVS CPU microtest backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lykenox_voice_engine.engines.nnsvs_engine import NnsvsEngine


class TrainingService:
    """Expose NNSVS microtest controls without enabling full training."""

    def __init__(self, root: Path | None = None) -> None:
        self.engine = NnsvsEngine(root)

    def check(self) -> dict[str, Any]:
        """Return NNSVS runtime readiness."""

        return self.engine.check_available()

    def prepare_microtest(self) -> dict[str, Any]:
        """Prepare the small authorized microtest dataset."""

        return self.engine.prepare_dataset("lykenox")

    def microtest(self) -> dict[str, Any]:
        """Run or block the NNSVS CPU microtraining gate."""

        prepared = self.prepare_microtest()
        result = self.engine.train("lykenox")
        result["dataset"] = prepared
        return result

    def stop(self) -> dict[str, str]:
        """Cancel active NNSVS backend jobs."""

        self.engine.cancel(None)
        return {"status": "cancelled"}
