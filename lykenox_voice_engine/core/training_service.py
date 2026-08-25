"""Training controls for validated singing synthesis backends."""

from __future__ import annotations


class TrainingService:
    """Expose preflight controls without launching Phase 1 training."""

    def check(self) -> dict[str, str]:
        """Return training readiness without starting a run."""

        return {"status": "blocked", "reason": "No validated SVS backend installed."}

    def microtest(self) -> dict[str, str]:
        """Return the required microtest state for Phase 2."""

        return {"status": "not_run", "reason": "Microtest must be implemented per backend."}
