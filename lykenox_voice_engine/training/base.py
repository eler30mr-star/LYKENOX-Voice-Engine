"""Training contracts for LYKENOX-owned model artifacts.

External research code may be studied or used during experiments, but a training
adapter must export a self-contained LYKENOX artifact. The shipped product must
not require that external trainer or its executable at inference time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingPlan:
    profile: str
    mode: str
    dataset_dir: Path
    output_dir: Path
    language: str = "es"


@dataclass(frozen=True)
class TrainingResult:
    success: bool
    model_dir: Path | None
    artifact_format: str
    architecture_family: str
    message: str


class LykenoxTrainer(ABC):
    """Interface for producing a persistent LYKENOX model artifact."""

    @abstractmethod
    def validate(self, plan: TrainingPlan) -> dict[str, object]:
        """Validate dataset, hardware, licenses, and export path before training."""

    @abstractmethod
    def train(self, plan: TrainingPlan) -> TrainingResult:
        """Produce a model artifact owned by the LYKENOX product contract."""

    @abstractmethod
    def export_runtime_artifact(self, training_output: Path, destination: Path) -> Path:
        """Export the final self-contained artifact used by the LYKENOX runtime."""
