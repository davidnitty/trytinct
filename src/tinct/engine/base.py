"""Training engine interfaces and run artifacts.

The heavy implementations (see :mod:`tinct.engine.hf_trainer`) are imported
lazily; this module only defines types and protocols with lightweight deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class TrainingRun:
    """The durable result of a training run, laid out under ``runs/<name>/``."""

    name: str
    run_dir: Path
    adapter_dir: Path
    metrics_path: Path
    trainer_state_path: Path
    family: str
    model: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "run_dir": str(self.run_dir),
            "adapter_dir": str(self.adapter_dir),
            "metrics_path": str(self.metrics_path),
            "trainer_state_path": str(self.trainer_state_path),
            "family": self.family,
            "model": self.model,
            "metrics": self.metrics,
        }


@runtime_checkable
class TrainingEngine(Protocol):
    """Protocol implemented by concrete trainers (e.g. Llama LoRA/QLoRA)."""

    family: str

    def train(
        self,
        run_name: str,
        run_dir: Path,
        model: str,
        train_records: List[Dict[str, Any]],
        valid_records: List[Dict[str, Any]],
        config: Any,  # TrainConfig
    ) -> TrainingRun:
        """Run training and persist artifacts into ``run_dir``."""
        ...
