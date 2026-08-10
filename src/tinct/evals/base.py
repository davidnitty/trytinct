"""Eval types shared by the gate and harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EvalResult:
    """A single evaluated metric against its threshold."""

    metric: str
    value: float
    threshold: float
    higher_is_better: bool

    @property
    def passed(self) -> bool:
        if self.higher_is_better:
            return self.value >= self.threshold
        return self.value <= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "higher_is_better": self.higher_is_better,
            "passed": self.passed,
        }
