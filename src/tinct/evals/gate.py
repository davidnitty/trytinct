"""Eval gate — decides whether a trained checkpoint clears the bar.

Pure and dependency-free: it consumes a metrics history (e.g. the ``log_history``
list that TRL/Transformers writes) plus an optional precomputed metric value, and
produces a fail-closed :class:`RuleReport`. Checkpoints that fail the gate must
not be shipped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tinct.core.config import EvalConfig
from tinct.core.rules import RuleReport, error, ok
from tinct.evals.base import EvalResult

# Metric keys that appear in a transformers log_history entry.
_LOWER_IS_BETTER = {"eval_loss", "perplexity"}


def extract_metric(history: List[Dict[str, Any]], metric: str) -> Optional[float]:
    """Return the last recorded value for ``metric`` in a log_history list."""
    found = None
    if history:
        for entry in reversed(history):
            if isinstance(entry, dict) and metric in entry and entry[metric] is not None:
                found = float(entry[metric])
                break
    return found


class EvalGate:
    """Fail-closed evaluator over a single metric vs a threshold."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def evaluate(self, history: List[Dict[str, Any]],
                 override_value: Optional[float] = None) -> RuleReport:
        report = RuleReport("Eval Gate")
        metric = self.config.metric
        higher_is_better = self.config.higher_is_better

        value = override_value if override_value is not None else extract_metric(history, metric)
        if value is None:
            report.add(error(
                "eval.metric_found", "Metric available",
                f"Metric {metric!r} was not found in the run's history. "
                "Cannot gate without an evaluation result.",
                meta={"metric": metric},
            ))
            return report

        result = EvalResult(metric, value, self.config.threshold, higher_is_better)
        if result.passed:
            report.add(ok(
                "eval.gate", "Eval gate",
                f"{metric} = {value:.4f} {'>=' if higher_is_better else '<='} threshold {self.config.threshold:.4f}",
                meta=result.to_dict(),
            ))
        else:
            report.add(error(
                "eval.gate", "Eval gate",
                f"{metric} = {value:.4f} does not clear threshold {self.config.threshold:.4f}. "
                f"Expected {'>=' if higher_is_better else '<='}.",
                meta=result.to_dict(),
            ))
        return report

    def infer_higher_is_better(self, metric: str) -> bool:
        """Heuristic: whether ``metric`` should be treated as higher-is-better."""
        return metric not in _LOWER_IS_BETTER and metric.lower().startswith(("acc", "score", "f1"))
