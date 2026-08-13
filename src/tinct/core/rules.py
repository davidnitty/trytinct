"""Rule engine — the shared pass/fail machinery used by the Data Doctor, the
eval gate, and security checks.

Every check produces a :class:`RuleResult`. The engine computes an overall
verdict in a *fail-closed* way: any failing ``error``-severity rule fails the
whole run. Warnings and info never fail a run but are surfaced to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - pydantic is a hard dependency, guard anyway
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    BaseModel = None  # type: ignore[assignment]


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class RuleResult:
    """The outcome of a single check."""

    rule_id: str
    name: str
    severity: Severity
    passed: bool
    message: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "meta": dict(self.meta),
        }


class RuleReport:
    """Accumulates :class:`RuleResult` objects and computes an overall verdict.

    Verdict rules (fail-closed):
    - If any error-severity rule failed  -> FAIL.
    - Else if any error-severity rule is still running/pending -> FAIL.
    - Otherwise -> PASS (warnings/info never fail the run).
    """

    def __init__(self, title: str) -> None:
        self.title = title
        self.results: List[RuleResult] = []

    def add(self, result: RuleResult) -> RuleResult:
        self.results.append(result)
        return result

    @property
    def failed_errors(self) -> List[RuleResult]:
        return [
            r for r in self.results
            if r.severity == Severity.ERROR and not r.passed
        ]

    @property
    def warnings(self) -> List[RuleResult]:
        return [r for r in self.results if r.severity == Severity.WARNING]

    @property
    def infos(self) -> List[RuleResult]:
        return [r for r in self.results if r.severity == Severity.INFO]

    @property
    def passed(self) -> bool:
        """Fail-closed: any failed error rule means the whole run fails."""
        return not self.failed_errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "passed": self.passed,
            "summary": {
                "total": len(self.results),
                "errors_failed": len(self.failed_errors),
                "warnings": len(self.warnings),
                "infos": len(self.infos),
            },
            "results": [r.to_dict() for r in self.results],
        }


# Convenience constructors ----------------------------------------------------

def ok(rule_id: str, name: str, message: str = "OK", severity: Severity = Severity.INFO,
       meta: Optional[Dict[str, Any]] = None) -> RuleResult:
    return RuleResult(rule_id, name, severity, True, message, meta or {})


def error(rule_id: str, name: str, message: str,
          meta: Optional[Dict[str, Any]] = None) -> RuleResult:
    return RuleResult(rule_id, name, Severity.ERROR, False, message, meta or {})


def warn(rule_id: str, name: str, message: str,
         meta: Optional[Dict[str, Any]] = None) -> RuleResult:
    return RuleResult(rule_id, name, Severity.WARNING, True, message, meta or {})


def info(rule_id: str, name: str, message: str,
         meta: Optional[Dict[str, Any]] = None) -> RuleResult:
    return RuleResult(rule_id, name, Severity.INFO, True, message, meta or {})


def as_dict(self: RuleReport) -> Dict[str, Any]:
    """Small shim so a report can be embedded in structured output."""
    return self.to_dict()


# Optional structured model for external consumers -----------------------------

if BaseModel is not None:  # pragma: no cover

    class RuleReportModel(BaseModel):
        """Pydantic view of a report, useful for serializing to project.yaml."""

        title: str
        passed: bool
        summary: Dict[str, int]
        results: List[Dict[str, Any]]
