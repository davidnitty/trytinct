"""Tests for the eval gate."""

from tinct.core.config import EvalConfig
from tinct.evals.gate import EvalGate, extract_metric


def _history(*eval_losses):
    out = []
    for step, val in enumerate(eval_losses):
        out.append({"step": step, "loss": 2.0, "eval_loss": val})
    return out


def test_gate_pass_lower_is_better():
    gate = EvalGate(EvalConfig(metric="eval_loss", threshold=1.5, higher_is_better=False))
    report = gate.evaluate(_history(3.0, 1.2))
    assert report.passed


def test_gate_fail_lower_is_better():
    gate = EvalGate(EvalConfig(metric="eval_loss", threshold=1.5, higher_is_better=False))
    report = gate.evaluate(_history(3.0, 2.1))
    assert not report.passed
    assert any(r.rule_id == "eval.gate" for r in report.failed_errors)


def test_gate_uses_last_value():
    gate = EvalGate(EvalConfig(metric="eval_loss", threshold=1.5))
    report = gate.evaluate(_history(3.0, 0.5, 1.2))
    assert report.passed  # last recorded (1.2) <= 1.5


def test_gate_missing_metric_fails():
    gate = EvalGate(EvalConfig(metric="eval_loss", threshold=1.5))
    report = gate.evaluate([])
    assert not report.passed


def test_extract_metric_uses_last():
    assert extract_metric(_history(3.0, 1.0), "eval_loss") == 1.0


def test_higher_is_better():
    gate = EvalGate(EvalConfig(metric="accuracy", threshold=0.8, higher_is_better=True))
    report = gate.evaluate([{"step": 0, "accuracy": 0.9}])
    assert report.passed
    gate2 = EvalGate(EvalConfig(metric="accuracy", threshold=0.8, higher_is_better=True))
    report2 = gate2.evaluate([{"step": 0, "accuracy": 0.7}])
    assert not report2.passed
