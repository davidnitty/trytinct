"""Tests for the rule engine (fail-closed semantics)."""

from tinct.core.rules import RuleReport, Severity, error, info, ok, warn


def test_empty_report_passes():
    report = RuleReport("x")
    assert report.passed


def test_error_fails_run():
    report = RuleReport("x")
    report.add(error("b", "B", "boom"))
    assert not report.passed
    assert len(report.failed_errors) == 1


def test_warning_does_not_fail_run():
    report = RuleReport("x")
    report.add(warn("w", "W", "heads up"))
    assert report.passed


def test_infos_and_warnings_counting():
    report = RuleReport("x")
    report.add(info("i1", "I", "a"))
    report.add(info("i2", "I", "b"))
    report.add(warn("w", "W", "c"))
    assert report.passed
    assert len(report.infos) == 2
    assert len(report.warnings) == 1


def test_to_dict_shape():
    report = RuleReport("t")
    report.add(error("e", "E", "msg", meta={"k": 1}))
    d = report.to_dict()
    assert d["title"] == "t"
    assert d["passed"] is False
    assert d["results"][0]["rule_id"] == "e"
    assert d["results"][0]["severity"] == "error"
