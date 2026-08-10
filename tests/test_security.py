"""Tests for the security auditor."""

from pathlib import Path

from tinct.core.project import Project
from tinct.security.checks import SecurityAuditor, _scan_for_secrets
from tinct.security.signing import SigningKey


def test_audit_passes_fresh(project: Project):
    report = SecurityAuditor(project.root).run()
    assert report.passed
    assert any(r.rule_id == "sec.config" for r in report.results)


def test_audit_detects_missing_key(project: Project):
    report = SecurityAuditor(project.root).run()
    assert any(r.rule_id == "sec.key.present" for r in report.results)


def test_secret_scan(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-12345\n")
    found = _scan_for_secrets(tmp_path, env)
    assert found  # detected the suspicious line


def test_audit_verifies_evidence(project: Project):
    key = SigningKey.generate("ship")
    key.save(project.keys_dir)
    # Create a valid signed evidence file.
    from tinct.security.evidence import EvidenceReport
    rep = EvidenceReport(project_name="demo", model="m", family="llama", decision="SHIP")
    rep.sign(key)
    rep.write(project.evidence_dir, "run_1")

    report = SecurityAuditor(project.root).run()
    assert report.passed
    assert any(r.rule_id == "sec.evidence" for r in report.results)
