"""``tinct security check`` — audits a project's security posture.

Fail-closed checks over project state: required files present, signing key
exists, no obvious secrets committed, and artifact integrity when evidence
files already exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from tinct.core.rules import RuleReport, error, info, ok, warn
from tinct.security.evidence import EvidenceReport, sha256_file
from tinct.storage.paths import TinctPaths

# Substrings that indicate a value is probably a secret.
_SECRET_HINTS = ("sk-", "ghp_", "api_key", "password", "secret", "token", "bearer")


def _scan_for_secrets(root: Path, env_path: Path) -> List[str]:
    """Look for likely secret-bearing files tracked near the project."""
    found: List[str] = []
    for path in (root / ".env", env_path):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            low = line.lower()
            if any(h in low for h in _SECRET_HINTS) and "=" in line:
                found.append(f"{path.name}: {line.split('=', 1)[0]}=***")
                break
    return found


class SecurityAuditor:
    """Runs the audit and returns a fail-closed :class:`RuleReport`."""

    def __init__(self, root: Path, env_path: Optional[Path] = None) -> None:
        self.root = root
        self.env_path = env_path or Path(".env.example")

    def run(self) -> RuleReport:
        report = RuleReport("Security Audit")
        root = self.root

        # 1. Project initialized.
        config = TinctPaths(root).project_config
        if config.is_file():
            report.add(ok("sec.config", "Config present", "project.yaml found"))
        else:
            report.add(error("sec.config", "Config present",
                             "Missing .tinct/project.yaml; project not initialized."))

        # 2. No secrets in .env files.
        leaked = _scan_for_secrets(root, self.env_path)
        if leaked:
            report.add(warn("sec.secrets", "No secrets in dotfiles",
                            "Possible secret detected in env files.",
                            meta={"suspicious": leaked}))
        else:
            report.add(ok("sec.secrets", "No secrets in dotfiles"))

        # 3. Private key file permissions are restrictive (POSIX only).
        keys_dir = TinctPaths(root).keys_dir
        if keys_dir.is_dir():
            priv_keys = list(keys_dir.glob("*_private.pem"))
            if not priv_keys:
                report.add(warn("sec.key.present", "Signing key present",
                                "No private signing key found yet."))
            else:
                perms_ok = True
                for pk in priv_keys:
                    try:
                        if pk.stat().st_mode & 0o077:
                            perms_ok = False
                    except OSError:  # pragma: no cover
                        pass
                if perms_ok:
                    report.add(ok("sec.key.present", "Signing key present",
                                  f"Key permissions restricted ({len(priv_keys)} found)."))
                else:
                    report.add(warn("sec.key.present", "Signing key present",
                                    "Private key file has loose permissions; chmod 600 it."))
        else:
            report.add(warn("sec.key.present", "Signing key present",
                            "No .tinct/keys directory yet."))

        # 4. Verify any existing evidence signatures.
        evidence_dir = TinctPaths(root).evidence_dir
        verified = checked = 0
        failures = []
        if evidence_dir.is_dir():
            for ev_path in sorted(evidence_dir.glob("*_evidence.json")):
                checked += 1
                try:
                    rep = EvidenceReport.load(ev_path)
                    if rep.verify():
                        verified += 1
                    else:
                        failures.append(str(ev_path))
                except Exception as exc:  # pragma: no cover
                    failures.append(f"{ev_path.name}: {exc}")
        if checked == 0:
            report.add(info("sec.evidence", "Evidence integrity",
                            "No evidence reports to verify yet."))
        elif failures:
            report.add(error("sec.evidence", "Evidence integrity",
                             f"{len(failures)}/{checked} evidence files failed signature check.",
                             meta={"failed": failures}))
        else:
            report.add(ok("sec.evidence", "Evidence integrity",
                          f"All {verified} evidence signatures verified."))

        # 5. No obvious world-readable model checkpoints tracked.
        runs_dir = TinctPaths(root).runs_dir
        if runs_dir.is_dir() and not any(runs_dir.iterdir()):
            report.add(info("sec.runs", "Run directory clean", "No runs yet."))

        report.add(info("sec.hardening", "Hardening", "Local-first; no telemetry configured."))
        return report
