#!/usr/bin/env python
"""Verify a tinct run's evidence: Ed25519 signature + artifact hash coverage.

Usage (on the GPU box, or anywhere evidence exists):
    python scripts/verify_evidence.py <run_id> [--root <project_dir>]

Exits 0 only when: decision is SHIP, signature verifies, and every required
artifact (training/validation data, config, adapter, adapter_sha256,
eval_report) is present in the manifest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_ARTIFACTS = [
    "train_data.jsonl",
    "valid_data.jsonl",
    "config",
    "adapter",
    "adapter_sha256",
    "eval_report.json",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id", help="The run ID to verify (e.g. run_20260813_232318).")
    ap.add_argument("--root", default=".",
                    help="Project root that contains .tinct/ (default: cwd).")
    args = ap.parse_args()

    from tinct.security.evidence import EvidenceReport
    from tinct.storage.paths import TinctPaths

    ev_path = TinctPaths(args.root).evidence_dir / f"{args.run_id}_evidence.json"
    if not ev_path.is_file():
        print(f"No evidence file for run {args.run_id!r}: {ev_path}")
        return 1

    rep = EvidenceReport.load(ev_path)

    print(f"run:            {args.run_id}")
    print(f"model:          {rep.model}")
    print(f"decision:       {rep.decision}")

    sig_ok = rep.verify()
    print(f"signature:      {('VALID' if sig_ok else 'INVALID')}")

    artifacts = rep.artifacts or {}
    missing = [k for k in REQUIRED_ARTIFACTS if k not in artifacts]
    print(f"hash coverage:  {'ALL REQUIRED PRESENT' if not missing else 'MISSING ' + ', '.join(missing)}")
    for k in REQUIRED_ARTIFACTS:
        v = artifacts.get(k)
        if isinstance(v, dict):
            label = v.get("sha256", "(dir-hash)")
        else:
            label = "(present)"
        print(f"    - {k}: {label}")

    eval_status = (rep.eval_report or {}).get("status")
    print(f"eval_report:    status={eval_status}")

    print()
    ok = bool(sig_ok and not missing and rep.decision == "SHIP" and eval_status == "PASS")
    print("RESULT:", "PASS — production-verified SHIP with signed, complete evidence"
          if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
