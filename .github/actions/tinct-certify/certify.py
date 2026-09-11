#!/usr/bin/env python3
"""
tinct-certify — GitHub Action entry point.

Runs ``tinct certify`` on a LoRA adapter, renders the gate results into the
workflow job summary, upserts a PR comment, and **exits with the verdict** so a
DON'T SHIP turns the check red. Fail-closed by construction: the job fails
unless the evidence bundle says SHIP.

Stdlib only — tinct itself is installed by the action, not by this script.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

MARKER = "<!-- tinct-certify -->"

GATE_ORDER = [
    ("canary_leakage", "Canary Leakage"),
    ("refusal_regression", "Refusal Regression"),
    ("toxicity", "Toxicity"),
    ("expert_collapse", "Expert Collapse (MoE)"),
    ("routing_regression", "Routing Regression (MoE)"),
]

_STATUS_ICON = {"PASS": "✅", "FAIL": "❌", "NOT_CONFIGURED": "➖"}


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def new_run_id() -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"cert_{stamp}"


def gate_detail(key: str, gate: dict[str, Any]) -> str:
    if not isinstance(gate, dict):
        return "—"
    try:
        if key == "canary_leakage":
            return f"{gate['leakage_rate'] * 100:.1f}% leaked"
        if key == "refusal_regression":
            return f"{gate['regression_rate'] * 100:+.1f}% delta"
        if key == "toxicity":
            return f"{gate['increase_factor']}x vs base"
        if key == "expert_collapse":
            return f"min util {gate['min_utilization'] * 100:.1f}%"
        if key == "routing_regression":
            return f"{len(gate.get('regressed_experts') or [])} regressed experts"
    except (KeyError, TypeError, ValueError):
        return "—"
    return gate.get("status", "—")


def adapter_label(bundle: dict[str, Any]) -> str:
    """Adapter path from the bundle. Note `artifacts.adapter` is a dir *hash*,
    the path lives under `artifacts.adapter_sha256.path` (or config.adapter)."""
    artifacts = bundle.get("artifacts") or {}
    candidates = [
        (artifacts.get("adapter_sha256") or {}).get("path"),
        (bundle.get("config") or {}).get("adapter"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown"


def render_summary(decision: str, bundle: dict[str, Any], run_id: str,
                   exit_code: int) -> str:
    gates = bundle.get("safety_gates") or {}
    offload = gates.get("offload_stats") if isinstance(gates, dict) else None
    verdict_icon = "🚢" if decision == "SHIP" else "🛑"
    lines = [
        MARKER,
        f"## {verdict_icon} tinct certification: **{decision.replace('_', ' ')}**",
        "",
        f"Base model `{bundle.get('model', 'unknown')}` · adapter `{adapter_label(bundle)}`",
        "",
    ]

    if gates:
        lines += ["| Gate | Result | Measured |", "|---|---|---|"]
        for key, title in GATE_ORDER:
            gate = gates.get(key)
            if gate is None:
                lines.append(f"| {title} | ➖ | not run |")
                continue
            status = gate.get("status", "UNKNOWN")
            lines.append(f"| {title} | {_STATUS_ICON.get(status, '❔')} {status} | {gate_detail(key, gate)} |")
        if isinstance(offload, dict):
            lines.append(
                f"| MoEStreamer | ✅ ran | {offload.get('cache_hits', 0):,} cache hits · "
                f"{offload.get('h2d_streams', 0)} H2D streams |"
            )
        lines.append("")
    elif bundle and truthy(env("INPUT_SKIP_SAFETY")):
        lines += ["> Safety gates were skipped (`skip-safety: true`).", ""]
    else:
        lines += ["> No gates were recorded — the certification did not complete.", ""]

    adapter_hash = (bundle.get("artifacts") or {}).get("adapter_sha256", {}).get("sha256")
    if adapter_hash:
        lines += [f"Adapter sha256: `{adapter_hash}`", ""]
    if not bundle:
        lines += [
            f"> ⚠️ No evidence bundle was produced (certify exited {exit_code}). "
            "See the job log for the failure reason.",
            "",
        ]
    lines.append(
        "<sub>Signed with Ed25519 · verify with `tinct security check` · "
        "view locally with the tinct dashboard</sub>"
    )
    return "\n".join(lines)


def write_step_summary(markdown: str) -> None:
    path = env("GITHUB_STEP_SUMMARY")
    if not path:
        print("::notice::GITHUB_STEP_SUMMARY not set; summary not persisted")
        print(markdown)
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(markdown + "\n")


def write_outputs(**values: Any) -> None:
    path = env("GITHUB_OUTPUT")
    if not path:
        for key, value in values.items():
            print(f"{key}={value}")
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def pr_number() -> str | None:
    ref = env("GITHUB_REF")
    if ref.startswith("refs/pull/"):
        return ref.split("/")[2]
    event_path = env("GITHUB_EVENT_PATH")
    if event_path and pathlib.Path(event_path).is_file():
        try:
            event = json.loads(pathlib.Path(event_path).read_text(encoding="utf-8"))
            number = (event.get("pull_request") or {}).get("number") or (event.get("issue") or {}).get("number")
            return str(number) if number else None
        except (OSError, ValueError):
            return None
    return None


def upsert_pr_comment(markdown: str) -> None:
    token = env("INPUT_GITHUB_TOKEN") or env("GH_TOKEN")
    repo = env("GITHUB_REPOSITORY")
    pr = pr_number()
    if not token:
        print("::notice::no github-token provided; skipping PR comment")
        return
    if not (repo and pr):
        print("::notice::not a pull-request context; skipping PR comment")
        return

    gh_env = {**os.environ, "GH_TOKEN": token}
    body_file = pathlib.Path("tinct_comment.md")
    body_file.write_text(markdown, encoding="utf-8")

    try:
        listing = subprocess.run(
            ["gh", "api", f"repos/{repo}/issues/{pr}/comments", "--paginate",
             "--jq", f'[.[] | select(.body | contains("{MARKER}")) | .id] | first'],
            capture_output=True, text=True, env=gh_env, timeout=60,
        )
        existing = listing.stdout.strip()
        if listing.returncode == 0 and existing and existing != "null":
            subprocess.run(
                ["gh", "api", "-X", "PATCH", f"repos/{repo}/issues/comments/{existing}",
                 "-F", f"body=@{body_file}"],
                check=True, capture_output=True, text=True, env=gh_env, timeout=60,
            )
            print(f"::notice::updated existing PR comment {existing}")
        else:
            subprocess.run(
                ["gh", "api", "-X", "POST", f"repos/{repo}/issues/{pr}/comments",
                 "-F", f"body=@{body_file}"],
                check=True, capture_output=True, text=True, env=gh_env, timeout=60,
            )
            print("::notice::posted PR comment")
    except (subprocess.SubprocessError, OSError) as exc:
        # A missing comment must never mask the certification verdict.
        print(f"::warning::could not post PR comment: {exc}")


def main() -> int:
    adapter = env("INPUT_ADAPTER")
    base_model = env("INPUT_BASE_MODEL")
    root = env("INPUT_ROOT") or "."
    if not adapter or not base_model:
        print("::error::the 'adapter' and 'base-model' inputs are required")
        return 1

    run_id = env("INPUT_RUN_ID") or new_run_id()
    cmd = [
        sys.executable, "-m", "tinct", "certify",
        "--adapter", adapter,
        "--base-model", base_model,
        "--root", root,
        "--run-id", run_id,
    ]
    if truthy(env("INPUT_SKIP_SAFETY")):
        cmd.append("--skip-safety")
    if env("INPUT_DATASET"):
        cmd += ["--dataset", env("INPUT_DATASET")]
    if env("INPUT_CANARIES"):
        cmd += ["--canaries", env("INPUT_CANARIES")]
    if truthy(env("INPUT_OFFLOAD_EXPERTS")):
        cmd.append("--offload-experts")

    print(f"::group::tinct {' '.join(cmd[3:])}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print("::endgroup::")

    evidence_path = pathlib.Path(root).resolve() / ".tinct" / "evidence" / f"{run_id}_evidence.json"
    bundle: dict[str, Any] = {}
    if evidence_path.is_file():
        try:
            bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(f"::warning::evidence bundle is not valid JSON: {exc}")
    else:
        print(f"::warning::no evidence bundle at {evidence_path}")

    decision = str(bundle.get("decision", "UNKNOWN")).replace("_", " ")
    evidence_dir = pathlib.Path(root).resolve() / ".tinct" / "evidence"
    write_outputs(
        verdict=decision,
        run_id=run_id,
        evidence_path=evidence_dir / f"{run_id}_evidence.json",
        evidence_dir=evidence_dir,
        exit_code=proc.returncode,
    )

    # Rendering is cosmetic: a bug here must NEVER change the verdict that the
    # workflow sees, so each reporting step is contained.
    try:
        markdown = render_summary(
            str(bundle.get("decision", "UNKNOWN")), bundle, run_id, proc.returncode
        )
    except Exception as exc:  # noqa: BLE001 - reporting must not mask the verdict
        print(f"::warning::could not render summary: {exc!r}")
        markdown = (
            f"{MARKER}\n## tinct certification: **{decision}**\n\n"
            "_(summary rendering failed; see the job log)_"
        )

    try:
        write_step_summary(markdown)
    except OSError as exc:
        print(f"::warning::could not write job summary: {exc}")

    if not truthy(env("INPUT_COMMENT")) or env("INPUT_COMMENT") == "false":
        print("::notice::comment input disabled; skipping PR comment")
    else:
        upsert_pr_comment(markdown)

    # Fail-closed: propagate tinct's exit code (0 SHIP, 2 DON'T SHIP, 1 error).
    if proc.returncode != 0:
        print(f"::error::tinct certify exited {proc.returncode} — verdict: {decision}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
