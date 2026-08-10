"""``tinct ship`` — compute the SHIP / DON'T-SHIP decision + signed evidence."""

from __future__ import annotations

from pathlib import Path

from tinct.cli.eval_cmd import resolve_run
from tinct.cli.render import print_decision
from tinct.core.datadoctor import DataDoctor
from tinct.core.project import Project
from tinct.evals.gate import EvalGate
from tinct.security.evidence import EvidenceReport, dump_json, hash_path
from tinct.security.signing import SigningKey
from tinct.utils.logging import get_console


def _load_history(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        import json
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return []


def _data_report(project: Project, run_dir: Path) -> dict:
    doctor = DataDoctor(
        project.config.data,
        max_seq_len=project.config.train.max_seq_len,
        seed=project.config.train.seed,
    )
    report, _ = doctor.run(run_dir / "train.jsonl")
    return report.to_dict()


def run_ship(project: Project, run_name: str | None) -> int:
    console = get_console()
    run_dir = resolve_run(project, run_name)
    if run_dir is None:
        console.print("[bold red]No run found to ship.[/] Run `tinct train` first.")
        return 1

    history = _load_history(run_dir)
    eval_cfg = project.config.eval
    gate = EvalGate(eval_cfg)
    eval_report = gate.evaluate(history)
    eval_pass = eval_report.passed
    print_decision(console, "SHIP" if eval_pass else "DON'T_SHIP")

    if not eval_pass:
        console.print("[bold red]Reason: the eval gate failed (fail-closed).[/]")

    data_report = _data_report(project, run_dir)

    artifacts = {
        "train_data.jsonl": hash_path(run_dir / "train.jsonl"),
        "valid_data.jsonl": hash_path(run_dir / "valid.jsonl"),
        "config": hash_path(project.config_path),
    }
    adapter_dir = run_dir / "adapter"
    if adapter_dir.is_dir():
        artifacts["adapter"] = hash_path(adapter_dir)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        artifacts["metrics.json"] = hash_path(metrics_path)

    report = EvidenceReport(
        project_name=project.config.project_name,
        model=project.config.train.model,
        family="llama",
        decision="SHIP" if eval_pass else "DON'T_SHIP",
        artifacts=artifacts,
        data_report=data_report,
        eval_report=eval_report.to_dict(),
        metrics={"n_runs": 1},
        config=project.config.model_dump(mode="json"),
    )

    # Cryptographic evidence is required to ship (secure by default).
    if report.decision == "SHIP" and project.config.security.sign_evidence:
        try:
            key = SigningKey.load(project.keys_dir, project.config.security.key_name)
        except FileNotFoundError as exc:
            console.print(f"[bold red]Cannot ship:[/] {exc}")
            console.print("  Run `tinct security key generate --name "
                          f"{project.config.security.key_name}` first.")
            return 1
        report.sign(key)
        ok = report.verify()
        if not ok:
            console.print("[bold red]Evidence signature verification failed; refusing to ship.[/]")
            return 1
        console.print("[green]Evidence signed and signature verified.[/]")
    elif report.decision != "SHIP":
        console.print("[yellow]Produced unsigned evidence for a failed gate (no ship).[/]")

    path = report.write(project.evidence_dir, run_dir.name)
    console.print(f"[bold green]Evidence report:[/] {path}")
    console.print(f"  decision: {report.decision}")
    console.print(f"  gate: {eval_report.passed and 'PASS' or 'FAIL'}")
    return 0 if report.decision == "SHIP" else 1
