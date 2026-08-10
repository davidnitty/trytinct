"""``tinct eval`` — gate a trained checkpoint against thresholds."""

from __future__ import annotations

import json
from pathlib import Path

from tinct.cli.render import print_report
from tinct.engine.deps import MissingDependencyError
from tinct.evals.gate import EvalGate
from tinct.evals.harness import LlamaEvalHarness
from tinct.utils.logging import get_console


def _load_history(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return None


def _load_valid_records(run_dir: Path):
    valid_path = run_dir / "valid.jsonl"
    if not valid_path.is_file():
        return []
    out = []
    for line in valid_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def resolve_run(project, run_name: str | None) -> Path | None:
    if run_name:
        run_dir = project.run_dir(run_name)
        return run_dir if run_dir.is_dir() else None
    return project.latest_run_dir()


def run_eval(project, run_name: str | None) -> int:
    console = get_console()
    run_dir = resolve_run(project, run_name)
    if run_dir is None:
        console.print("[bold red]No run found to evaluate.[/] Run `tinct train` first.")
        return 1

    eval_cfg = project.config.eval
    gate = EvalGate(eval_cfg)
    history = _load_history(run_dir)

    # If the run history already recorded the metric, gate from it.
    report = gate.evaluate(history or [])
    result = None
    if not report.failed_errors and report.results:
        result = report.results[-1].meta if report.results[-1].meta else None

    # Otherwise (no eval step during training), run the harness on the adapter.
    if result is None:
        valid = _load_valid_records(run_dir)
        adapter_dir = run_dir / "adapter"
        if not adapter_dir.is_dir():
            console.print("[bold red]No adapter and no in-run eval metric to gate on.[/]")
            console.print("  Re-run `tinct train` with an eval split, or pass a metrics file.")
            return 1
        if not valid:
            console.print("[bold red]No validation records found to evaluate.[/]")
            return 1
        try:
            harness = LlamaEvalHarness()
            result = harness.run(
                adapter_dir, project.config.train.model, valid,
                project.config.train, eval_cfg,
            )
            report = gate.evaluate(history or [], override_value=result.value)
        except MissingDependencyError as exc:
            console.print(f"[bold red]Cannot evaluate:[/] {str(exc).replace('[', '\\\\[')}")
            return 3

    print_report(console, report)
    console.print(f"\n  run: {run_dir.name}")
    console.print(f"  adapter: {run_dir / 'adapter'}")
    if not report.passed:
        console.print("[bold red]Checkpoint did not pass the gate; it must not ship.[/]")
        return 1
    console.print("[bold green]Checkpoint passed the gate.[/]\nNext: `tinct ship --run {run_dir.name}`")
    return 0
