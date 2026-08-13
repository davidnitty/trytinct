"""``tinct eval`` — gate a trained checkpoint against thresholds."""

from __future__ import annotations

import json
from pathlib import Path

from tinct.cli.render import print_report
from tinct.engine.deps import MissingDependencyError
from tinct.evals.gate import EvalGate
from tinct.utils.logging import get_console


def _load_history(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return None


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

    # Fail-closed: a run halted by the training guard can never be evaluated
    # as healthy — it must not ship.
    if (run_dir / "fail_state.json").exists():
        console.print("\n[tinct] VERDICT: DON'T SHIP")
        console.print("[tinct] Reason: Training aborted by fail-closed guard.\n")
        return 2

    adapter_dir = run_dir / "adapter"

    # 1. Loss gate (secondary/informational): only applies when an eval metric
    # was actually recorded during training. With no eval split, the generation
    # smoke test below is the authoritative gate.
    eval_cfg = project.config.eval
    gate = EvalGate(eval_cfg)
    history = _load_history(run_dir) or []
    metric_recorded = any(
        isinstance(e, dict) and eval_cfg.metric in e and e[eval_cfg.metric] is not None
        for e in history
    )
    report = gate.evaluate(history)
    print_report(console, report)

    # 2. Generation smoke test: proves the adapter generates non-empty,
    # non-repetitive text. This is the authoritative `tinct eval` gate.
    if not adapter_dir.is_dir():
        console.print("[bold red]Cannot run generation smoke test: no adapter found.[/]")
        console.print("  Re-run `tinct train` so the run produces an adapter.")
        return 1

    try:
        from tinct.evals.smoke_test import run_generation_smoke_test
        eval_report_path = run_dir / "eval_report.json"
        smoke_pass = run_generation_smoke_test(
            project.config.train.model,
            adapter_dir,
            eval_report_path,
        )
    except MissingDependencyError as exc:
        console.print(f"[bold red]Cannot evaluate:[/] {str(exc).replace('[', '\\\\[')}")
        return 3

    console.print(f"\n  run: {run_dir.name}")
    console.print(f"  adapter: {adapter_dir}")
    console.print(f"  eval report: {run_dir / 'eval_report.json'}")
    if not smoke_pass:
        console.print("\n[tinct] VERDICT: DON'T SHIP")
        console.print("[tinct] Reason: Generation smoke test failed.\n")
        return 2
    if metric_recorded and not report.passed:
        console.print("[bold red]Recorded loss gate did not pass; checkpoint must not ship.[/]")
        return 2
    console.print("\n[tinct] VERDICT: READY TO SHIP")
    console.print("Next: `tinct ship --run {run_dir.name}` to certify.\n")
    return 0
