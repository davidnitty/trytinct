"""Console rendering helpers for tinct CLI output (via rich)."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.table import Table

from tinct.core.rules import RuleReport, Severity

_VERDICT = {"passed": "[green]PASS[/]", "failed": "[bold red]FAIL[/]"}


def print_report(console: Console, report: RuleReport) -> None:
    """Render a RuleReport as a table plus an overall verdict."""
    table = Table(title=report.title, expand=False)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Severity")
    table.add_column("Result")
    table.add_column("Message")

    for r in report.results:
        sev_color = {"error": "red", "warning": "yellow", "info": "blue"}[r.severity.value]
        result = _VERDICT["passed"] if r.passed else _VERDICT["failed"]
        table.add_row(
            r.rule_id,
            r.name,
            f"[{sev_color}]{r.severity.value}[/]",
            result,
            r.message,
        )

    console.print(table)
    if report.passed:
        console.print(f"[bold green]{report.title}: PASS[/]")
    else:
        console.print(f"[bold red]{report.title}: FAIL[/]")
        console.print("[red]Failures block the pipeline (fail-closed).[/]")


def print_decision(console: Console, decision: str) -> None:
    if decision == "SHIP":
        console.print("\n[bold green]Decision: SHIP[/]")
    else:
        console.print("\n[bold red]Decision: DON'T_SHIP[/]")


def print_rule_message(console: Console, msg: str, style: Optional[str] = None) -> None:
    console.print(msg, style=style)
