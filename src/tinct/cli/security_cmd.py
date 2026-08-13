"""``tinct security`` — audit posture and manage signing keys.

Command tree::

    tinct security check
    tinct security key generate --name ship
"""

from __future__ import annotations

from pathlib import Path

import typer

from tinct.cli.render import print_report
from tinct.core.rules import RuleReport, ok
from tinct.security.checks import SecurityAuditor
from tinct.security.signing import SigningKey
from tinct.utils.logging import get_console

security_app = typer.Typer(
    name="security",
    help="Audit project security and manage ship signing keys.",
    no_args_is_help=True,
)

key_app = typer.Typer(
    name="key",
    help="Manage Ed25519 signing keys.",
    no_args_is_help=True,
)
security_app.add_typer(key_app, name="key")


@security_app.command("check")
def security_check(
    root: Path = typer.Option(".", "--root", help="Project root."),
    run: str | None = typer.Option(None, "--run", help="Verify the evidence signature for a specific run."),
) -> None:
    """Audit the project's security posture (fail-closed)."""
    console = get_console()
    auditor = SecurityAuditor(root, env_path=root / ".env.example")
    report = auditor.run()
    print_report(console, report)

    # Explicit per-run evidence verification: `security check --run <id>`.
    if run:
        from tinct.security.evidence import EvidenceReport
        from tinct.storage.paths import TinctPaths
        ev_path = TinctPaths(root).evidence_dir / f"{run}_evidence.json"
        if not ev_path.is_file():
            console.print(f"[red]No evidence file for run {run!r}: {ev_path}[/]")
            raise typer.Exit(code=1)
        rep = EvidenceReport.load(ev_path)
        if rep.verify():
            console.print(f"[bold green]Signature valid. Manifest verified for run {run!r}.[/]")
        else:
            console.print(f"[bold red]Signature INVALID for run {run!r}.[/]")
            raise typer.Exit(code=1)



@key_app.command("generate")
def key_generate(
    name: str = typer.Option("ship", "--name", help="Key name."),
    root: Path = typer.Option(".", "--root", help="Project root."),
) -> None:
    """Generate an Ed25519 signing keypair under .tinct/keys."""
    console = get_console()
    keys_dir = root / ".tinct" / "keys"
    key = SigningKey.generate(name)
    priv, pub = key.save(keys_dir)
    report = RuleReport("Key generation")
    report.add(ok("key.generated", "Signing key", f"Created key {name!r}"))
    print_report(console, report)
    console.print(f"  private: {priv}")
    console.print(f"  public:  {pub}")
