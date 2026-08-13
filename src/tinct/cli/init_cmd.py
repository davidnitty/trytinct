"""``tinct init`` — scaffold a new tinct project."""

from __future__ import annotations

from pathlib import Path

from tinct.core.init_project import init_tinct_project
from tinct.security.signing import SigningKey
from tinct.utils.logging import get_console


def run_init(project_name: str, model: str, root: Path, generate_key: bool) -> None:
    console = get_console()
    project = init_tinct_project(root / project_name, model, project_name=project_name)
    console.print(f"[bold green]Initialized project[/] {project_name!r} at {project.root}")
    console.print(f"  model:   {model}")
    console.print(f"  config:  {project.config_path}")

    if generate_key:
        key = SigningKey.generate(project.config.security.key_name)
        priv, pub = key.save(project.keys_dir)
        console.print(f"  signing key: {priv.name} (private) + {pub.name} (public)")
        console.print("  Keep the private key safe; it never leaves this project.")
    else:
        console.print("  [yellow]No signing key generated (use `tinct security key generate`).[/]")

    console.print("\nNext steps:")
    console.print("  tinct validate data.jsonl")
    console.print("  tinct train   ")
    console.print("  tinct eval    ")
    console.print("  tinct ship    ")
