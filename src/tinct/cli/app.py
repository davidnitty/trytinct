"""tinct CLI application.

Command tree::

    tinct init <name> <model>
    tinct validate <dataset>
    tinct advise <dataset>
    tinct train  <dataset> [--run NAME] [--model ID]
    tinct eval   [--run NAME]
    tinct ship   [--run NAME]
    tinct security check | key generate
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from tinct import __version__
from tinct.core.project import Project
from tinct.utils.logging import get_console, setup_logging

from . import certify_cmd, doctor_cmd, eval_cmd, init_cmd, ship_cmd, train_cmd, validate_cmd
from .security_cmd import security_app

app = typer.Typer(
    name="tinct",
    help="tinct — CLI-first post-training stack for LLMs.",
    add_completion=False,
    no_args_is_help=True,
    invoke_without_command=True,
)
app.add_typer(security_app)


def _open_project(root: Path) -> Project:
    try:
        return Project.open(root)
    except (FileNotFoundError, ValueError) as exc:
        get_console().print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1) from exc


def _exit(code: int) -> None:
    if code != 0:
        raise typer.Exit(code=code)


@app.callback()
def _entry_callback(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose (debug) logging."),
) -> None:
    """tinct entry point."""
    setup_logging(verbose=verbose)
    if version:
        get_console().print(f"tinct {__version__}")
        # Returning (not raising) lets the click runner print nothing extra.
        raise typer.Exit()


# -- init -------------------------------------------------------------------

@app.command("init")
def init(
    project_name: str = typer.Argument(..., help="New project directory name."),
    model: str = typer.Argument(..., help="HF model id, e.g. meta-llama/Llama-3.1-8B."),
    root: Path = typer.Option(".", "--root", help="Parent directory for the project."),
    no_key: bool = typer.Option(False, "--no-key", help="Skip signing key generation."),
) -> None:
    """Scaffold a new tinct project."""
    init_cmd.run_init(project_name, model, root, generate_key=not no_key)


# -- validate / advise ------------------------------------------------------

@app.command("validate")
def validate(
    dataset: Path = typer.Argument(..., help="JSON/JSONL instruction dataset."),
    root: Path = typer.Option(".", "--root", help="Project root."),
    model: str | None = typer.Option(
        None, "--model",
        help="Model id for family-aware template validation (e.g. Qwen/Qwen2.5-7B-Instruct).",
    ),
) -> None:
    """Validate an instruction dataset with the Data Doctor (fail-closed)."""
    project = _open_project(root)
    _exit(0 if validate_cmd.run_validate(project, dataset, model_override=model) else 1)


@app.command("advise")
def advise(
    dataset: Path = typer.Argument(..., help="JSON/JSONL instruction dataset."),
    root: Path = typer.Option(".", "--root", help="Project root."),
) -> None:
    """Validate and recommend a post-training method."""
    project = _open_project(root)
    validate_cmd.run_advise(project, dataset)


# -- training ---------------------------------------------------------------

@app.command("train")
def train(
    dataset: Path | None = typer.Argument(None, help="JSON/JSONL instruction dataset (or use --dataset)."),
    dataset_opt: Path | None = typer.Option(None, "--dataset", help="Dataset path (alias for the positional arg)."),
    root: Path = typer.Option(".", "--root", help="Project root."),
    run: str | None = typer.Option(None, "--run", help="Run name (defaults to timestamp)."),
    model: str | None = typer.Option(None, "--model", help="Override the configured base model."),
    method: str = typer.Option("sft", "--method", help="Training method: 'sft' or 'dpo' (V0.2)."),
    lora_rank: int | None = typer.Option(None, "--lora-rank", help="Override the LoRA rank."),
    max_loss_threshold: float | None = typer.Option(
        None, "--max-loss-threshold", help="Override the fail-closed loss threshold."
    ),
    accelerator: str = typer.Option(
        "none", "--accelerator",
        help="Acceleration backend: 'none' or 'unsloth' (low-VRAM, needs tinct[unsloth]).",
    ),
) -> None:
    """Validate then fine-tune a Llama adapter (LoRA/QLoRA), fail-closed."""
    dataset_path = dataset_opt or dataset
    if dataset_path is None:
        raise typer.BadParameter("a dataset is required (positional or --dataset)")
    project = _open_project(root)
    _exit(train_cmd.run_train(project, dataset_path, run, model,
                              method=method, lora_rank_override=lora_rank,
                              max_loss_threshold_override=max_loss_threshold,
                              accelerator=accelerator))


@app.command("eval")
def evaluate(
    root: Path = typer.Option(".", "--root", help="Project root."),
    run: str | None = typer.Option(None, "--run", help="Run name (defaults to latest)."),
    safety: bool = typer.Option(
        False, "--safety", help="Run behavioral certification gates (canary leakage + refusal regression)."
    ),
) -> None:
    """Gate a trained checkpoint against thresholds."""
    project = _open_project(root)
    _exit(eval_cmd.run_eval(project, run, safety=safety))


@app.command("ship")
def ship(
    root: Path = typer.Option(".", "--root", help="Project root."),
    run: str | None = typer.Option(None, "--run", help="Run name (defaults to latest)."),
) -> None:
    """Produce the SHIP / DON'T-SHIP decision with signed evidence."""
    project = _open_project(root)
    _exit(ship_cmd.run_ship(project, run))


# -- integration layer --------------------------------------------------------

@app.command("certify")
def certify(
    adapter: Path = typer.Option(..., "--adapter",
                                 help="Path to the trained LoRA adapter directory."),
    base_model: str = typer.Option(..., "--base-model",
                                   help="Base model id the adapter was trained on."),
    root: Path = typer.Option(".", "--root", help="Project/state root (.tinct)."),
    canaries: Path | None = typer.Option(
        None, "--canaries", help="canaries.json for the leakage gate (optional)."
    ),
    dataset: Path | None = typer.Option(
        None, "--dataset", help="Training dataset (optional; hashed into evidence)."
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Custom certification ID (auto-generated if omitted)."
    ),
    skip_safety: bool = typer.Option(
        False, "--skip-safety", help="Skip behavioral safety gates (canary/refusal/toxicity)."
    ),
) -> None:
    """Certify an externally trained adapter: eval gates + signed evidence."""
    _exit(certify_cmd.run_certify(adapter, base_model, root, canaries_path=canaries,
                                  dataset_path=dataset, run_id=run_id,
                                  skip_safety=skip_safety))


# -- preflight ---------------------------------------------------------------

@app.command("doctor")
def doctor(
    root: Path = typer.Option(".", "--root", help="Project/state root (.tinct)."),
) -> None:
    """Preflight: dependencies, GPU, model access, project state."""
    _exit(doctor_cmd.run_doctor(root))


def main() -> None:
    """CLI launcher — invoked by the console script / ``python -m tinct``."""
    # Bound Hugging Face network stalls (gated/unreachable models can hang
    # model loads for many minutes on slow links). Users can still override
    # these by exporting their own values.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")
    app()
