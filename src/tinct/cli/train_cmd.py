"""``tinct train`` — validate then fine-tune a Llama adapter (LoRA/QLoRA)."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from tinct.cli.render import print_report
from tinct.core.datadoctor import DataDoctor, DatasetLoadError
from tinct.core.project import Project
from tinct.engine.deps import MissingDependencyError, ensure_train_deps
from tinct.engine.hf_trainer import HfLlamaTrainer
from tinct.utils.logging import get_console


def _default_run_name() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def _escape(text: str) -> str:
    """Escape rich markup so optional-extra brackets like ``[train]`` show up."""
    return text.replace("[", "\\[")


def _write_jsonl(records, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_train(project: Project, dataset: Path, run_name: str | None,
              model_override: str | None) -> int:
    """Validate, split, train. Returns 0 on success, non-zero otherwise."""
    console = get_console()

    # Fail-closed 1: unsupported model family.
    if model_override:
        project.config.train.model = model_override
    try:
        project.refuse_if_unsupported_model()
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        return 2

    # Fail-closed 2: data must pass the Data Doctor.
    doctor = DataDoctor(
        project.config.data,
        max_seq_len=project.config.train.max_seq_len,
        seed=project.config.train.seed,
    )
    try:
        report, records = doctor.run(dataset)
    except DatasetLoadError as exc:
        console.print(f"[bold red]Could not load dataset:[/] {exc}")
        return 2
    print_report(console, report)
    if not report.passed:
        console.print("[bold red]Training blocked: data validation failed (fail-closed).[/]")
        return 1

    train_records, valid_records = doctor.split(records)
    console.print(f"[bold]Split:[/] {len(train_records)} train / {len(valid_records)} valid")

    train_cfg = project.config.train
    # Check heavy deps BEFORE scaffolding a run dir (fail fast, no partial run).
    try:
        ensure_train_deps(quant=train_cfg.quant)
    except MissingDependencyError as exc:
        console.print(f"[bold red]Cannot train:[/] {_escape(str(exc))}")
        return 3

    name = run_name or _default_run_name()
    run_dir = project.create_run(name)
    train_p, valid_p = run_dir / "train.jsonl", run_dir / "valid.jsonl"
    _write_jsonl(train_records, train_p)
    _write_jsonl(valid_records, valid_p)
    console.print(f"[green]Run {name!r} created at[/] {run_dir}")

    engine = HfLlamaTrainer()
    result = engine.train(
        run_name=name,
        run_dir=run_dir,
        model=train_cfg.model,
        train_records=train_records,
        valid_records=valid_records,
        config=train_cfg,
    )

    console.print("[bold green]Training complete.[/]")
    console.print(f"  adapter: {result.adapter_dir}")
    console.print(f"  metrics: {result.metrics_path}")
    console.print(f"\nNext: run `tinct eval --run {name}` to gate the checkpoint.")
    return 0
