"""``tinct validate`` — run the Data Doctor over an instruction dataset."""

from __future__ import annotations

from pathlib import Path

from tinct.cli.render import print_report
from tinct.core.datadoctor import DataDoctor, DatasetLoadError, family_for_model
from tinct.utils.logging import get_console


def _doctor_for(project, model_override: str | None = None) -> DataDoctor:
    """Build a DataDoctor, resolving the model family from the override (if any)
    or the project's configured model."""
    model = model_override or project.config.train.model
    return DataDoctor(
        project.config.data,
        max_seq_len=project.config.train.max_seq_len,
        seed=project.config.train.seed,
        model_family=family_for_model(model),
    )


def run_validate(project, dataset: Path, model_override: str | None = None) -> bool:
    """Validate ``dataset``; return True if the report passes (fail-closed)."""
    console = get_console()
    doctor = _doctor_for(project, model_override)
    try:
        report, _records = doctor.run(dataset)
    except DatasetLoadError as exc:
        console.print(f"[bold red]Could not load dataset:[/] {exc}")
        return False

    print_report(console, report)
    return report.passed


def run_advise(project, dataset: Path) -> None:
    """Summarize validation and recommend a post-training method."""
    console = get_console()
    doctor = _doctor_for(project)
    report, records = doctor.run(dataset)
    print_report(console, report)
    if not report.passed:
        console.print("[red]Recommendation: fix the data. Training is blocked until it passes.[/]")
        return

    n = len(records)
    quant = project.config.train.quant
    console.print("\n[bold]Post-training recommendation[/]")
    if n < 1000:
        console.print(f"  Dataset is small ({n} rows) → supervised fine-tuning (SFT) with LoRA/{quant}.")
        console.print("  Prefer quick iteration; add evals before scaling data.")
    else:
        console.print(f"  Dataset is substantial ({n} rows) → SFT with LoRA/{quant}, then DPO/GRPO later.")
    console.print(f"  Model: {project.config.train.model} ({project.config.train.method})")
