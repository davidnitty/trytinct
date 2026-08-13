"""``tinct train`` — validate then fine-tune a Llama adapter (LoRA/QLoRA)."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from tinct.cli.render import print_report
from tinct.core.datadoctor import DataDoctor, DatasetLoadError
from tinct.core.project import Project
from tinct.engine.chunking import ModelChunker
from tinct.engine.deps import MissingDependencyError, ensure_train_deps
from tinct.engine.hf_trainer import _to_text
from tinct.storage.paths import get_cache_dir, resolve_model
from tinct.trainers.sft_trainer import run_sft
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


def _materialize_metrics(run_dir: Path) -> None:
    """Normalize the trainer's ``train_log.jsonl`` into ``metrics.json`` so the
    eval gate / ship can read a log history (matches the expected shape)."""
    log_file = run_dir / "train_log.jsonl"
    if not log_file.is_file():
        return
    entries = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    (run_dir / "metrics.json").write_text(
        json.dumps(entries, indent=2), encoding="utf-8"
    )


def prepare_base_model_chunks(
    project: Project, run_dir: Path, model_id: str, chunk_size_mb: int = 500
) -> Path:
    """Resolve the base model, chunk it (AirLLM style), and record the chunk
    hashes into the run's evidence bundle.

    The manifest is persisted as ``.tinct/runs/<name>/base_model_chunks.json``
    so ``tinct ship`` can prove exactly which weights training used.

    Returns the resolved local path to the model (used as the training base).
    """
    console = get_console()
    cache_dir = get_cache_dir(project.root)
    console.print("[tinct] Preparing base model for low-VRAM streaming...")

    model_path = resolve_model(model_id, cache_dir=cache_dir)
    chunker = ModelChunker(cache_dir)
    chunk_manifest = chunker.chunk_model(model_path, chunk_size_mb=chunk_size_mb)

    manifest_path = run_dir / "base_model_chunks.json"
    manifest_path.write_text(
        json.dumps(chunk_manifest, indent=2), encoding="utf-8"
    )
    console.print(f"  model:     {model_path}")
    console.print(f"  chunks:    {chunker.cache_dir} ({len(chunk_manifest)} files)")
    console.print(f"  manifest:  {manifest_path}")
    return model_path


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

    # Automatic model prep: resolve + chunk + hash the base model. Fail-closed:
    # a model without a safetensors index is rejected before any training.
    try:
        model_path = prepare_base_model_chunks(project, run_dir, train_cfg.model)
    except ValueError as exc:
        console.print(f"[bold red]Model prep blocked:[/] {exc}")
        return 2

    # The Data Doctor validates raw columns; format them into the chat-text
    # field the fail-closed SFT trainer consumes.
    text_records = [_to_text(r, train_cfg, True) for r in train_records]
    text_path = run_dir / "train_text.jsonl"
    _write_jsonl([{"text": t} for t in text_records], text_path)

    ok_run = run_sft(
        model_name_or_path=str(model_path),
        dataset_path=text_path,
        run_dir=run_dir,
        lora_rank=train_cfg.lora_r,
        max_loss_threshold=project.config.max_loss_threshold,
    )
    # Normalize the fail-closed log into metrics.json for the eval gate.
    _materialize_metrics(run_dir)

    if not ok_run:
        console.print("[bold red]Training was halted by a fail-closed guard. DO NOT SHIP this run.[/]")
        console.print(f"  inspect: {run_dir / 'fail_state.json'} (if present)")
        return 1

    console.print("[bold green]Training complete (fail-closed guards passed).[/]")
    console.print(f"  adapter: {run_dir / 'adapter'}")
    console.print(f"\nNext: run `tinct eval --run {name}` to gate the checkpoint.")
    return 0
