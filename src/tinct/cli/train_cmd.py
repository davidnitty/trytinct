"""``tinct train`` — validate then fine-tune a Llama adapter (LoRA/QLoRA)."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from rich.markup import escape

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


def _run_dpo_training(project: Project, doctor: DataDoctor, records, dataset: Path,
                      run_name: str | None) -> int:
    """DPO training: preference dataset (prompt/chosen/rejected) + the Reward
    Inversion Guard. Returns 0 on success, 2 on a fail-closed halt."""
    console = get_console()
    train_cfg = project.config.train

    if doctor.format_used != "dpo":
        console.print("[bold red]Method 'dpo' requires a preference dataset "
                      "with prompt/chosen/rejected fields.[/]")
        return 1

    try:
        ensure_train_deps()
    except MissingDependencyError as exc:
        console.print(f"[bold red]Cannot train:[/] {escape(str(exc))}")
        return 3

    name = run_name or _default_run_name()
    run_dir = project.create_run(name)
    # Preserve the exact training data in the run for the evidence bundle.
    dataset_out = run_dir / "train.jsonl"
    dataset_out.write_bytes(Path(dataset).read_bytes())
    console.print(f"[green]Run {name!r} created at[/] {run_dir}")

    from tinct.trainers.dpo_trainer import run_dpo
    ok_run = run_dpo(
        model_name_or_path=train_cfg.model,
        dataset_path=dataset_out,
        run_dir=run_dir,
        lora_rank=train_cfg.lora_r,
    )
    _materialize_metrics(run_dir)

    if not ok_run:
        console.print("\n[tinct] VERDICT: DON'T SHIP (DPO halted by fail-closed guard).")
        console.print(f"  inspect: {run_dir / 'fail_state.json'} (if present)")
        return 2

    console.print("\n[tinct] SUCCESS. Artifacts saved to:")
    console.print(f"  adapter: {run_dir / 'adapter'}")
    console.print("Run `tinct eval` and `tinct ship` to certify this run.\n")
    return 0


def _doctor_for(project: Project) -> DataDoctor:
    try:
        from tinct.core.model_gate import detect_model_family
        family = detect_model_family(project.config.train.model)
    except ValueError:
        family = None
    return DataDoctor(
        project.config.data,
        max_seq_len=project.config.train.max_seq_len,
        seed=project.config.train.seed,
        model_family=family,
    )


def run_train(project: Project, dataset: Path, run_name: str | None,
              model_override: str | None, method: str = "sft",
              lora_rank_override: int | None = None,
              max_loss_threshold_override: float | None = None,
              accelerator: str = "none") -> int:
    """Validate, split, train. Returns 0 on success, non-zero otherwise."""
    console = get_console()

    # Fail-closed 0: training method must be supported.
    method = method.lower()
    if method not in ("sft", "dpo"):
        console.print(f"[bold red]Method {method!r} is not supported. Use 'sft' or 'dpo'.[/]")
        return 1

    # Fail-closed 0b: accelerator must be known.
    if accelerator not in ("none", "unsloth"):
        console.print(f"[bold red]Accelerator {accelerator!r} is not supported. Use 'none' or 'unsloth'.[/]")
        return 1
    if method == "dpo" and accelerator == "unsloth":
        console.print("[yellow]Unsloth is not yet wired into DPO; falling back to 'none'.[/]")
        accelerator = "none"

    # Fail-closed 0c: verify unsloth is importable BEFORE doing any expensive
    # work (model download/chunking). Fails fast with an actionable hint.
    if accelerator == "unsloth":
        try:
            import unsloth  # noqa: F401
        except ImportError:
            console.print("[bold red]Unsloth is required for --accelerator unsloth "
                          "but is not installed.[/]")
            console.print("  Install it with: pip install " + escape("'tinct[unsloth]'"))
            return 3

    # Fail-closed 1: unsupported model family (security gate).
    if model_override:
        project.config.train.model = model_override
    try:
        project.refuse_if_unsupported_model()
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        return 2

    if lora_rank_override is not None:
        project.config.train.lora_r = lora_rank_override

    # Fail-closed 2: data must pass the Data Doctor.
    doctor = _doctor_for(project)
    try:
        report, records = doctor.run(dataset)
    except DatasetLoadError as exc:
        console.print(f"[bold red]Could not load dataset:[/] {exc}")
        return 2
    print_report(console, report)
    if not report.passed:
        console.print("[bold red]Training blocked: data validation failed (fail-closed).[/]")
        return 1

    if method == "dpo":
        return _run_dpo_training(project, doctor, records, dataset, run_name)

    train_records, valid_records = doctor.split(records)
    console.print(f"[bold]Split:[/] {len(train_records)} train / {len(valid_records)} valid")

    train_cfg = project.config.train
    # Check heavy deps BEFORE scaffolding a run dir (fail fast, no partial run).
    try:
        ensure_train_deps(quant=train_cfg.quant)
    except MissingDependencyError as exc:
        console.print(f"[bold red]Cannot train:[/] {escape(str(exc))}")
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

    # The Data Doctor validates the schema (columns or a pre-formatted text
    # field); build the chat-text dataset the fail-closed SFT trainer consumes.
    if doctor.format_used == "text":
        text_records = [{"text": str(r.get("text", ""))} for r in train_records]
    else:
        text_records = [{"text": _to_text(r, train_cfg, True)} for r in train_records]
    text_path = run_dir / "train_text.jsonl"
    _write_jsonl(text_records, text_path)

    max_loss = (max_loss_threshold_override
                if max_loss_threshold_override is not None
                else project.config.max_loss_threshold)
    console.print(f"[bold]Fail-closed loss threshold:[/] {max_loss}")

    try:
        ok_run = run_sft(
            model_name_or_path=str(model_path),
            dataset_path=text_path,
            run_dir=run_dir,
            lora_rank=train_cfg.lora_r,
            max_loss_threshold=max_loss,
            num_train_epochs=train_cfg.num_epochs,
            per_device_batch_size=train_cfg.batch_size,
            grad_accum_steps=train_cfg.grad_accum_steps,
            learning_rate=train_cfg.learning_rate,
            logging_steps=train_cfg.logging_steps,
            max_seq_length=train_cfg.max_seq_len,
            accelerator=accelerator,
        )
    except ImportError as exc:
        console.print(f"[bold red]Cannot train:[/] {escape(str(exc))}")
        return 3
    # Normalize the fail-closed log into metrics.json for the eval gate.
    _materialize_metrics(run_dir)

    if not ok_run:
        console.print("\n[tinct] VERDICT: DON'T SHIP (Training failed / fail-closed guard).")
        console.print(f"  inspect: {run_dir / 'fail_state.json'} (if present)")
        return 2

    console.print("\n[tinct] SUCCESS. Artifacts saved to:")
    console.print(f"  adapter: {run_dir / 'adapter'}")
    console.print(f"  run:     {run_dir}")
    console.print("Run `tinct eval` and `tinct ship` to certify this run.\n")
    return 0
