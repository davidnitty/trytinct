"""``tinct ship`` — the certification engine: SHIP/DON'T-SHIP verdict + signed evidence."""

from __future__ import annotations

import json
from pathlib import Path

from tinct.cli.eval_cmd import resolve_run
from tinct.cli.render import print_decision
from tinct.core.datadoctor import DataDoctor
from tinct.core.project import Project
from tinct.security.evidence import EvidenceReport, hash_directory, hash_path
from tinct.security.signing import SigningKey
from tinct.utils.logging import get_console


def _data_report(project: Project, run_dir: Path) -> dict:
    doctor = DataDoctor(
        project.config.data,
        max_seq_len=project.config.train.max_seq_len,
        seed=project.config.train.seed,
    )
    report, _ = doctor.run(run_dir / "train.jsonl")
    return report.to_dict()


def run_ship(project: Project, run_name: str | None) -> int:
    console = get_console()
    run_dir = resolve_run(project, run_name)
    if run_dir is None:
        console.print("[bold red]No run found to ship.[/] Run `tinct train` first.")
        return 1

    # 1. Fail-closed: a run aborted by the training guard must never ship.
    if (run_dir / "fail_state.json").exists():
        print_decision(console, "DON'T_SHIP")
        console.print("[tinct] Reason: Training aborted by fail-closed guard.\n")
        return 2

    # 2. Eval check: the generation smoke test must have passed.
    eval_report_path = run_dir / "eval_report.json"
    if not eval_report_path.is_file():
        console.print("[red]Error: Run `tinct eval` before shipping.[/]")
        return 1
    eval_data = json.loads(eval_report_path.read_text(encoding="utf-8"))
    if eval_data.get("status") != "PASS":
        print_decision(console, "DON'T_SHIP")
        console.print("[tinct] Reason: Generation smoke test failed.\n")
        return 2

    # 3. DPO certification gate (only for DPO runs, which persist
    #    dpo_metrics.json). Answers "did alignment happen?" — not just
    #    "does it run?".
    training_method = "sft"
    dpo_metrics = None
    dpo_metrics_path = run_dir / "dpo_metrics.json"
    if dpo_metrics_path.is_file():
        dpo_metrics = json.loads(dpo_metrics_path.read_text(encoding="utf-8"))
        training_method = "dpo"
        # DPO gate 1: no reward inversion may ship (backstop for the guard).
        if dpo_metrics.get("reward_inversion_detected"):
            print_decision(console, "DON'T_SHIP")
            console.print("[tinct] Reason: reward inversion detected during training.\n")
            return 2
        # DPO gate 2: final alignment must be positive (chosen > rejected).
        if dpo_metrics.get("final_reward_margin", 0) <= 0:
            print_decision(console, "DON'T_SHIP")
            console.print("[tinct] Reason: non-positive reward margin "
                          "(model does not prefer chosen).\n")
            return 2

    # 4. Hash the adapter — prove exactly which weights are shipping.
    adapter_dir = run_dir / "adapter"
    if not adapter_dir.is_dir():
        console.print("[red]Error: no adapter directory in the run.[/]")
        return 1
    adapter_hash = hash_directory(adapter_dir)

    data_report = _data_report(project, run_dir)
    artifacts = {
        "train_data.jsonl": hash_path(run_dir / "train.jsonl"),
        "valid_data.jsonl": hash_path(run_dir / "valid.jsonl"),
        "config": hash_path(project.config_path),
        "adapter": hash_path(adapter_dir),
        "adapter_sha256": {"path": str(adapter_dir), "sha256": adapter_hash},
        "eval_report.json": hash_path(eval_report_path),
    }
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        artifacts["metrics.json"] = hash_path(metrics_path)
    if dpo_metrics_path.is_file():
        artifacts["dpo_metrics.json"] = hash_path(dpo_metrics_path)
    chunk_manifest = run_dir / "base_model_chunks.json"
    if chunk_manifest.is_file():
        artifacts["base_model_chunks.json"] = hash_path(chunk_manifest)

    report = EvidenceReport(
        project_name=project.config.project_name,
        model=project.config.train.model,
        family="llama",
        decision="SHIP",
        artifacts=artifacts,
        data_report=data_report,
        eval_report=eval_data,
        metrics={"training_method": training_method, "dpo_metrics": dpo_metrics},
        config=project.config.model_dump(mode="json"),
    )

    # Cryptographic evidence is required to ship (secure by default).
    if project.config.security.sign_evidence:
        try:
            key = SigningKey.load(project.keys_dir, project.config.security.key_name)
        except FileNotFoundError as exc:
            console.print(f"[bold red]Cannot ship:[/] {exc}")
            console.print("  Run `tinct security key generate --name "
                          f"{project.config.security.key_name}` first.")
            return 1
        report.sign(key)
        if not report.verify():
            console.print("[bold red]Evidence signature verification failed; refusing to ship.[/]")
            return 1
        console.print("[green]Evidence signed and signature verified.[/]")

    path = report.write(project.evidence_dir, run_dir.name)
    print_decision(console, "SHIP")
    console.print(f"[bold green]Evidence signed and saved:[/] {path}")
    console.print(f"  adapter_sha256: {adapter_hash}")
    return 0
