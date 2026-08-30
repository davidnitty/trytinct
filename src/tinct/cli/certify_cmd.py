"""``tinct certify`` — the integration layer.

Users train with whatever they like (LLaMA-Factory, Unsloth, Axolotl, or tinct
itself); certification happens here. ``tinct certify`` loads an externally
trained LoRA adapter onto its base model, runs the eval gates (generation
smoke test + behavioral safety gates), signs the evidence bundle, and issues a
SHIP / DON'T-SHIP verdict with cryptographic proof.

Does NOT run training — assumes the adapter was trained externally.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tinct.cli.render import print_decision
from tinct.core.adapter_validator import (
    validate_adapter_compatible,
    validate_adapter_structure,
)
from tinct.core.model_gate import check_model_family
from tinct.engine.deps import MissingDependencyError
from tinct.security.evidence import EvidenceReport, hash_directory, hash_path
from tinct.security.signing import SigningKey
from tinct.storage.paths import TinctPaths
from tinct.utils.logging import get_console


def _default_cert_id() -> str:
    return datetime.now(timezone.utc).strftime("cert_%Y%m%d_%H%M%S")


def _ensure_signing_key(paths: TinctPaths, key_name: str) -> SigningKey:
    """Load the project's signing key, generating one for standalone use."""
    console = get_console()
    try:
        return SigningKey.load(paths.keys_dir, key_name)
    except FileNotFoundError:
        key = SigningKey.generate(key_name)
        key.save(paths.keys_dir)
        console.print(
            f"[yellow]Generated new signing key:[/] "
            f"{paths.keys_dir / (key_name + '_private.pem')}"
        )
        return key


def run_certify(
    adapter: Path,
    base_model: str,
    root: Path = Path("."),
    canaries_path: Path | None = None,
    dataset_path: Path | None = None,
    run_id: str | None = None,
    skip_safety: bool = False,
    offload_experts: bool = False,
) -> int:
    """Certify an externally trained adapter.

    Returns 0 (SHIP) or 2 (DON'T SHIP); 1 for usage errors, 3 for missing
    dependencies. Both verdicts produce signed evidence.
    """
    console = get_console()
    adapter = Path(adapter)

    # 1. Model family gate — llama + mistral supported.
    try:
        family = check_model_family(base_model)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        return 2

    # 2. The adapter must be a real PEFT adapter from a known tool shape.
    validation = validate_adapter_structure(adapter)
    if not validation["valid"]:
        for err in validation["errors"]:
            console.print(f"[bold red]{err}[/]")
        return 1
    training_tool = validation["training_tool"]
    if training_tool == "unknown":
        training_tool = "external"
    console.print(f"  adapter type: {validation['adapter_type']}")
    console.print(f"  training tool: {training_tool}")

    # 2b. The adapter must be compatible with the base model — a LoRA trained
    #     on a different architecture would silently invalidate every gate
    #     result below.
    compat = validate_adapter_compatible(adapter, base_model)
    if not compat["compatible"]:
        for err in compat["errors"]:
            console.print(f"[bold red]{err}[/]")
        return 1
    if compat.get("adapter_base_model"):
        console.print(f"  trained on:  {compat['adapter_base_model']}")

    # 3. Standalone state: ensure the .tinct tree and a signing key exist.
    paths = TinctPaths(Path(root).resolve())
    paths.ensure_dirs()

    cert_id = run_id or _default_cert_id()
    work_dir = paths.runs_dir / cert_id
    work_dir.mkdir(parents=True, exist_ok=True)

    canaries: list[dict] = []
    if canaries_path is not None:
        canaries_file = Path(canaries_path)
        if not canaries_file.is_file():
            console.print(f"[bold red]Canaries file not found:[/] {canaries_file}")
            return 1
        canaries = json.loads(canaries_file.read_text(encoding="utf-8"))

    console.print(f"[bold]Certification[/] {cert_id}")
    console.print(f"  base model: {base_model} ({family})")
    console.print(f"  adapter:    {adapter}")

    # 4. Persist adapter provenance (hashed into the evidence bundle).
    adapter_metadata = {
        "adapter_path": str(adapter.resolve()),
        "base_model": base_model,
        "adapter_type": validation["adapter_type"],
        "training_tool": training_tool,
        "adapter_config": json.loads(
            (adapter / "adapter_config.json").read_text(encoding="utf-8")
        ),
        "certified_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = work_dir / "adapter_metadata.json"
    metadata_path.write_text(json.dumps(adapter_metadata, indent=2), encoding="utf-8")

    # 5. Eval gates. Any gate failure still produces signed evidence — the
    #    DON'T-SHIP verdict carries the cryptographic proof of why.
    decision = "SHIP"
    try:
        from tinct.evals.smoke_test import run_generation_smoke_test

        eval_report_path = work_dir / "eval_report.json"
        smoke_pass = run_generation_smoke_test(base_model, adapter, eval_report_path)
        if not smoke_pass:
            decision = "DON'T_SHIP"
    except MissingDependencyError as exc:
        console.print(f"[bold red]Cannot certify:[/] {exc}")
        return 3
    except Exception as exc:
        # Model-load failures (gated repo, missing weights, no GPU) surface
        # here as a clean error rather than a traceback.
        console.print(f"[bold red]Certification failed while loading the model:[/] {exc}")
        console.print("  Gated model? Authenticate with `huggingface-cli login`.")
        return 1

    safety_gates: dict = {}
    if not skip_safety:
        try:
            from tinct.safety.gates import run_safety_gates_for_run

            safety_gates = run_safety_gates_for_run(
                base_model, adapter, canaries, offload_experts=offload_experts
            )
            safety_path = work_dir / "safety_gates.json"
            safety_path.write_text(json.dumps(safety_gates, indent=2), encoding="utf-8")
            failed = [
                name
                for name, gate in safety_gates.items()
                if isinstance(gate, dict) and gate.get("status") == "FAIL"
            ]
            if failed or safety_gates.get("result") == "FAIL":
                decision = "DON'T_SHIP"
        except MissingDependencyError as exc:
            console.print(f"[bold red]Cannot run safety gates:[/] {exc}")
            return 3
        except Exception as exc:
            console.print(f"[bold red]Safety gates failed while loading the model:[/] {exc}")
            return 1

    # 6. Build and sign the evidence bundle — both verdicts are signed, so a
    #    DON'T-SHIP carries cryptographic proof of why.
    adapter_hash = hash_directory(adapter)
    artifacts = {
        "adapter": hash_directory(adapter),
        "adapter_sha256": {"path": str(adapter), "sha256": adapter_hash},
        "adapter_metadata.json": hash_path(metadata_path),
        "eval_report.json": hash_path(work_dir / "eval_report.json"),
    }
    if canaries_path is not None:
        artifacts["canaries.json"] = hash_path(Path(canaries_path))
    if dataset_path is not None and Path(dataset_path).is_file():
        artifacts["dataset"] = hash_path(Path(dataset_path))
    if not skip_safety:
        artifacts["safety_gates.json"] = hash_path(work_dir / "safety_gates.json")

    report = EvidenceReport(
        project_name=Path(root).resolve().name,
        model=base_model,
        family=family,
        decision=decision,
        artifacts=artifacts,
        eval_report=json.loads(
            (work_dir / "eval_report.json").read_text(encoding="utf-8")
        ),
        safety_gates=safety_gates,
        training_tool=training_tool,
        training_executed=False,  # certify never trains — adapter came from outside
        config={
            "base_model": base_model,
            "adapter": str(adapter),
            "certified_at": datetime.now(timezone.utc).isoformat(),
            "integration": "certify",
        },
    )

    key = _ensure_signing_key(paths, "ship")
    report.sign(key)
    if not report.verify():
        console.print("[bold red]Evidence signature verification failed; refusing to certify.[/]")
        return 1
    evidence_path = report.write(paths.evidence_dir, cert_id)
    console.print("[green]Evidence signed and signature verified.[/]")

    # 7. Verdict.
    print_decision(console, decision)
    console.print(f"[bold green]Evidence signed and saved:[/] {evidence_path}")
    console.print(f"  adapter_sha256: {adapter_hash}")
    return 0 if decision == "SHIP" else 2
