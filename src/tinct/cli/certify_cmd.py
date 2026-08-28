"""``tinct certify`` — the integration layer.

Users train with whatever they like (LLaMA-Factory, Unsloth, Axolotl, or tinct
itself); certification happens here. ``tinct certify`` loads an externally
trained LoRA adapter onto its base model, runs the eval gates (generation
smoke test + behavioral safety gates), signs the evidence bundle, and issues a
SHIP / DON'T-SHIP verdict with cryptographic proof.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tinct.cli.render import print_decision
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


def _validate_adapter_dir(adapter: Path) -> Optional[str]:
    """Fail-closed check that ``adapter`` looks like a LoRA adapter directory."""
    if not adapter.is_dir():
        return f"Adapter not found: {adapter}"
    if not (adapter / "adapter_config.json").is_file() and \
            not any(adapter.glob("*.safetensors")):
        return (
            "Directory does not look like a LoRA adapter (expected "
            "adapter_config.json or *.safetensors)."
        )
    return None


def run_certify(
    adapter: Path,
    base_model: str,
    root: Path = Path("."),
    canaries_path: Path | None = None,
    skip_safety: bool = False,
) -> int:
    """Certify an externally trained adapter. Returns 0 (SHIP) or 2 (DON'T SHIP);
    1 for usage errors, 3 for missing dependencies."""
    console = get_console()
    adapter = Path(adapter)

    # 1. Model family gate — llama + mistral supported.
    try:
        family = check_model_family(base_model)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        return 2

    # 2. The adapter must be a real LoRA adapter directory.
    error = _validate_adapter_dir(adapter)
    if error:
        console.print(f"[bold red]{error}[/]")
        return 1

    # 3. Standalone state: ensure the .tinct tree and a signing key exist.
    paths = TinctPaths(Path(root).resolve())
    paths.ensure_dirs()

    canaries: list[dict] = []
    if canaries_path is not None:
        canaries_file = Path(canaries_path)
        if not canaries_file.is_file():
            console.print(f"[bold red]Canaries file not found:[/] {canaries_file}")
            return 1
        canaries = json.loads(canaries_file.read_text(encoding="utf-8"))

    cert_id = _default_cert_id()
    work_dir = paths.runs_dir / cert_id
    work_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Certification[/] {cert_id}")
    console.print(f"  base model: {base_model} ({family})")
    console.print(f"  adapter:    {adapter}")

    # 4. Eval gates. Any gate failure still produces signed evidence — the
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

    safety_gates: dict = {}
    if not skip_safety:
        try:
            from tinct.safety.gates import run_safety_gates_for_run

            safety_gates = run_safety_gates_for_run(base_model, adapter, canaries)
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

    # 5. Build and sign the evidence bundle — both verdicts are signed, so a
    #    DON'T-SHIP carries cryptographic proof of why.
    adapter_hash = hash_directory(adapter)
    artifacts = {
        "adapter": hash_directory(adapter),
        "adapter_sha256": {"path": str(adapter), "sha256": adapter_hash},
        "eval_report.json": hash_path(work_dir / "eval_report.json"),
    }
    if canaries_path is not None:
        artifacts["canaries.json"] = hash_path(Path(canaries_path))
    if not skip_safety:
        artifacts["safety_gates.json"] = hash_path(work_dir / "safety_gates.json")

    report = EvidenceReport(
        project_name=Path(root).resolve().name,
        model=base_model,
        family=family,
        decision=decision,
        artifacts=artifacts,
        eval_report=json.loads((work_dir / "eval_report.json").read_text(encoding="utf-8")),
        safety_gates=safety_gates,
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

    # 6. Verdict.
    print_decision(console, decision)
    console.print(f"[bold green]Evidence signed and saved:[/] {evidence_path}")
    console.print(f"  adapter_sha256: {adapter_hash}")
    return 0 if decision == "SHIP" else 2
