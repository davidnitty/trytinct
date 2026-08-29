"""``tinct doctor`` — preflight checks: deps, GPU, model access, project state."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from tinct.cli.render import print_report
from tinct.core.rules import RuleReport, error, ok, warn
from tinct.storage.paths import TinctPaths
from tinct.utils.logging import get_console

_CORE_DEPS = ("typer", "pydantic", "yaml", "cryptography", "rich")
_TRAIN_DEPS = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _hf_token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    token = Path.home() / ".cache" / "huggingface" / "token"
    return token.is_file()


def run_doctor(root: Path = Path(".")) -> int:
    """Run preflight checks and print the report. Returns 0 when healthy."""
    console = get_console()
    report = RuleReport("Doctor — Preflight")

    # Python version.
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        report.add(ok("doctor.python", "Python version",
                      f"{v.major}.{v.minor}.{v.micro}"))
    else:
        report.add(error("doctor.python", "Python version",
                         f"{v.major}.{v.minor} is below the required 3.10"))

    # Core dependencies (hard requirement — tinct cannot run without them).
    missing_core = [m for m in _CORE_DEPS if not _importable(m)]
    if missing_core:
        report.add(error("doctor.core_deps", "Core dependencies",
                         f"Missing: {', '.join(missing_core)}"))
    else:
        report.add(ok("doctor.core_deps", "Core dependencies",
                      f"{len(_CORE_DEPS)} packages importable"))

    # Training extras (optional — needed for train/eval/certify).
    missing_train = [m for m in _TRAIN_DEPS if not _importable(m)]
    if missing_train:
        report.add(warn("doctor.train_deps", "Training dependencies",
                        f"Missing: {', '.join(missing_train)} — "
                        "install with: pip install 'tinct[train]'"))
    else:
        report.add(ok("doctor.train_deps", "Training dependencies",
                      f"{len(_TRAIN_DEPS)} packages importable"))

    # Toxicity extra (optional — Tier 2 scoring).
    if _importable("detoxify"):
        report.add(ok("doctor.toxicity_deps", "Toxicity scoring",
                      "detoxify available (Tier 2)"))
    else:
        report.add(warn("doctor.toxicity_deps", "Toxicity scoring",
                        "detoxify not installed — toxicity gate falls back to "
                        "the keyword heuristic (pip install 'tinct[toxicity]')"))

    # GPU.
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            report.add(ok("doctor.gpu", "GPU", f"CUDA available — {name}"))
        else:
            report.add(warn("doctor.gpu", "GPU",
                            "No CUDA device — training will fall back to CPU "
                            "(eval may still work on small models)"))
    except ImportError:
        report.add(warn("doctor.gpu", "GPU",
                        "torch not installed — GPU status unknown"))

    # Hugging Face access (gated models).
    if _hf_token_present():
        report.add(ok("doctor.hf_token", "Hugging Face access",
                      "token found (env or cache)"))
    else:
        report.add(warn("doctor.hf_token", "Hugging Face access",
                        "No token found — gated models (Llama, Mistral) will "
                        "fail to download. Run `huggingface-cli login`."))

    # Project state.
    paths = TinctPaths(Path(root))
    if paths.project_config.is_file():
        report.add(ok("doctor.project", "Project state",
                      f"Initialized at {paths.project_root}"))
    else:
        report.add(warn("doctor.project", "Project state",
                        f"No project at {paths.project_root} — run `tinct init`"))

    # Signing key.
    key_path = paths.keys_dir / "ship_private.pem"
    if key_path.is_file():
        report.add(ok("doctor.signing_key", "Signing key", f"{key_path}"))
    else:
        report.add(warn("doctor.signing_key", "Signing key",
                        "No ship key — `tinct ship` will generate one on first use"))

    print_report(console, report)
    return 0 if report.passed else 1
