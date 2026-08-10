"""Lazy dependency guards.

Importing ``tinct`` never touches heavy ML packages. These helpers import them
on demand and raise a clear, actionable error when a command needs one that is
not installed — pointing at the matching optional extra.
"""

from __future__ import annotations

import importlib
from typing import Any, TypeVar

T = TypeVar("T")


class MissingDependencyError(RuntimeError):
    """Raised when a lazily-loaded heavy dependency is unavailable."""


def import_optional(module_name: str, extra: str, package: str = "trytinct") -> Any:
    """Import ``module_name`` or raise a helpful :class:`MissingDependencyError`.

    ``extra`` is the PEP 508 extra name (e.g. ``train``) to suggest.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on env
        raise MissingDependencyError(
            f"Command requires the optional dependency {module_name!r}, which is not "
            f"installed. Install it with:\n\n    pip install {package}[{extra}]\n"
        ) from exc


def ensure_train_deps(quant: str = "qlora") -> None:
    """Verify all packages needed for LoRA/QLoRA training are importable.

    Raises :class:`MissingDependencyError` as soon as one is missing.
    """
    for mod in ("torch", "transformers", "trl", "peft", "accelerate", "datasets"):
        import_optional(mod, "train")
    if quant == "qlora":
        import_optional("bitsandbytes", "train")


def ensure_eval_deps() -> None:
    """Verify packages needed for eval gating are importable."""
    import_optional("torch", "train")
