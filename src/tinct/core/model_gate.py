"""Model-family gate — a fail-closed whitelist of supported architectures.

Before any train/eval work, tinct refuses models it cannot safely handle. The
gate has two parts:

1. :func:`detect_model_family` recognizes a set of *known* families by name.
2. :func:`check_model_family` enforces the **allowed** families (from
   ``project.yaml`` → ``model_families_allowed``). By default only ``llama``
   is permitted; Qwen/DeepSeek are detected but blocked until V2/V3.

This is the single source of truth for model-family policy. ``Project`` and
the CLI route through :func:`check_model_family`.
"""

from __future__ import annotations

from typing import Sequence

# Families this release can *identify*. Which are *allowed* is a config
# decision (see SUPPORTED_MODEL_FAMILIES / project.yaml).
KNOWN_MODEL_FAMILIES: tuple[str, ...] = (
    "llama",
    "qwen",
    "deepseek",
    "mistral",
    "gemma",
    "phi",
)

# Default whitelist for V0: Llama only.
SUPPORTED_MODEL_FAMILIES: tuple[str, ...] = ("llama",)


class UnsupportedModelFamily(ValueError):
    """Raised when a model is unknown or outside the allowed families."""


def detect_model_family(model: str) -> str:
    """Return the (lowercased) family name for ``model``, or raise.

    Matches a known family name appearing in the final path segment. An
    unrecognized architecture fails-closed (we cannot safely handle it).
    """
    name = model.rsplit("/", 1)[-1].lower()
    for family in KNOWN_MODEL_FAMILIES:
        if family in name:
            return family
    raise UnsupportedModelFamily(
        f"Model {model!r} is not a recognized family. tinct can identify: "
        f"{list(KNOWN_MODEL_FAMILIES)}."
    )


def check_model_family(model: str,
                       allowed_families: Sequence[str] | None = None) -> str:
    """Fail-closed gate: ensure ``model``'s family is allowed.

    Returns the detected family name. Raises :class:`UnsupportedModelFamily`
    if the family cannot be identified or is not in ``allowed_families``
    (defaults to :data:`SUPPORTED_MODEL_FAMILIES`).
    """
    family = detect_model_family(model)
    allowed = tuple(allowed_families) if allowed_families else SUPPORTED_MODEL_FAMILIES
    if family not in allowed:
        raise UnsupportedModelFamily(
            f"Model family {family!r} is not among the allowed families "
            f"configured for this project: {list(allowed)}."
        )
    return family

