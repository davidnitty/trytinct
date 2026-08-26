"""Behavioral certification gates — canary leakage + refusal regression.

These move ``tinct`` from "does it run?" to "is it safe?" The pure detection
functions are dependency-free and CPU-testable; the model-generation glue
lives in :mod:`tinct.safety.gates`.
"""

from tinct.safety.canaries import (
    canary_text,
    check_canary_leakage,
    generate_canaries,
    inject_canaries,
    save_canaries,
)
from tinct.safety.refusal import (
    SAFETY_PROMPTS,
    check_refusal_regression,
    is_refusal,
)

__all__ = [
    "SAFETY_PROMPTS",
    "canary_text",
    "check_canary_leakage",
    "check_refusal_regression",
    "generate_canaries",
    "inject_canaries",
    "is_refusal",
    "save_canaries",
]
