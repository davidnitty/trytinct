"""Behavioral safety gates and model provenance."""

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
from tinct.safety.toxicity import (
    NEUTRAL_PROMPTS,
    check_toxicity_increase,
    score_toxicity,
    score_toxicity_heuristic,
)

__all__ = [
    "NEUTRAL_PROMPTS",
    "SAFETY_PROMPTS",
    "canary_text",
    "check_canary_leakage",
    "check_refusal_regression",
    "check_toxicity_increase",
    "generate_canaries",
    "inject_canaries",
    "is_refusal",
    "save_canaries",
    "score_toxicity",
    "score_toxicity_heuristic",
]
