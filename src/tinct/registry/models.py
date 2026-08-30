"""Model registry: defines support levels for different model families.

Tiers:
- **Tier 1**: Full support (template validation + safety gates + accelerator)
- **Tier 2**: Basic support (template validation only)
- **Tier 3**: Experimental (no validation, use at your own risk)
"""

from __future__ import annotations

MODEL_REGISTRY = {
    # Tier 1: Full support
    "meta-llama/Llama-3.1-8B": {
        "family": "llama",
        "tier": 1,
        "template": "llama-3-chat",
        "accelerator": ["unsloth", "hf"],
        "safety_gates": ["canary", "refusal", "toxicity"],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "family": "llama",
        "tier": 1,
        "template": "llama-3-chat",
        "accelerator": ["unsloth", "hf"],
        "safety_gates": ["canary", "refusal", "toxicity"],
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "family": "qwen",
        "tier": 1,
        "template": "qwen-chat",
        "accelerator": ["unsloth", "hf"],
        "safety_gates": ["canary", "refusal", "toxicity"],
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "family": "mistral",
        "tier": 1,
        "template": "mistral-chat",
        "accelerator": ["unsloth", "hf"],
        "safety_gates": ["canary", "refusal", "toxicity"],
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "family": "mistral",  # Mixtral uses Mistral's [INST] template
        "tier": 1,
        "template": "mistral-chat",
        "architecture": "moe",
        "num_experts": 8,
        "accelerator": ["hf", "moe_stream"],  # unsloth MoE support is experimental
        "safety_gates": ["canary", "refusal", "toxicity", "expert_collapse", "routing_regression"],
    },
    # Tier 2: Basic support (template validation only)
    "google/gemma-2-9b-it": {
        "family": "gemma",
        "tier": 2,
        "template": "gemma-chat",
        "accelerator": ["hf"],
        "safety_gates": [],  # No safety gates yet
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "family": "phi",
        "tier": 2,
        "template": "phi-chat",
        "accelerator": ["hf"],
        "safety_gates": [],
    },
}

_TIER_3_DEFAULT = {
    "family": "unknown",
    "tier": 3,
    "template": None,
    "accelerator": [],
    "safety_gates": [],
}


def get_model_info(model_name: str) -> dict:
    """Get model info from the registry.

    Args:
        model_name: HuggingFace model ID or path.

    Returns:
        Dict with model info, or a default Tier 3 entry if not found.
    """
    # Try exact match first.
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Try partial match (e.g. "llama-3.1-8b" matches "meta-llama/Llama-3.1-8B").
    model_lower = model_name.lower()
    for registry_name, info in MODEL_REGISTRY.items():
        registry_lower = registry_name.lower()
        if registry_lower in model_lower or model_lower in registry_lower:
            return info

    # Default to Tier 3 (experimental).
    return dict(_TIER_3_DEFAULT)


def is_fully_supported(model_name: str) -> bool:
    """Check if a model has full Tier 1 support."""
    return get_model_info(model_name)["tier"] == 1


def is_moe_model(model_name: str) -> bool:
    """Check if a model is a Mixture of Experts architecture."""
    info = get_model_info(model_name)
    return info.get("architecture") == "moe"


def get_supported_accelerators(model_name: str) -> list[str]:
    """Get the list of supported accelerators for a model."""
    return get_model_info(model_name).get("accelerator", [])


def get_safety_gates(model_name: str) -> list[str]:
    """Get the list of safety gates to run for a model."""
    return get_model_info(model_name).get("safety_gates", [])
