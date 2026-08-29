"""Tests for the model registry (support tiers)."""

from tinct.registry.models import (
    get_model_info,
    get_safety_gates,
    get_supported_accelerators,
    is_fully_supported,
)


def test_exact_match_tier1():
    info = get_model_info("mistralai/Mistral-7B-Instruct-v0.3")
    assert info["tier"] == 1
    assert info["family"] == "mistral"
    assert "unsloth" in info["accelerator"]
    assert info["safety_gates"] == ["canary", "refusal", "toxicity"]


def test_partial_match_short_name():
    info = get_model_info("llama-3.1-8b")
    assert info["tier"] == 1
    assert info["family"] == "llama"


def test_unknown_model_defaults_tier3():
    info = get_model_info("some/unknown-arch-9000")
    assert info["tier"] == 3
    assert info["family"] == "unknown"
    assert info["accelerator"] == []
    assert info["safety_gates"] == []


def test_is_fully_supported():
    assert is_fully_supported("meta-llama/Llama-3.1-8B") is True
    assert is_fully_supported("Qwen/Qwen2.5-7B-Instruct") is True
    assert is_fully_supported("google/gemma-2-9b-it") is False   # tier 2
    assert is_fully_supported("some/unknown-arch") is False      # tier 3


def test_get_supported_accelerators():
    assert get_supported_accelerators("meta-llama/Llama-3.1-8B") == ["unsloth", "hf"]
    assert get_supported_accelerators("google/gemma-2-9b-it") == ["hf"]
    assert get_supported_accelerators("some/unknown-arch") == []


def test_get_safety_gates():
    assert get_safety_gates("mistralai/Mistral-7B-Instruct-v0.3") == [
        "canary", "refusal", "toxicity"
    ]
    assert get_safety_gates("google/gemma-2-9b-it") == []


def test_mixtral_maps_to_mistral_family():
    # Family detection maps Mixtral to mistral; the registry carries a
    # Mixtral entry so partial matches on the MoE line stay Tier 1.
    info = get_model_info("mistralai/Mixtral-8x7B-Instruct-v0.1")
    assert info["tier"] == 1
    assert info["family"] == "mistral"
