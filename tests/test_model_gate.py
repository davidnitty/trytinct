"""Tests for the model-family gate."""

import pytest

from tinct.core.model_gate import (
    UnsupportedModelFamily,
    check_model_family,
    detect_model_family,
)


def test_detect_llama():
    assert detect_model_family("meta-llama/Llama-3.1-8B") == "llama"


def test_detect_other_known_families():
    assert detect_model_family("Qwen/Qwen2.5-7B") == "qwen"
    assert detect_model_family("deepseek-ai/DeepSeek-R1") == "deepseek"


def test_detect_rejects_unknown_family():
    with pytest.raises(UnsupportedModelFamily):
        detect_model_family("foo/bar-arch-9000")


def test_check_allowed_default():
    assert check_model_family("meta-llama/Llama-3.1-8B") == "llama"


def test_check_rejects_known_but_not_allowed():
    # deepseek is *detectable* but not *allowed* by default -> fail-closed.
    with pytest.raises(UnsupportedModelFamily):
        check_model_family("deepseek-ai/DeepSeek-R1")


def test_check_uses_project_allowed_list():
    # A project that adds 'deepseek' to model_families_allowed passes.
    assert check_model_family("deepseek-ai/DeepSeek-R1", allowed_families=["llama", "deepseek"]) == "deepseek"


def test_check_rejects_outside_allowed():
    with pytest.raises(UnsupportedModelFamily):
        check_model_family("meta-llama/Llama-3.1-8B", allowed_families=["qwen"])
