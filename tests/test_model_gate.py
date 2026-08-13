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
    # qwen is *detectable* but not *allowed* in V0 -> fail-closed.
    with pytest.raises(UnsupportedModelFamily):
        check_model_family("Qwen/Qwen2.5-7B")


def test_check_uses_project_allowed_list():
    # A project that adds 'qwen' to model_families_allowed passes.
    assert check_model_family("Qwen/Qwen2.5-7B", allowed_families=["llama", "qwen"]) == "qwen"


def test_check_rejects_outside_allowed():
    with pytest.raises(UnsupportedModelFamily):
        check_model_family("meta-llama/Llama-3.1-8B", allowed_families=["qwen"])
