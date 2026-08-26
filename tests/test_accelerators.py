"""Tests for the accelerators module (unsloth guard)."""

from unittest.mock import MagicMock, patch

import pytest

from tinct.engine.accelerators import (
    UNSLOTH_TARGET_MODULES,
    _load_unsloth,
    load_model_with_accelerator,
)


def test_unsloth_missing_raises_actionable_error():
    # unsloth is NOT installed -> requesting it yields the install hint.
    # We simulate by forcing the import to fail deterministically.
    import builtins
    import tinct.engine.accelerators as acc

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "unsloth":
            raise ImportError("No module named 'unsloth'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(ImportError, match=r"tinct\[unsloth\]"):
            load_model_with_accelerator("meta-llama/Llama-3.1-8B",
                                        accelerator="unsloth")
    finally:
        builtins.__import__ = real_import


def test_unsloth_target_modules_are_standard_llama():
    assert UNSLOTH_TARGET_MODULES == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]


# -- Test 5: accelerator selection routing -----------------------------------

def test_accelerator_selection_unsloth():
    """Verify the unsloth accelerator is selected correctly."""
    with patch("tinct.engine.accelerators._load_unsloth") as mock_unsloth:
        mock_unsloth.return_value = (MagicMock(), MagicMock())

        model, tokenizer = load_model_with_accelerator(
            model_name="test-model",
            accelerator="unsloth",
            lora_rank=16,
        )

        mock_unsloth.assert_called_once()
        assert model is not None and tokenizer is not None


def test_accelerator_selection_standard():
    """Verify the standard HF accelerator is selected correctly."""
    with patch("tinct.engine.accelerators._load_standard") as mock_hf:
        mock_hf.return_value = (MagicMock(), MagicMock())

        model, tokenizer = load_model_with_accelerator(
            model_name="test-model",
            accelerator="none",
            lora_rank=16,
        )

        mock_hf.assert_called_once()
        assert model is not None and tokenizer is not None


def test_unsloth_import_error_message():
    """Verify the helpful error message when Unsloth is missing."""
    with patch.dict("sys.modules", {"unsloth": None}):
        with pytest.raises(ImportError) as exc_info:
            _load_unsloth("test-model", 16, 2048)

        assert "tinct[unsloth]" in str(exc_info.value)