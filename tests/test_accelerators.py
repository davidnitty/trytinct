"""Tests for the accelerators module (unsloth guard)."""

import pytest

from tinct.engine.accelerators import (
    UNSLOTH_TARGET_MODULES,
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