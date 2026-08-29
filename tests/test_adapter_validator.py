"""Tests for external adapter validation (PEFT structure + tool detection)."""

import json
from pathlib import Path

import pytest

from tinct.core.adapter_validator import (
    validate_adapter_compatible,
    validate_adapter_structure,
)


def _safetensors_bytes() -> bytes:
    """A minimal valid safetensors file (header + 4 bytes of data)."""
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"


def _write_adapter(path: Path, config: dict, weights: bool = True,
                   weight_bytes: bytes | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    if weights:
        (path / "adapter_model.safetensors").write_bytes(
            weight_bytes if weight_bytes is not None else _safetensors_bytes()
        )


def test_valid_peft_adapter(tmp_path: Path):
    _write_adapter(tmp_path, {
        "peft_type": "LORA",
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
    })
    result = validate_adapter_structure(tmp_path)
    assert result["valid"] is True
    assert result["adapter_type"] == "peft-lora"
    assert result["errors"] == []


def test_missing_adapter_config(tmp_path: Path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"dummy")
    result = validate_adapter_structure(tmp_path)
    assert result["valid"] is False
    assert "adapter_config.json" in result["errors"][0]


def test_missing_adapter_weights(tmp_path: Path):
    _write_adapter(tmp_path, {"peft_type": "LORA"}, weights=False)
    result = validate_adapter_structure(tmp_path)
    assert result["valid"] is False
    assert "adapter_model" in result["errors"][0]


def test_detect_unsloth_tool(tmp_path: Path):
    _write_adapter(tmp_path, {
        "peft_type": "LORA",
        "unsloth_version": "2024.1",
    })
    result = validate_adapter_structure(tmp_path)
    assert result["valid"] is True
    assert result["training_tool"] == "unsloth"


def test_detect_llama_factory_tool(tmp_path: Path):
    _write_adapter(tmp_path, {
        "peft_type": "LORA",
        "llama_factory_training_args": {"finetuning_type": "lora"},
    })
    result = validate_adapter_structure(tmp_path)
    assert result["training_tool"] == "llama-factory"


def test_detect_axolotl_tool(tmp_path: Path):
    _write_adapter(tmp_path, {
        "peft_type": "LORA",
        "axolotl_config_version": 1,
    })
    result = validate_adapter_structure(tmp_path)
    assert result["training_tool"] == "axolotl"


def test_detect_qlora_adapter(tmp_path: Path):
    _write_adapter(tmp_path, {
        "peft_type": "LORA",
        "quantization_config": {"load_in_4bit": True},
    })
    result = validate_adapter_structure(tmp_path)
    assert result["adapter_type"] == "peft-qlora"


def test_empty_directory(tmp_path: Path):
    # An empty adapter dir fails on the missing adapter_config.json.
    (tmp_path / "adapter").mkdir()
    result = validate_adapter_structure(tmp_path / "adapter")
    assert result["valid"] is False
    assert result["adapter_type"] is None


def test_detect_bin_weights(tmp_path: Path):
    # Pickle .bin weights are accepted for import (hashed, never executed).
    _write_adapter(tmp_path, {"peft_type": "LORA"}, weights=False)
    (tmp_path / "adapter_model.bin").write_bytes(b"dummy")
    result = validate_adapter_structure(tmp_path)
    assert result["valid"] is True
    assert result["has_bin"] is True
    assert result["has_safetensors"] is False


def test_corrupt_safetensors_rejected(tmp_path: Path):
    # Garbage weights (e.g. a truncated download) fail the header check
    # before any base-model download is attempted.
    _write_adapter(tmp_path, {"peft_type": "LORA"}, weight_bytes=b"x")
    result = validate_adapter_structure(tmp_path)
    assert result["valid"] is False
    assert any("Invalid or corrupt" in e for e in result["errors"])


def test_non_lora_peft_type_reported(tmp_path: Path):
    _write_adapter(tmp_path, {"peft_type": "PREFIX_TUNING"})
    result = validate_adapter_structure(tmp_path)
    assert result["adapter_type"] == "prefix_tuning"


def test_nonexistent_path(tmp_path: Path):
    result = validate_adapter_structure(tmp_path / "nope")
    assert result["valid"] is False
    assert result["errors"] == ["Adapter path does not exist"]


def test_non_directory_path(tmp_path: Path):
    file_path = tmp_path / "a-file"
    file_path.write_text("x")
    result = validate_adapter_structure(file_path)
    assert result["valid"] is False
    assert result["errors"] == ["Adapter path is not a directory"]


def test_unparseable_adapter_config(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{not json", encoding="utf-8")
    result = validate_adapter_structure(tmp_path)
    assert result["valid"] is False
    assert any("Failed to parse" in e for e in result["errors"])


# -- adapter / base-model compatibility ---------------------------------------

def test_compatible_exact_same_base(tmp_path: Path):
    _write_adapter(tmp_path, {
        "peft_type": "LORA",
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
    })
    result = validate_adapter_compatible(tmp_path, "meta-llama/Llama-3.1-8B")
    assert result["compatible"] is True
    assert result["adapter_base_model"] == "meta-llama/Llama-3.1-8B"


def test_compatible_short_name_matches_full_id(tmp_path: Path):
    _write_adapter(tmp_path, {
        "peft_type": "LORA",
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
    })
    result = validate_adapter_compatible(tmp_path, "llama-3.1-8b")
    assert result["compatible"] is True


def test_incompatible_base_model_rejected(tmp_path: Path):
    _write_adapter(tmp_path, {
        "peft_type": "LORA",
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
    })
    result = validate_adapter_compatible(tmp_path, "Qwen/Qwen2.5-7B-Instruct")
    assert result["compatible"] is False
    assert any("may be incompatible" in e for e in result["errors"])
    assert result["adapter_base_model"] == "meta-llama/Llama-3.1-8B"


def test_no_base_model_in_config_allows_check(tmp_path: Path):
    _write_adapter(tmp_path, {"peft_type": "LORA"})
    result = validate_adapter_compatible(tmp_path, "meta-llama/Llama-3.1-8B")
    assert result["compatible"] is True
    assert result["adapter_base_model"] == ""


def test_incompatible_invalid_structure(tmp_path: Path):
    # No adapter config -> structure fails -> compatibility fails with the
    # structure errors surfaced.
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
    result = validate_adapter_compatible(tmp_path, "meta-llama/Llama-3.1-8B")
    assert result["compatible"] is False
    assert any("adapter_config.json" in e for e in result["errors"])


def test_structure_reports_weight_formats(tmp_path: Path):
    _write_adapter(tmp_path, {"peft_type": "LORA"})
    result = validate_adapter_structure(tmp_path)
    assert result["has_safetensors"] is True
    assert result["has_bin"] is False