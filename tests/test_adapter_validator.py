"""Tests for external adapter validation (PEFT structure + tool detection)."""

import json
from pathlib import Path

from tinct.core.adapter_validator import validate_adapter_structure


def _write_adapter(path: Path, config: dict, weights: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    if weights:
        (path / "adapter_model.safetensors").write_bytes(b"dummy")


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