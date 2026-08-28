"""Validates adapter structure from external training tools.

Supports:
- LLaMA-Factory (PEFT LoRA)
- Unsloth (PEFT LoRA)
- Axolotl (PEFT LoRA)
- Any tool that outputs standard PEFT adapters
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _result(valid: bool, errors: list[str],
            adapter_type: Optional[str], training_tool: Optional[str]) -> dict:
    return {
        "valid": valid,
        "errors": errors,
        "adapter_type": adapter_type,
        "training_tool": training_tool,
    }


def validate_adapter_structure(adapter_path: Path) -> dict:
    """Validate that an adapter directory has the expected PEFT structure.

    Returns:
        Dict with validation results::

            {
                "valid": bool,
                "errors": list[str],
                "adapter_type": str,     # "peft-lora", "peft-qlora", ...
                "training_tool": str,    # "llama-factory", "unsloth",
                                         # "axolotl", "unknown"
            }
    """
    if not adapter_path.exists():
        return _result(False, ["Adapter path does not exist"], None, None)

    if not adapter_path.is_dir():
        return _result(False, ["Adapter path is not a directory"], None, None)

    errors: list[str] = []

    # The adapter config is mandatory — without it we can't interpret anything.
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return _result(False, ["Missing adapter_config.json"], None, None)

    try:
        adapter_config: dict[str, Any] = json.loads(
            adapter_config_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return _result(
            False, [f"Failed to parse adapter_config.json: {exc}"], None, None
        )

    # Adapter weights: safetensors preferred, pickle .bin tolerated on import
    # (the weights are hashed, never executed).
    has_safetensors = (adapter_path / "adapter_model.safetensors").exists()
    has_bin = (adapter_path / "adapter_model.bin").exists()
    if not has_safetensors and not has_bin:
        errors.append("Missing adapter_model.safetensors or adapter_model.bin")

    # Detect adapter type from peft_type (+ quantization markers for QLoRA).
    adapter_type = "unknown"
    peft_type = str(adapter_config.get("peft_type", "")).lower()
    if "lora" in peft_type:
        config_blob = str(adapter_config).lower()
        has_quant_key = any("quantization" in str(k).lower() for k in adapter_config)
        if "4bit" in config_blob or "quantization" in config_blob or has_quant_key:
            adapter_type = "peft-qlora"
        else:
            adapter_type = "peft-lora"
    elif peft_type:
        adapter_type = peft_type

    # Detect the training tool (heuristic on config field names/values).
    training_tool = "unknown"
    config_blob = str(adapter_config).lower()
    if "llama_factory" in config_blob or "llama-factory" in config_blob:
        training_tool = "llama-factory"
    elif "unsloth" in config_blob:
        training_tool = "unsloth"
    elif "axolotl" in config_blob:
        training_tool = "axolotl"

    return _result(len(errors) == 0, errors, adapter_type, training_tool)