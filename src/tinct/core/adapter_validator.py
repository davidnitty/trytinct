"""Validates adapter structure from external training tools.

Supports:
- LLaMA-Factory (PEFT LoRA)
- Unsloth (PEFT LoRA)
- Axolotl (PEFT LoRA)
- Any tool that outputs standard PEFT adapters

Also verifies adapter/base-model compatibility (:func:`validate_adapter_compatible`)
— a LoRA trained on model X must not be certified on top of model Y.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _result(valid: bool, errors: list[str],
            adapter_type: Optional[str], training_tool: Optional[str],
            has_safetensors: bool = False, has_bin: bool = False) -> dict:
    return {
        "valid": valid,
        "errors": errors,
        "adapter_type": adapter_type,
        "training_tool": training_tool,
        "has_safetensors": has_safetensors,
        "has_bin": has_bin,
    }


def _safetensors_header_ok(path: Path) -> bool:
    """Cheap, dependency-free sanity check of a safetensors file header.

    A safetensors file starts with an 8-byte little-endian header length
    followed by a JSON header. Truncated downloads and garbage files fail
    here — long before any base-model download.
    """
    try:
        with path.open("rb") as fh:
            header_size = int.from_bytes(fh.read(8), "little")
            if header_size <= 0 or header_size > 100_000_000:
                return False
            header = fh.read(header_size)
        json.loads(header.decode("utf-8"))
        return True
    except Exception:
        return False


def validate_adapter_structure(adapter_path: Path) -> dict:
    """Validate that an adapter directory has the expected PEFT structure.

    Args:
        adapter_path: Path to the adapter directory.

    Returns:
        Dict with validation results::

            {
                "valid": bool,
                "errors": list[str],
                "adapter_type": str,      # "peft-lora", "peft-qlora", ...
                "training_tool": str,     # "llama-factory", "unsloth",
                                          # "axolotl", "unknown"
                "has_safetensors": bool,
                "has_bin": bool,
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
    # (the weights are hashed, never executed). Safetensors files are
    # header-validated here so corrupt/garbage files fail before any
    # base-model download.
    has_safetensors = (adapter_path / "adapter_model.safetensors").exists()
    has_bin = (adapter_path / "adapter_model.bin").exists()
    if not has_safetensors and not has_bin:
        errors.append("Missing adapter_model.safetensors or adapter_model.bin")
    elif has_safetensors and not _safetensors_header_ok(
            adapter_path / "adapter_model.safetensors"):
        errors.append(
            "Invalid or corrupt adapter_model.safetensors "
            "(failed to read safetensors header)"
        )

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

    return _result(len(errors) == 0, errors, adapter_type, training_tool,
                   has_safetensors=has_safetensors, has_bin=has_bin)


def validate_adapter_compatible(adapter_path: Path, base_model: str) -> dict:
    """Validate that an adapter is compatible with the base model it will be
    loaded onto.

    A LoRA trained on model X, attached to model Y, produces silently invalid
    results — every downstream gate would certify garbage. The adapter's
    ``base_model_name_or_path`` is compared against the requested base model
    with a lenient token-overlap heuristic (org prefixes and dash/underscore
    formatting differ across tools).

    Returns:
        ``{"compatible": bool, "errors": [...], "structure": {...},
        "adapter_base_model": str}``
    """
    structure = validate_adapter_structure(adapter_path)

    if not structure["valid"]:
        return {
            "compatible": False,
            "errors": list(structure["errors"]),
            "structure": structure,
            "adapter_base_model": "",
        }

    adapter_config_path = adapter_path / "adapter_config.json"
    try:
        adapter_config: dict[str, Any] = json.loads(
            adapter_config_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return {
            "compatible": False,
            "errors": [f"Failed to parse adapter_config.json: {exc}"],
            "structure": structure,
            "adapter_base_model": "",
        }

    adapter_base_model = str(adapter_config.get("base_model_name_or_path", ""))

    # If the adapter doesn't record its base model, we can't check — allow it,
    # but surface what (little) we know.
    if not adapter_base_model:
        return {
            "compatible": True,
            "errors": [],
            "structure": structure,
            "adapter_base_model": "",
        }

    adapter_base_lower = adapter_base_model.lower()
    base_model_lower = base_model.lower()

    # Exact or prefix containment: the tool may have recorded the full HF id
    # while the user passed a short name (or vice versa).
    if adapter_base_lower in base_model_lower or base_model_lower in adapter_base_lower:
        return {
            "compatible": True,
            "errors": [],
            "structure": structure,
            "adapter_base_model": adapter_base_model,
        }

    # Lenient token-overlap heuristic: compare significant parts after
    # normalizing separators. Fewer than 2 shared tokens means these are
    # probably different architectures.
    def _parts(name: str) -> set[str]:
        cleaned = name.replace("-", " ").replace("_", " ").replace("/", " ")
        return {p for p in cleaned.lower().split() if p}

    common_parts = _parts(adapter_base_model) & _parts(base_model)
    if len(common_parts) < 2:
        return {
            "compatible": False,
            "errors": [
                f"Adapter was trained on '{adapter_base_model}' but base model "
                f"is '{base_model}'. These may be incompatible."
            ],
            "structure": structure,
            "adapter_base_model": adapter_base_model,
        }

    return {
        "compatible": True,
        "errors": [],
        "structure": structure,
        "adapter_base_model": adapter_base_model,
    }