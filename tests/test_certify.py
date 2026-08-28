"""Tests for tinct certify — the integration layer (CPU-testable gates only).

The full model-backed certification needs GPU + weights; here we verify the
fail-closed gating (family gate, adapter validation) that runs before any
model load.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tinct.cli.app import app

runner = CliRunner()

MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B"


def _make_adapter(path: Path, config: dict | None = None) -> Path:
    """Create a minimal valid PEFT LoRA adapter directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps(config or {"peft_type": "LORA"}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(b"x")
    return path


def test_certify_missing_adapter_dir(tmp_path):
    """A valid family but a nonexistent adapter directory -> exit 1."""
    result = runner.invoke(app, ["certify",
                                 "--adapter", str(tmp_path / "nope"),
                                 "--base-model", LLAMA_MODEL,
                                 "--root", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "Adapter path does not exist" in result.output


def test_certify_rejects_non_adapter_directory(tmp_path):
    """A directory without adapter markers -> exit 1."""
    not_an_adapter = tmp_path / "some_dir"
    not_an_adapter.mkdir()
    (not_an_adapter / "readme.txt").write_text("hello")
    result = runner.invoke(app, ["certify",
                                 "--adapter", str(not_an_adapter),
                                 "--base-model", LLAMA_MODEL,
                                 "--root", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "Missing adapter_config.json" in result.output


def test_certify_rejects_unknown_family(tmp_path):
    result = runner.invoke(app, ["certify",
                                 "--adapter", str(tmp_path),
                                 "--base-model", "unknown/arch-9000",
                                 "--root", str(tmp_path)])
    assert result.exit_code == 2, result.output
    assert "not a recognized family" in result.output


def test_certify_rejects_gated_family(tmp_path):
    """Qwen is detected but not enabled by default -> exit 2."""
    result = runner.invoke(app, ["certify",
                                 "--adapter", str(tmp_path),
                                 "--base-model", "Qwen/Qwen2.5-7B-Instruct",
                                 "--root", str(tmp_path)])
    assert result.exit_code == 2, result.output
    assert "not among the allowed families" in result.output


def test_certify_missing_canaries_file(tmp_path):
    """A missing --canaries file -> exit 1 (checked before model load)."""
    adapter = _make_adapter(tmp_path / "adapter")
    result = runner.invoke(app, ["certify",
                                 "--adapter", str(adapter),
                                 "--base-model", LLAMA_MODEL,
                                 "--canaries", str(tmp_path / "nope.json"),
                                 "--root", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "Canaries file not found" in result.output


def test_certify_reports_training_tool(tmp_path):
    """The detected training tool is surfaced before any model load."""
    adapter = _make_adapter(tmp_path / "adapter",
                            {"peft_type": "LORA", "unsloth_version": "2024.1"})
    result = runner.invoke(app, ["certify",
                                 "--adapter", str(adapter),
                                 "--base-model", LLAMA_MODEL,
                                 "--root", str(tmp_path)])
    # The validator runs BEFORE the model load; what happens after depends on
    # the environment (GPU/token/gated model), so only the tool line is
    # asserted here.
    assert "training tool: unsloth" in result.output