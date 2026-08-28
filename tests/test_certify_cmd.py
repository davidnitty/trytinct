"""Tests for the certify command (adapter validation, CPU-testable gates only).

Note: the CLI app lives in ``tinct.cli.app`` (there is no ``tinct.cli.main``).
"""

from pathlib import Path

from typer.testing import CliRunner

from tinct.cli.app import app

runner = CliRunner()


def test_certify_validates_adapter_exists():
    result = runner.invoke(app, [
        "certify",
        "--adapter", "/nonexistent/path",
        "--base-model", "meta-llama/Llama-3.1-8B",
    ])

    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_certify_validates_adapter_structure(tmp_path):
    adapter_path = Path(tmp_path) / "adapter"
    adapter_path.mkdir()

    # Empty adapter directory
    result = runner.invoke(app, [
        "certify",
        "--adapter", str(adapter_path),
        "--base-model", "meta-llama/Llama-3.1-8B",
    ])

    assert result.exit_code == 1
    assert "empty" in result.output or "adapter_config.json" in result.output


def test_certify_validates_missing_weights(tmp_path):
    """adapter_config.json present but no weights -> exit 1."""
    adapter_path = Path(tmp_path) / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        '{"peft_type": "LORA"}', encoding="utf-8"
    )

    result = runner.invoke(app, [
        "certify",
        "--adapter", str(adapter_path),
        "--base-model", "meta-llama/Llama-3.1-8B",
    ])

    assert result.exit_code == 1
    assert "adapter_model" in result.output