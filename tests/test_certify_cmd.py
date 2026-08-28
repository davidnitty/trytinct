"""Tests for the certify command (adapter validation, CPU-testable gates only).

Note: the CLI app lives in ``tinct.cli.app`` (there is no ``tinct.cli.main``).
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from tinct.cli.app import app

runner = CliRunner()


class TestCertifyCommand:
    """Tests for the tinct certify command."""

    def test_certify_nonexistent_adapter(self):
        result = runner.invoke(app, [
            "certify",
            "--adapter", "/nonexistent/path/to/adapter",
            "--base-model", "meta-llama/Llama-3.1-8B",
        ])

        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_certify_empty_adapter_directory(self, tmp_path):
        adapter_path = Path(tmp_path) / "adapter"
        adapter_path.mkdir()

        result = runner.invoke(app, [
            "certify",
            "--adapter", str(adapter_path),
            "--base-model", "meta-llama/Llama-3.1-8B",
        ])

        assert result.exit_code == 1
        assert "adapter_config.json" in result.output

    def test_certify_missing_weights(self, tmp_path):
        adapter_path = Path(tmp_path) / "adapter"
        adapter_path.mkdir()

        # Only config, no weights
        (adapter_path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA"}), encoding="utf-8"
        )

        result = runner.invoke(app, [
            "certify",
            "--adapter", str(adapter_path),
            "--base-model", "meta-llama/Llama-3.1-8B",
        ])

        assert result.exit_code == 1
        assert "adapter_model" in result.output
