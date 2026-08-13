"""CLI integration tests using typer.testing.CliRunner (no ML deps needed)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from tinct.cli.app import app

runner = CliRunner()


def _write_dataset(path: Path, n: int = 40):
    rows = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_init(tmp_path: Path):
    result = runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "proj" / ".tinct" / "project.yaml").is_file()
    assert (tmp_path / "proj" / ".tinct" / "keys").is_dir()


def test_init_no_key(tmp_path: Path):
    result = runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B",
                                 "--root", str(tmp_path), "--no-key"])
    assert result.exit_code == 0
    keys = tmp_path / "proj" / ".tinct" / "keys"
    assert not list(keys.glob("*_private.pem"))


def test_validate_passes(tmp_path: Path):
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    data = tmp_path / "proj" / "data.jsonl"
    _write_dataset(data)
    result = runner.invoke(app, ["validate", str(data), "--root", str(tmp_path / "proj")])
    assert result.exit_code == 0, result.output


def test_validate_blocks_bad_data(tmp_path: Path):
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    data = tmp_path / "proj" / "bad.jsonl"
    data.write_text(json.dumps({}) + "\n")
    result = runner.invoke(app, ["validate", str(data), "--root", str(tmp_path / "proj")])
    assert result.exit_code != 0
    assert "FAIL" in result.output


def test_security_check_on_fresh_project(tmp_path: Path):
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    result = runner.invoke(app, ["security", "check", "--root", str(tmp_path / "proj")])
    assert result.exit_code == 0, result.output


def _seed_run(proj: Path, name: str, final_eval_loss: float,
              eval_status: str | None = None, fail_state: bool = False):
    run = proj / ".tinct" / "runs" / name
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_model.safetensors").write_text("x")
    (run / "train.jsonl").write_text("")
    (run / "valid.jsonl").write_text("")
    import json as _j
    (run / "metrics.json").write_text(
        _j.dumps([{"step": 0, "eval_loss": 9.9}, {"step": 1, "eval_loss": final_eval_loss}])
    )
    if eval_status is not None:
        (run / "eval_report.json").write_text(
            _j.dumps({"gate": "generation_smoke_test", "status": eval_status,
                      "empty_responses": 0, "repetitive_responses": 0, "details": []})
        )
    if fail_state:
        (run / "fail_state.json").write_text(
            _j.dumps({"reason": "loss_explosion", "value": "10.38"})
        )


def test_ship_ships_on_passing_gate(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    _seed_run(proj, "run_1", final_eval_loss=1.2, eval_status="PASS")
    result = runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 0, result.output
    assert "SHIP" in result.output
    assert "Evidence signed" in result.output
    assert (proj / ".tinct" / "evidence" / "run_1_evidence.json").is_file()


def test_ship_refuses_if_fail_state_present(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    _seed_run(proj, "run_1", final_eval_loss=1.2, eval_status="PASS", fail_state=True)
    result = runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 2
    assert "DON'T_SHIP" in result.output


def test_ship_requires_eval_report(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    _seed_run(proj, "run_1", final_eval_loss=1.2)  # no eval_report.json
    result = runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 1
    assert "Run `tinct eval` before shipping" in result.output


def test_ship_refuses_if_eval_failed(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    _seed_run(proj, "run_1", final_eval_loss=1.2, eval_status="FAIL")
    result = runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 2
    assert "DON'T_SHIP" in result.output

