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
              eval_status: str | None = None, fail_state: bool = False,
              dpo_metrics: dict | None = None):
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
    if dpo_metrics is not None:
        (run / "dpo_metrics.json").write_text(_j.dumps(dpo_metrics))
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


def test_security_check_run_verifies_signature(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    _seed_run(proj, "run_1", final_eval_loss=1.2, eval_status="PASS")
    assert runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)]).exit_code == 0
    result = runner.invoke(app, ["security", "check", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 0, result.output
    assert "Signature valid" in result.output


def test_security_check_run_missing_evidence(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    result = runner.invoke(app, ["security", "check", "--run", "nope", "--root", str(proj)])
    assert result.exit_code == 1


# -- DPO certification gates -------------------------------------------------

_HEALTHY_DPO = {
    "training_method": "dpo",
    "final_chosen_reward": 1.2, "final_rejected_reward": 0.3,
    "final_reward_margin": 0.9, "max_reward_margin": 0.9,
    "min_reward_margin": 0.5, "num_logged_steps": 3,
    "reward_inversion_detected": False,
}


def test_ship_blocks_inverted_dpo_run(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    dpo = dict(_HEALTHY_DPO, reward_inversion_detected=True, final_reward_margin=-0.8)
    _seed_run(proj, "run_1", 1.2, eval_status="PASS", dpo_metrics=dpo)
    result = runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 2
    assert "reward inversion" in result.output


def test_ship_blocks_negative_margin(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    dpo = dict(_HEALTHY_DPO, final_reward_margin=-0.1)
    _seed_run(proj, "run_1", 1.2, eval_status="PASS", dpo_metrics=dpo)
    result = runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 2
    assert "non-positive reward margin" in result.output


def test_ship_includes_dpo_metrics_in_evidence(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    _seed_run(proj, "run_1", 1.2, eval_status="PASS", dpo_metrics=_HEALTHY_DPO)
    result = runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 0, result.output
    import json as _j
    ev = _j.loads((proj / ".tinct" / "evidence" / "run_1_evidence.json").read_text())
    assert ev["metrics"]["training_method"] == "dpo"
    assert ev["metrics"]["dpo_metrics"]["final_reward_margin"] == 0.9
    assert "dpo_metrics.json" in ev["artifacts"]


def test_ship_sft_unchanged_without_dpo_metrics(tmp_path: Path):
    proj = tmp_path / "proj"
    runner.invoke(app, ["init", "proj", "meta-llama/Llama-3.1-8B", "--root", str(tmp_path)])
    _seed_run(proj, "run_1", 1.2, eval_status="PASS")  # no dpo_metrics.json
    result = runner.invoke(app, ["ship", "--run", "run_1", "--root", str(proj)])
    assert result.exit_code == 0, result.output
    import json as _j
    ev = _j.loads((proj / ".tinct" / "evidence" / "run_1_evidence.json").read_text())
    assert ev["metrics"]["training_method"] == "sft"
    assert ev["metrics"]["dpo_metrics"] is None

