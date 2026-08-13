"""Tests for the fail-closed SFT trainer (no heavy ML deps required)."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinct.trainers.sft_trainer import FailClosedCore, loss_is_fatal


def test_import_does_not_require_heavy_deps():
    # No torch/trl/peft/transformers at module scope.
    import tinct.trainers.sft_trainer  # noqa: F401


# -- fail-closed guard (the V0.1-GPU smoke-test logic, no GPU needed) --------

def _state(step: int = 3):
    return SimpleNamespace(global_step=step)


def _control():
    return SimpleNamespace(should_training_stop=False)


def test_guard_aborts_on_over_threshold_loss(tmp_path: Path):
    core = FailClosedCore(threshold=0.1, log_path=tmp_path / "log.jsonl",
                          fail_path=tmp_path / "fail_state.json")
    control = _control()
    halted = core.on_log(_state(step=7), control, {"loss": 5.3})

    assert halted is True
    assert core.aborted is True
    assert control.should_training_stop is True
    # Failure evidence written for the ship gate.
    fail = json.loads((tmp_path / "fail_state.json").read_text(encoding="utf-8"))
    assert fail["reason"] == "loss_explosion"
    assert fail["step"] == 7
    # Structured log populated.
    line = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["step"] == 7
    assert line["loss"] == 5.3


def test_guard_aborts_on_nan(tmp_path: Path):
    core = FailClosedCore(threshold=10.0, log_path=tmp_path / "log.jsonl",
                          fail_path=tmp_path / "fail_state.json")
    halted = core.on_log(_state(), _control(), {"loss": float("nan")})
    assert halted is True
    assert core.aborted is True


def test_guard_passes_on_normal_loss(tmp_path: Path):
    core = FailClosedCore(threshold=10.0, log_path=tmp_path / "log.jsonl",
                          fail_path=tmp_path / "fail_state.json")
    halted = core.on_log(_state(), _control(), {"loss": 1.2, "learning_rate": 2e-4})
    assert halted is False
    assert core.aborted is False
    assert not (tmp_path / "fail_state.json").exists()
    # But the structured log still populates (for evidence).
    assert (tmp_path / "log.jsonl").is_file()


def test_guard_ignores_empty_logs(tmp_path: Path):
    core = FailClosedCore(threshold=10.0, log_path=tmp_path / "log.jsonl",
                          fail_path=tmp_path / "fail_state.json")
    assert core.on_log(_state(), _control(), None) is False
    assert core.aborted is False


# -- loss_is_fatal predicate -------------------------------------------------

def test_loss_is_fatal_nan():
    assert loss_is_fatal(float("nan"), 10.0)


def test_loss_is_fatal_inf():
    assert loss_is_fatal(float("inf"), 10.0)
    assert loss_is_fatal(float("-inf"), 10.0)


def test_loss_is_fatal_above_threshold():
    assert loss_is_fatal(11.0, 10.0)


def test_loss_below_threshold_is_fine():
    assert not loss_is_fatal(3.2, 10.0)
    assert not loss_is_fatal(10.0, 10.0)  # equal to threshold is allowed


def test_threshold_from_init_default():
    # The default (as wired from project.yaml) is 10.0.
    assert not loss_is_fatal(9.99, 10.0)
    assert loss_is_fatal(10.01, 10.0)