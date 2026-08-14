"""Tests for the DPO Reward Inversion core + metrics persistence (no ML deps)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinct.trainers.dpo_trainer import RewardInversionCore, _persist_metrics


def test_import_does_not_require_heavy_deps():
    import tinct.trainers.dpo_trainer  # noqa: F401


def _core(threshold: int = 3) -> RewardInversionCore:
    return RewardInversionCore(inversion_threshold=threshold)


def _state(step: int = 1):
    return SimpleNamespace(global_step=step)


def _control():
    return SimpleNamespace(should_training_stop=False)


# -- observe() --------------------------------------------------------------

def test_observe_healthy_resets_counter():
    core = _core()
    core.observe(0.8, 0.2, 1)   # chosen > rejected -> healthy
    assert core.consecutive_inversions == 0
    assert core.aborted is False


def test_observe_persistent_inversion_aborts():
    core = _core(threshold=3)
    for step in (1, 2):
        core.observe(0.1, 0.9, step)
        assert core.aborted is False
    core.observe(0.1, 0.9, 3)  # third consecutive inversion
    assert core.aborted is True
    assert core.reason == "reward_inversion"


def test_observe_single_inversion_is_noise():
    core = _core(threshold=3)
    core.observe(0.1, 0.9, 1)
    assert core.consecutive_inversions == 1
    core.observe(0.9, 0.1, 2)  # healthy step resets
    assert core.consecutive_inversions == 0


def test_observe_missing_rewards_ignored():
    core = _core()
    core.observe(None, None, 1)
    assert core.margins == []
    assert core.aborted is False


def test_on_log_extracts_and_stops_trainer():
    core = _core(threshold=1)
    control = _control()
    halted = core.on_log(_state(5), control,
                         {"rewards_chosen": 0.0, "rewards_rejected": 0.5})
    assert halted is True
    assert control.should_training_stop is True
    assert core.aborted is True


# -- final_metrics() --------------------------------------------------------

def test_final_metrics_healthy():
    core = _core()
    core.observe(0.6, 0.4, 1)
    core.observe(0.9, 0.2, 2)
    m = core.final_metrics()
    assert m is not None
    assert m["training_method"] == "dpo"
    assert m["final_reward_margin"] == pytest.approx(0.7)
    assert m["max_reward_margin"] == pytest.approx(0.7)
    assert m["min_reward_margin"] == pytest.approx(0.2)
    assert m["num_logged_steps"] == 2
    assert m["reward_inversion_detected"] is False


def test_final_metrics_inverted_records_reason():
    core = _core(threshold=1)
    core.observe(0.1, 0.9, 1)
    m = core.final_metrics()
    assert m is not None
    assert m["reward_inversion_detected"] is True
    assert m["final_reward_margin"] == pytest.approx(-0.8)


def test_final_metrics_no_rewards_returns_none():
    assert _core().final_metrics() is None


# -- persistence ------------------------------------------------------------

def test_metrics_persisted_on_halt(tmp_path: Path):
    core = _core(threshold=1)
    core.observe(0.0, 1.0, 1)  # immediate inversion -> abort
    assert core.aborted is True
    _persist_metrics(core, tmp_path)
    path = tmp_path / "dpo_metrics.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["reward_inversion_detected"] is True
    assert data["final_reward_margin"] < 0


def test_persist_without_rewards_writes_nothing(tmp_path: Path):
    _persist_metrics(_core(), tmp_path)
    assert not (tmp_path / "dpo_metrics.json").exists()
