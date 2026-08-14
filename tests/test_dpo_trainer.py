"""Tests for the DPO trainer guard core (no ML deps required)."""

import json
from pathlib import Path
from types import SimpleNamespace

from tinct.trainers.dpo_trainer import RewardInversionCore


def test_import_does_not_require_heavy_deps():
    import tinct.trainers.dpo_trainer  # noqa: F401


def _state(step=1):
    return SimpleNamespace(global_step=step)


def _control():
    return SimpleNamespace(should_training_stop=False)


def _core(tmp_path: Path, threshold=3):
    return RewardInversionCore(
        log_path=tmp_path / "train_log.jsonl",
        fail_path=tmp_path / "fail_state.json",
        threshold=threshold,
    )


def test_healthy_preference_resets_counter(tmp_path: Path):
    core = _core(tmp_path)
    # Chosen > rejected -> healthy.
    assert core.on_log(_state(1), _control(), {"rewards/chosen": 0.8, "rewards/rejected": 0.2}) is False
    assert core.inversion_count == 0
    assert not (tmp_path / "fail_state.json").exists()


def test_persistent_inversion_halts(tmp_path: Path):
    core = _core(tmp_path, threshold=3)
    control = _control()
    for step in (1, 2):
        assert core.on_log(_state(step), control,
                           {"rewards/chosen": 0.1, "rewards/rejected": 0.9}) is False
    # Third consecutive inversion -> fatal.
    halted = core.on_log(_state(3), control,
                         {"rewards/chosen": 0.1, "rewards/rejected": 0.9})
    assert halted is True
    assert core.aborted is True
    assert control.should_training_stop is True

    fail = json.loads((tmp_path / "fail_state.json").read_text(encoding="utf-8"))
    assert fail["reason"] == "reward_inversion"
    assert fail["step"] == 3
    assert fail["rejected_reward"] == 0.9


def test_single_inversion_is_noise(tmp_path: Path):
    core = _core(tmp_path, threshold=3)
    # One inversion, then a healthy step resets the counter.
    assert core.on_log(_state(1), _control(),
                       {"rewards/chosen": 0.1, "rewards/rejected": 0.9}) is False
    assert core.inversion_count == 1
    assert core.on_log(_state(2), _control(),
                       {"rewards/chosen": 0.9, "rewards/rejected": 0.1}) is False
    assert core.inversion_count == 0


def test_missing_rewards_ignored(tmp_path: Path):
    core = _core(tmp_path)
    assert core.on_log(_state(), _control(), {"loss": 1.2}) is False
    assert core.inversion_count == 0


def test_accepts_trl_1x_spellings(tmp_path: Path):
    core = _core(tmp_path, threshold=1)
    halted = core.on_log(_state(5), _control(),
                         {"rewards_chosen": 0.0, "rewards_rejected": 0.5})
    assert halted is True
    assert core.aborted is True
