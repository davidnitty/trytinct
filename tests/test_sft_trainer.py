"""Tests for the fail-closed SFT trainer (no heavy ML deps required)."""

import math

import pytest

from tinct.trainers.sft_trainer import loss_is_fatal


def test_import_does_not_require_heavy_deps():
    # No torch/trl/peft/transformers at module scope.
    import tinct.trainers.sft_trainer  # noqa: F401


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