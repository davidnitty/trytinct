"""Tests for the MoE expert-collapse safety gate (v1.1 Phase 1 — Mixtral vanguard).

The tracker and collapse verdict are exercised against a tiny fake MoE model
built from plain ``torch.nn`` modules that mimic Mixtral's module naming
(``...block_sparse_moe.gate``), so the forward-hook mechanics run for real
on CPU without any model downloads.
"""

import torch
import torch.nn as nn

from tinct.safety.canaries import generate_canaries
from tinct.safety.gates import run_safety_gates
from tinct.safety.moe_gates import (
    MoEExpertTracker,
    check_expert_collapse,
    iter_moe_routers,
)
from tinct.safety.refusal import SAFETY_PROMPTS


# -- fake MoE model (Mixtral-style naming) ------------------------------------

class _Router(nn.Module):
    """Stand-in for Mixtral's ``block_sparse_moe.gate`` (an nn.Linear)."""

    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.gate = nn.Linear(4, num_experts)
        nn.init.zeros_(self.gate.weight)


class _MoELayer(nn.Module):
    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.block_sparse_moe = _Router(num_experts)


class _FakeMoEModel(nn.Module):
    def __init__(self, num_layers: int = 2, num_experts: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_MoELayer(num_experts) for _ in range(num_layers)])


class _FakeDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


# Bias vector that makes the top-2 routing deterministic: every token is
# routed to experts 0 and 1, starving experts 2-7 (guaranteed collapse).
_COLLAPSE_BIAS = [9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _bias_routers(model: nn.Module, bias: list[float]) -> None:
    for layer in model.layers:
        layer.block_sparse_moe.gate.bias = nn.Parameter(torch.tensor(bias))


# -- router detection ----------------------------------------------------------

class TestRouterDetection:
    def test_finds_mixtral_style_routers(self):
        model = _FakeMoEModel(num_layers=2)
        routers = list(iter_moe_routers(model))
        assert len(routers) == 2
        assert all("block_sparse_moe" in name for name, _ in routers)
        assert all(gate.out_features == 8 for _, gate in routers)

    def test_dense_model_has_no_routers(self):
        assert list(iter_moe_routers(_FakeDenseModel())) == []


# -- collapse verdict (pure, over tallies) --------------------------------------

class TestCollapseVerdict:
    def test_not_configured_when_nothing_routed(self):
        tracker = MoEExpertTracker(num_experts=8)
        result = tracker.check_collapse()
        assert result["status"] == "NOT_CONFIGURED"
        assert "reason" in result

    def test_pass_when_all_experts_used(self):
        tracker = MoEExpertTracker(num_experts=8)
        tracker.expert_counts = [100] * 8
        tracker.total_selections = 800
        result = tracker.check_collapse()
        assert result["status"] == "PASS"
        assert result["min_utilization"] == 0.125  # 1/8 uniform

    def test_fail_when_expert_starved(self):
        tracker = MoEExpertTracker(num_experts=8)
        tracker.expert_counts = [100, 100, 100, 100, 100, 100, 100, 0]
        tracker.total_selections = 700
        result = tracker.check_collapse()
        assert result["status"] == "FAIL"
        assert result["laziest_expert_id"] == 7
        assert result["min_utilization"] == 0.0

    def test_report_shape(self):
        tracker = MoEExpertTracker(num_experts=8, top_k=2)
        tracker.expert_counts = [50] * 8
        tracker.total_selections = 400
        result = tracker.check_collapse()
        assert result["num_experts"] == 8
        assert result["total_tokens_routed"] == 200  # 400 selections / top-2
        assert result["threshold"] == 0.01
        assert result["max_utilization"] == 0.125

    def test_custom_threshold(self):
        tracker = MoEExpertTracker(num_experts=8)
        tracker.expert_counts = [90, 90, 90, 90, 10, 10, 10, 10]
        tracker.total_selections = 400
        # min 2.5%: above the default 1% but below a 5% threshold
        assert tracker.check_collapse()["status"] == "PASS"
        assert tracker.check_collapse(min_utilization_threshold=0.05)["status"] == "FAIL"


# -- forward-hook mechanics ------------------------------------------------------

class TestForwardHookTracking:
    def test_hooks_tally_top_k_selections(self):
        model = _FakeMoEModel(num_layers=1)
        _bias_routers(model, [5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tracker = MoEExpertTracker(num_experts=8, top_k=2)
        tracker.attach_to_model(model)
        try:
            x = torch.randn(6, 4)
            _ = model.layers[0].block_sparse_moe.gate(x)
        finally:
            tracker.detach()

        assert tracker.expert_counts[0] == 6
        assert tracker.expert_counts[1] == 6
        assert tracker.total_selections == 12
        assert sum(tracker.expert_counts) == 12

    def test_detach_stops_tallying(self):
        model = _FakeMoEModel(num_layers=1)
        _bias_routers(model, [5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tracker = MoEExpertTracker(num_experts=8, top_k=2)
        tracker.attach_to_model(model)

        x = torch.randn(4, 4)
        _ = model.layers[0].block_sparse_moe.gate(x)
        tracker.detach()
        _ = model.layers[0].block_sparse_moe.gate(x)

        assert tracker.total_selections == 8  # only the first pass counted

    def test_collapse_fail_end_to_end(self):
        model = _FakeMoEModel(num_layers=2)
        _bias_routers(model, _COLLAPSE_BIAS)

        def generate(prompt: str) -> str:
            x = torch.randn(10, 4)
            for layer in model.layers:
                _ = layer.block_sparse_moe.gate(x)
            return "ok"

        result = check_expert_collapse(
            generate, ["p1", "p2"], num_experts=8, top_k=2, model_instance=model
        )

        assert result["status"] == "FAIL"
        assert result["laziest_expert_id"] >= 2
        assert result["total_tokens_routed"] == 40  # 2 layers x 2 prompts x 10 tokens

    def test_hooks_always_cleaned_up(self):
        model = _FakeMoEModel(num_layers=2)
        _bias_routers(model, _COLLAPSE_BIAS)

        def generate(prompt: str) -> str:
            x = torch.randn(4, 4)
            for layer in model.layers:
                _ = layer.block_sparse_moe.gate(x)
            return "ok"

        check_expert_collapse(generate, ["p1"], num_experts=8, model_instance=model)

        for layer in model.layers:
            assert len(layer.block_sparse_moe.gate._forward_hooks) == 0

    def test_dense_model_not_configured(self):
        result = check_expert_collapse(
            lambda p: "ok", ["p1"], num_experts=8, model_instance=_FakeDenseModel()
        )
        assert result["status"] == "NOT_CONFIGURED"

    def test_without_model_instance_not_configured(self):
        result = check_expert_collapse(lambda p: "ok", ["p1"], num_experts=8)
        assert result["status"] == "NOT_CONFIGURED"


# -- pipeline integration (aggregate verdict) ------------------------------------

class TestPipelineIntegration:
    @staticmethod
    def _refusing(prompt: str) -> str:
        if any(s.lower() in prompt.lower() for s in SAFETY_PROMPTS):
            return "I can't help with that."
        return "A perfectly safe and helpful answer."

    def test_extra_gate_fail_fails_aggregate(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        result = run_safety_gates(
            self._refusing,
            self._refusing,
            canaries,
            extra_gates={"expert_collapse": {"status": "FAIL", "min_utilization": 0.0}},
        )
        assert result["expert_collapse"]["status"] == "FAIL"
        assert result["result"] == "FAIL"

    def test_extra_gate_not_configured_ignored(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        result = run_safety_gates(
            self._refusing,
            self._refusing,
            canaries,
            extra_gates={"expert_collapse": {"status": "NOT_CONFIGURED"}},
        )
        assert result["result"] == "PASS"

    def test_no_extra_gates_backward_compatible(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        result = run_safety_gates(self._refusing, self._refusing, canaries)
        assert result["result"] == "PASS"
        assert "expert_collapse" not in result
