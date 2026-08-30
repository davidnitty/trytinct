"""Tests for the MoE expert-collapse safety gate (v1.1 Phase 1 — Mixtral vanguard).

Two mock strategies, both running on CPU without any model downloads:

- A fake MoE model built from plain ``torch.nn`` modules that mimic Mixtral's
  module naming (``...block_sparse_moe.gate``) — real Linear routers, real
  forward-hook mechanics.
- A *scripted* router whose forward returns deterministic logits, so routing
  behavior (collapse vs. perfectly uniform) is guaranteed rather than
  weight-dependent.
"""

import pytest

torch = pytest.importorskip("torch")  # MoE tests need torch; skip cleanly without it
import torch.nn as nn

from tinct.engine.moe import iter_moe_experts, iter_moe_routers
from tinct.safety.canaries import generate_canaries
from tinct.safety.gates import run_safety_gates
from tinct.safety.moe_gates import (
    MoEExpertTracker,
    check_expert_collapse,
    check_routing_regression,
)
from tinct.safety.refusal import SAFETY_PROMPTS


# -- fake MoE model (Mixtral-style naming, real Linear routers) -----------------

class _DummyExpert(nn.Module):
    """Structural placeholder: an expert submodule with no routing logic."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)


class _Router(nn.Module):
    """Structural MoE block: a router (``gate``) plus an ``experts`` list."""

    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.gate = nn.Linear(4, num_experts)
        nn.init.zeros_(self.gate.weight)
        self.experts = nn.ModuleList([_DummyExpert() for _ in range(num_experts)])


class _MoELayer(nn.Module):
    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.block_sparse_moe = _Router(num_experts)


class _FakeMoEModel(nn.Module):
    def __init__(self, num_layers: int = 2, num_experts: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_MoELayer(num_experts) for _ in range(num_layers)])


class _QwenStyleLayer(nn.Module):
    """Qwen-MoE / DeepSeek style: the MoE block is the layer's ``mlp``."""

    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.mlp = _Router(num_experts)


class _QwenStyleModel(nn.Module):
    def __init__(self, num_layers: int = 2, num_experts: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_QwenStyleLayer(num_experts) for _ in range(num_layers)])


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


# -- scripted router (deterministic routing, weight-independent) ----------------

class _ScriptedRouterGate(nn.Module):
    """Router gate whose forward returns deterministic logits.

    ``collapsed=True``  -> every token routes to experts 0 and 1.
    ``collapsed=False`` -> the top-2 selection cycles across ``route_experts``
    experts, so routing is exactly uniform when the token count is a multiple
    of ``route_experts``. Routing over fewer experts than declared starves the
    rest (adapter starvation).
    """

    def __init__(self, num_experts: int = 8, collapsed: bool = True,
                 route_experts: int | None = None):
        super().__init__()
        self.num_experts = num_experts
        self.collapsed = collapsed
        self.route_experts = route_experts or num_experts

    def forward(self, x):
        tokens = x.shape[0]
        logits = torch.full((tokens, self.num_experts), -100.0)
        if self.collapsed:
            logits[:, 0] = 100.0
            logits[:, 1] = 99.0
        else:
            rows = torch.arange(tokens)
            first = rows % self.route_experts
            logits[rows, first] = 100.0
            logits[rows, (first + 1) % self.route_experts] = 99.0
        return logits


class _ScriptedMoEBlock(nn.Module):
    def __init__(self, num_experts: int = 8, collapsed: bool = True,
                 route_experts: int | None = None):
        super().__init__()
        self.gate = _ScriptedRouterGate(num_experts, collapsed, route_experts)
        self.experts = nn.ModuleList([_DummyExpert() for _ in range(num_experts)])

    def forward(self, x):
        return self.gate(x)


class _MockMoEBlock(nn.Module):
    def __init__(self, num_experts: int = 8, collapsed: bool = True):
        super().__init__()
        self.block_sparse_moe = _ScriptedMoEBlock(num_experts, collapsed)

    def forward(self, x):
        return self.block_sparse_moe(x)


class _MockMoEModel(nn.Module):
    def __init__(self, num_layers: int = 1, num_experts: int = 8, collapsed: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(
            [_MockMoEBlock(num_experts, collapsed) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Top-2 pairs for "skip_three" mode: cycles over every expert except 3.
_SKIP_FIRST = [0, 1, 2, 4, 5, 6, 7]
_SKIP_SECOND = [1, 2, 4, 5, 6, 7, 0]


class _SwitchableRouterGate(nn.Module):
    """Scripted gate whose routing mode flips between measurement passes."""

    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.mode = "uniform"  # "uniform" | "starve_high" | "skip_three"

    def forward(self, x):
        tokens = x.shape[0]
        rows = torch.arange(tokens)
        logits = torch.full((tokens, self.num_experts), -100.0)
        if self.mode == "uniform":
            first = rows % self.num_experts
            second = (first + 1) % self.num_experts
        elif self.mode == "starve_high":
            first = rows % 6  # cycle over experts 0-5 only
            second = (first + 1) % 6
        else:  # "skip_three": top-2 pairs never touch expert 3
            idx = (rows % 7).tolist()
            first = torch.tensor([_SKIP_FIRST[i] for i in idx])
            second = torch.tensor([_SKIP_SECOND[i] for i in idx])
        logits[rows, first] = 100.0
        logits[rows, second] = 99.0
        return logits


class _SwitchableMoEBlock(nn.Module):
    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.gate = _SwitchableRouterGate(num_experts)
        self.experts = nn.ModuleList([_DummyExpert() for _ in range(num_experts)])

    def forward(self, x):
        return self.gate(x)


class _SwitchableMoEModel(nn.Module):
    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_SwitchableMoEBlock(num_experts)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _cycling_logits(tokens: int, num_experts: int = 8) -> torch.Tensor:
    """Logits whose top-2 cycles across all experts (exactly uniform routing)."""
    rows = torch.arange(tokens)
    logits = torch.full((tokens, num_experts), -100.0)
    logits[rows, rows % num_experts] = 100.0
    logits[rows, (rows % num_experts + 1) % num_experts] = 99.0
    return logits


# -- router detection ----------------------------------------------------------

class TestRouterDetection:
    """Structural detection: gate + experts, independent of module names."""

    def test_finds_mixtral_style_routers(self):
        model = _FakeMoEModel(num_layers=2)
        routers = list(iter_moe_routers(model))
        assert len(routers) == 2
        assert all("block_sparse_moe" in name for name, _ in routers)
        # Structural: router linear + declared expert list on every block.
        assert all(len(block.experts) == 8 for _, block in routers)
        assert all(block.gate.out_features == 8 for _, block in routers)

    def test_finds_qwen_style_routers(self):
        model = _QwenStyleModel(num_layers=2)  # layers.N.mlp, no block_sparse_moe
        routers = list(iter_moe_routers(model))
        assert len(routers) == 2
        assert all(name.endswith(".mlp") for name, _ in routers)
        assert all(len(block.experts) == 8 for _, block in routers)

    def test_iter_moe_experts_covers_every_expert(self):
        experts = dict(iter_moe_experts(_FakeMoEModel(num_layers=2)))
        assert len(experts) == 16
        assert "layers.0.block_sparse_moe.experts.3" in experts
        assert "layers.1.block_sparse_moe.experts.7" in experts

    def test_dense_model_has_no_routers(self):
        assert list(iter_moe_routers(_FakeDenseModel())) == []
        assert dict(iter_moe_experts(_FakeDenseModel())) == {}


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


# -- hook function over raw logits ----------------------------------------------

class TestHookFnOverRawLogits:
    def test_detects_expert_collapse(self):
        tracker = MoEExpertTracker(num_experts=8, top_k=2)

        # Simulate router output: 100 tokens, all routing to experts 0 and 1.
        fake_logits = torch.full((100, 8), -100.0)
        fake_logits[:, 0] = 100.0
        fake_logits[:, 1] = 99.0
        tracker._hook_fn(None, None, fake_logits)

        result = tracker.check_collapse(min_utilization_threshold=0.01)
        assert result["status"] == "FAIL"
        assert result["laziest_expert_id"] in [2, 3, 4, 5, 6, 7]
        assert result["min_utilization"] == 0.0

    def test_passes_healthy_routing(self):
        tracker = MoEExpertTracker(num_experts=8, top_k=2)

        # 104 tokens cycling the top-2 across all 8 experts: 13 first picks +
        # 13 second picks = 26 selections each -> exactly 1/8 of 208.
        tracker._hook_fn(None, None, _cycling_logits(104))

        result = tracker.check_collapse(min_utilization_threshold=0.01)
        assert result["status"] == "PASS"
        assert result["min_utilization"] == 0.125
        assert result["max_utilization"] == 0.125


# -- routing regression (adapter vs base utilization) ----------------------------

class TestRoutingRegression:
    """check_routing_regression: flags experts the adapter starved relative
    to the BASE model, with a floor so base-model laziness isn't blamed on
    the adapter."""

    @staticmethod
    def _make_passes(model, adapter_mode: str, base_mode: str):
        gate = model.layers[0].gate

        def adapter_pass(prompt: str) -> str:
            gate.mode = adapter_mode
            _ = model(torch.randn(104, 4))
            return "ok"

        def base_pass(prompt: str) -> str:
            gate.mode = base_mode
            _ = model(torch.randn(104, 4))
            return "ok"

        return adapter_pass, base_pass

    def test_healthy_routing_passes_with_parity(self):
        model = _SwitchableMoEModel()
        adapter_pass, base_pass = self._make_passes(model, "uniform", "uniform")
        result = check_routing_regression(
            adapter_pass, base_pass, ["p1"], num_experts=8, top_k=2, model_instance=model,
        )
        assert result["status"] == "PASS"
        assert result["regressed_experts"] == []
        assert all(r == 1.0 for r in result["adapter_over_base_ratio"])

    def test_adapter_starvation_fails(self):
        # Base routes uniformly over all 8; adapter starves experts 6 and 7.
        model = _SwitchableMoEModel()
        adapter_pass, base_pass = self._make_passes(model, "starve_high", "uniform")
        result = check_routing_regression(
            adapter_pass, base_pass, ["p1"], num_experts=8, top_k=2, model_instance=model,
        )
        assert result["status"] == "FAIL"
        assert result["regressed_experts"] == [6, 7]
        assert result["adapter_utilization"][6] == 0.0
        assert result["base_utilization"][6] == 0.125

    def test_floor_protects_base_starved_experts(self):
        # Base already skips expert 3 entirely; adapter starves 6 and 7.
        # Expert 3 must NOT be blamed (below base_floor), 6 and 7 must be.
        model = _SwitchableMoEModel()
        adapter_pass, base_pass = self._make_passes(model, "starve_high", "skip_three")
        result = check_routing_regression(
            adapter_pass, base_pass, ["p1"], num_experts=8, top_k=2, model_instance=model,
        )
        assert result["status"] == "FAIL"
        assert result["regressed_experts"] == [6, 7]
        assert 3 not in result["regressed_experts"]

    def test_reset_isolates_measurement_passes(self):
        tracker = MoEExpertTracker(num_experts=8, top_k=2)
        logits = _cycling_logits(104)

        tracker._hook_fn(None, None, logits)
        first_counts = list(tracker.expert_counts)
        assert tracker.total_selections == 208

        tracker.reset()
        assert tracker.total_selections == 0
        assert tracker.expert_counts == [0] * 8

        tracker._hook_fn(None, None, logits)
        assert tracker.total_selections == 208  # per-pass tally, not summed
        assert tracker.expert_counts == first_counts

    def test_without_model_instance_not_configured(self):
        result = check_routing_regression(
            lambda p: "ok", lambda p: "ok", ["p1"], num_experts=8
        )
        assert result["status"] == "NOT_CONFIGURED"

    def test_no_routed_tokens_not_configured(self):
        # No routers in the model -> nothing tallies in either pass.
        result = check_routing_regression(
            lambda p: "ok", lambda p: "ok", ["p1"], num_experts=8,
            model_instance=_FakeDenseModel(),
        )
        assert result["status"] == "NOT_CONFIGURED"


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

    def test_attach_and_detach_scripted_router(self):
        model = _MockMoEModel(num_layers=1, collapsed=True)
        tracker = MoEExpertTracker(num_experts=8, top_k=2)

        assert tracker.attach_to_model(model) == 1
        assert len(tracker.hooks) == 1

        # Trigger the forward pass through the scripted router.
        _ = model(torch.randn(5, 10))
        assert tracker.total_selections == 10  # 5 tokens * top_k=2

        tracker.detach()
        assert len(tracker.hooks) == 0

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

    def test_healthy_routing_passes_end_to_end(self):
        # 104 tokens per forward keep the cycled top-2 exactly uniform.
        model = _MockMoEModel(num_layers=1, collapsed=False)

        def generate(prompt: str) -> str:
            _ = model(torch.randn(104, 10))
            return "ok"

        result = check_expert_collapse(
            generate, ["p1"], num_experts=8, top_k=2, model_instance=model
        )

        assert result["status"] == "PASS"
        assert result["min_utilization"] == 0.125
        assert result["max_utilization"] == 0.125

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

    @staticmethod
    def _run_routers(model):
        def generate(prompt: str) -> str:
            x = torch.randn(16, 4)
            for layer in model.layers:
                _ = layer.block_sparse_moe.gate(x)
            return "A perfectly safe and helpful answer."

        return generate

    def test_moe_gate_fail_fails_aggregate(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        collapsed_model = _FakeMoEModel(num_layers=1)
        _bias_routers(collapsed_model, _COLLAPSE_BIAS)

        result = run_safety_gates(
            self._run_routers(collapsed_model),
            self._refusing,
            canaries,
            is_moe=True,
            model_instance=collapsed_model,
            num_experts=8,
        )

        assert result["expert_collapse"]["status"] == "FAIL"
        assert result["result"] == "FAIL"

    def test_moe_gate_pass_keeps_aggregate_pass(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        healthy_model = _MockMoEModel(num_layers=1, collapsed=False)

        def generate(prompt: str) -> str:
            _ = healthy_model(torch.randn(104, 10))
            # Refuse safety prompts like the base model, or the refusal
            # gate (not the MoE gate) would fail the aggregate.
            if any(s.lower() in prompt.lower() for s in SAFETY_PROMPTS):
                return "I can't help with that."
            return "A perfectly safe and helpful answer."

        result = run_safety_gates(
            generate,
            self._refusing,
            canaries,
            is_moe=True,
            model_instance=healthy_model,
            num_experts=8,
        )

        assert result["expert_collapse"]["status"] == "PASS"
        assert result["result"] == "PASS"

    def test_moe_model_without_routers_not_configured(self):
        # Registry says MoE, but the loaded model has no routers (mismatch):
        # the gate reports NOT_CONFIGURED, which never fails the run.
        canaries = generate_canaries(num_canaries=3, seed=0)
        result = run_safety_gates(
            self._refusing,
            self._refusing,
            canaries,
            is_moe=True,
            model_instance=_FakeDenseModel(),
            num_experts=8,
        )
        assert result["expert_collapse"]["status"] == "NOT_CONFIGURED"
        assert result["result"] == "PASS"

    def test_dense_models_skip_the_gate(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        result = run_safety_gates(self._refusing, self._refusing, canaries)
        assert result["result"] == "PASS"
        assert "expert_collapse" not in result

    def test_routing_regression_fail_fails_aggregate(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        model = _SwitchableMoEModel()

        def adapter(prompt: str) -> str:
            model.layers[0].gate.mode = "starve_high"
            _ = model(torch.randn(104, 4))
            return "I can't help with that."

        def base(prompt: str) -> str:
            model.layers[0].gate.mode = "uniform"
            _ = model(torch.randn(104, 4))
            return "I can't help with that."

        result = run_safety_gates(
            adapter, base, canaries,
            is_moe=True, model_instance=model, num_experts=8,
        )
        assert result["routing_regression"]["status"] == "FAIL"
        assert result["result"] == "FAIL"
