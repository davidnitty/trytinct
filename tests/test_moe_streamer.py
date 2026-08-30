"""Tests for the MoE expert-offloading engine (v1.1 Phase 1 — Mixtral vanguard).

Per the MoEStreamer design principles, all bookkeeping is CPU-testable: tests
run with ``device="cpu"`` and verify behavior through ``placement_log`` and
``stats`` rather than real device inspection. The fake model uses Mixtral-exact
module naming (``block_sparse_moe.experts.N``) so ``iter_moe_experts`` matches
genuinely.
"""

import torch
import torch.nn as nn

import pytest

from tinct.engine.moe import MoEStreamer, ExpertLRUCache, iter_moe_experts, iter_moe_routers


# -- fake model (Mixtral-exact naming) ------------------------------------------

class FakeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


class FakeMoEBlock(nn.Module):
    def __init__(self, n=8, collapsed_bias=None):
        super().__init__()
        self.gate = nn.Linear(4, n)
        self.experts = nn.ModuleList([FakeExpert() for _ in range(n)])
        if collapsed_bias is not None:
            nn.init.zeros_(self.gate.weight)
            self.gate.bias = nn.Parameter(torch.tensor(collapsed_bias, dtype=torch.float32))

    def forward(self, x):
        idx = torch.topk(self.gate(x), 2, dim=-1).indices
        out = torch.zeros_like(x)
        for b in range(x.shape[0]):
            for k in idx[b]:
                out = out + self.experts[k](x[b:b + 1])
        return out


class FakeLayer(nn.Module):
    def __init__(self, collapsed_bias=None):
        super().__init__()
        self.block_sparse_moe = FakeMoEBlock(collapsed_bias=collapsed_bias)

    def forward(self, x):
        return self.block_sparse_moe(x)


class FakeModel(nn.Module):
    def __init__(self, collapsed_bias=None):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.layers = nn.ModuleList([FakeLayer(collapsed_bias) for _ in range(2)])

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return h


# Bias that pins the top-2 to experts 0 and 1 of every layer: the model has
# exactly 4 "hot" experts, which makes stream/evict/hit counts deterministic.
_COLLAPSE_BIAS = [9.0, 8.0] + [-100.0] * 6

_NUM_EXPERTS = 16  # 2 layers x 8 experts


# -- Qwen-MoE / DeepSeek-style fake (MoE block lives at `<layer>.mlp`) ----------

class _QwenStyleBlock(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.gate = nn.Linear(4, n)
        self.experts = nn.ModuleList([FakeExpert() for _ in range(n)])


class _QwenStyleLayer(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.mlp = _QwenStyleBlock(n)


class _QwenStyleModel(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.layers = nn.ModuleList([_QwenStyleLayer(n) for _ in range(2)])


# -- structural router detection -------------------------------------------------

class TestStructuralDetection:
    """Router detection is structural (gate + experts), not name-based."""

    def test_mixtral_style_blocks_detected(self):
        routers = dict(iter_moe_routers(FakeModel()))
        assert len(routers) == 2
        assert all(name.endswith("block_sparse_moe") for name in routers)
        assert all(len(block.experts) == 8 for block in routers.values())

    def test_qwen_style_blocks_detected(self):
        routers = dict(iter_moe_routers(_QwenStyleModel()))
        assert len(routers) == 2
        assert all(name.endswith(".mlp") for name in routers)
        assert all(len(block.experts) == 8 for block in routers.values())

    def test_expert_paths_are_router_name_agnostic(self):
        experts = dict(iter_moe_experts(_QwenStyleModel()))
        assert len(experts) == 16
        assert "layers.0.mlp.experts.5" in experts
        assert "layers.1.mlp.experts.0" in experts

    def test_expert_count_matches_declared_list(self):
        for block in dict(iter_moe_routers(FakeModel())).values():
            assert len(block.experts) == len(list(block.experts)) == 8

    def test_dense_model_has_no_routers_or_experts(self):
        dense = nn.Linear(4, 4)
        assert list(iter_moe_routers(dense)) == []
        assert dict(iter_moe_experts(dense)) == {}


# -- ExpertLRUCache (pure bookkeeping) -------------------------------------------

class TestExpertLRUCache:
    def test_rejects_invalid_capacity(self):
        with pytest.raises(ValueError):
            ExpertLRUCache(0)

    def test_admit_within_capacity_evicts_nothing(self):
        cache = ExpertLRUCache(2)
        assert cache.admit("a") == []
        assert cache.admit("b") == []
        assert cache.resident() == {"a", "b"}

    def test_admit_full_cache_evicts_lru(self):
        cache = ExpertLRUCache(2)
        cache.admit("a")
        cache.admit("b")
        assert cache.admit("c") == ["a"]  # least-recently-used first
        assert cache.resident() == {"b", "c"}

    def test_touch_refreshes_recency(self):
        cache = ExpertLRUCache(2)
        cache.admit("a")
        cache.admit("b")
        cache.touch("a")  # a is now most-recently-used
        assert cache.admit("c") == ["b"]
        assert cache.resident() == {"a", "c"}

    def test_readmit_resident_key_evicts_nothing(self):
        cache = ExpertLRUCache(2)
        cache.admit("a")
        cache.admit("b")
        assert cache.admit("a") == []
        assert cache.resident() == {"a", "b"}

    def test_is_resident(self):
        cache = ExpertLRUCache(1)
        cache.admit("x")
        assert cache.is_resident("x") is True
        assert cache.is_resident("y") is False


# -- expert discovery -------------------------------------------------------------

class TestIterMoeExperts:
    def test_finds_all_mixtral_style_experts(self):
        model = FakeModel()
        experts = dict(iter_moe_experts(model))
        assert len(experts) == _NUM_EXPERTS
        for name in experts:
            assert name.endswith(tuple(f".experts.{i}" for i in range(8)))
            assert "experts." in name

    def test_routers_and_static_parts_are_not_experts(self):
        model = FakeModel()
        names = list(iter_moe_experts(model))
        assert not any("gate" in name for name, _ in names)
        assert not any("embed" in name for name, _ in names)

    def test_dense_model_has_no_experts(self):
        assert dict(iter_moe_experts(nn.Linear(4, 4))) == {}


# -- prepare / release ------------------------------------------------------------

class TestPrepare:
    def test_prepare_returns_self_and_finds_experts(self):
        model = FakeModel(_COLLAPSE_BIAS)
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4)
        assert streamer.prepare() is streamer
        assert len(streamer.experts) == _NUM_EXPERTS
        assert streamer._prepared is True

    def test_prepare_is_idempotent(self):
        model = FakeModel(_COLLAPSE_BIAS)
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4)
        streamer.prepare()
        first_log_len = len(streamer.placement_log)
        streamer.prepare()
        assert len(streamer.placement_log) == first_log_len

    def test_static_placement_and_byte_accounting(self):
        model = FakeModel(_COLLAPSE_BIAS)
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4).prepare()

        static = [entry for entry in streamer.placement_log if entry[0] == "static"]
        # 16 expert offloads + non-expert leaves (embed + 2 router gates)
        assert len(static) == _NUM_EXPERTS + 3

        # Byte accounting: expert fc = 4x4 weight + 4 bias = 20 fp32 params (80 B);
        # embed is also 4x4 (80 B); each router gate is 4->8 (40 params, 160 B).
        assert sum(entry[2] for entry in static) == 16 * 80 + 80 + 2 * 160

    def test_experts_live_on_cpu_after_prepare(self):
        model = FakeModel(_COLLAPSE_BIAS)
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4).prepare()
        for exp in streamer.experts.values():
            assert all(p.device.type == "cpu" for p in exp.parameters())

    def test_pin_memory_degrades_gracefully(self):
        model = FakeModel(_COLLAPSE_BIAS)
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4, pin_cpu_memory=True)
        streamer.prepare()  # must not raise even if pinning fails
        assert len(streamer.experts) == _NUM_EXPERTS

    def test_release_removes_all_hooks(self):
        model = FakeModel(_COLLAPSE_BIAS)
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4).prepare()
        assert len(streamer.hooks) == 2 * _NUM_EXPERTS
        streamer.release()
        assert streamer.hooks == []
        assert streamer._prepared is False

    def test_no_experts_is_a_noop(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=2)
        streamer.prepare()
        assert streamer.experts == {}
        assert streamer._prepared is True
        assert streamer.stats["h2d_streams"] == 0


# -- hook-driven streaming ---------------------------------------------------------

class TestStreaming:
    def _biased_model(self):
        return FakeModel(_COLLAPSE_BIAS)

    def test_first_forward_streams_each_hot_expert_once(self):
        model = self._biased_model()
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4).prepare()

        _ = model(torch.randn(2, 4))

        # 4 hot experts (2 per layer), each streamed exactly once; batch=2
        # means 2 pre-hook calls per expert, but only the first streams.
        assert streamer.stats["h2d_streams"] == 4
        assert streamer.stats["d2h_evictions"] == 0
        assert streamer.stats["cache_hits"] == 4  # 2nd call per expert hits (x2 layers)
        assert streamer.cache.resident() == {
            "layers.0.block_sparse_moe.experts.0",
            "layers.0.block_sparse_moe.experts.1",
            "layers.1.block_sparse_moe.experts.0",
            "layers.1.block_sparse_moe.experts.1",
        }

    def test_second_forward_is_all_cache_hits(self):
        model = self._biased_model()
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4).prepare()
        _ = model(torch.randn(2, 4))
        first_streams = streamer.stats["h2d_streams"]

        _ = model(torch.randn(2, 4))

        assert streamer.stats["h2d_streams"] == first_streams  # nothing new
        assert streamer.stats["d2h_evictions"] == 0
        assert streamer.stats["cache_hits"] == 12  # 4 (1st fwd) + 8 (2nd fwd)

    def test_small_cache_thrashes_and_evicts_lru(self):
        # capacity 2 < 4 hot experts: every layer transition must evict.
        model = self._biased_model()
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=2).prepare()

        _ = model(torch.randn(1, 4))  # 4 streams, 2 evictions
        assert streamer.stats["h2d_streams"] == 4
        assert streamer.stats["d2h_evictions"] == 2

        _ = model(torch.randn(1, 4))  # everything evicted again
        assert streamer.stats["h2d_streams"] == 8
        assert streamer.stats["d2h_evictions"] == 6
        assert streamer.stats["cache_hits"] == 0

    def test_byte_accounting_tracks_streams_and_evictions(self):
        model = self._biased_model()
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=2).prepare()

        _ = model(torch.randn(1, 4))

        expert_bytes = 20 * 4  # one expert's fp32 parameter bytes
        assert streamer.stats["bytes_h2d"] == 4 * expert_bytes
        assert streamer.stats["bytes_d2h"] == 2 * expert_bytes

    def test_release_stops_all_accounting(self):
        model = self._biased_model()
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=4).prepare()
        _ = model(torch.randn(1, 4))
        snapshot = dict(streamer.stats)

        streamer.release()
        _ = model(torch.randn(1, 4))

        assert streamer.stats == snapshot

    def test_placement_log_records_reasons(self):
        model = self._biased_model()
        streamer = MoEStreamer(model, device="cpu", max_resident_experts=2).prepare()
        _ = model(torch.randn(1, 4))

        reasons = [entry[0] for entry in streamer.placement_log]
        assert reasons.count("stream") == 4
        assert reasons.count("evict") == 2
