"""
MoE expert offloading engine.

Keeps router, attention, norms, embed, and lm_head on GPU.
Keeps expert MLPs on CPU and streams them to GPU on demand
with an LRU residency cache.

Makes a ~93GB fp16 Mixtral 8x7B fit in a 24GB GPU for certification.

Design principles:

- **No VRAM spike at load.** The full model is never ``.to(device)`` —
  non-expert leaves move to the target device individually; experts stay on
  CPU from the start, so nothing transiently occupies GPU memory.
- **LRU residency, not per-forward round-trips.** Streaming an expert to the
  device and evicting it after every forward would be PCIe-bound. Up to
  ``max_resident_experts`` stay hot; eviction happens only under pressure.
- **Synchronous eviction.** D2H eviction runs after the expert's forward
  completed (sequential generation) and is synchronous — no ``non_blocking``
  — so a pending kernel can never read a half-transferred weight.
- **CPU-testable.** All bookkeeping lives in the pure-Python
  :class:`ExpertLRUCache`; hooks are verified via ``placement_log`` rather
  than real device inspection.
"""

import logging
import re
from typing import Iterator

log = logging.getLogger(__name__)

# Mixtral naming. Phase 2 extends this (Qwen-MoE/DeepSeek use mlp.gate etc.)
_EXPERT_RE = re.compile(r"block_sparse_moe\.experts\.\d+$")


def iter_moe_experts(model) -> Iterator[tuple[str, "torch.nn.Module"]]:
    """Yields (path, module) for every expert MLP in the model."""
    for name, module in model.named_modules():
        if _EXPERT_RE.search(name):
            yield name, module


class ExpertLRUCache:
    """Pure-Python LRU bookkeeping for expert residency. Unit-testable."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._order: list[str] = []  # most-recently-used at the end

    def is_resident(self, key: str) -> bool:
        return key in self._order

    def resident(self) -> set[str]:
        return set(self._order)

    def touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def admit(self, key: str) -> list[str]:
        """Make key resident. Returns keys evicted to make room."""
        if key in self._order:
            self.touch(key)
            return []
        evicted = []
        while len(self._order) >= self.capacity:
            evicted.append(self._order.pop(0))  # evict least-recently-used
        self._order.append(key)
        return evicted


class MoEStreamer:
    """
    Offloads expert MLPs to CPU, streams them to GPU on demand.

    Usage:
        streamer = MoEStreamer(model, device="cuda", max_resident_experts=2)
        streamer.prepare()
        ... run generation ...
        streamer.release()
    """

    def __init__(
        self,
        model,
        device: str = "cuda",
        max_resident_experts: int = 2,
        pin_cpu_memory: bool = False,
    ):
        self.model = model
        self.device = device
        self.cache = ExpertLRUCache(max_resident_experts)
        self.pin = pin_cpu_memory
        self.experts: dict[str, object] = {}
        self.hooks: list = []
        self.stats = {
            "h2d_streams": 0,
            "d2h_evictions": 0,
            "cache_hits": 0,
            "bytes_h2d": 0,
            "bytes_d2h": 0,
        }
        self.placement_log: list[tuple[str, str, int]] = []
        self._prepared = False

    # ------------------------------------------------------------------ setup

    def prepare(self) -> "MoEStreamer":
        if self._prepared:
            return self

        self.experts = dict(iter_moe_experts(self.model))
        if not self.experts:
            log.warning("[tinct] MoEStreamer.prepare(): no MoE experts found; no-op.")
            self._prepared = True
            return self

        # Module ids to skip when moving static parts to GPU
        skip = set()
        for exp in self.experts.values():
            for m in exp.modules():
                skip.add(id(m))

        # Move non-expert LEAF modules to GPU (no full-model .to() spike)
        for m in self.model.modules():
            if id(m) not in skip and not list(m.children()):
                self._move(m, self.device, "static")

        # Experts: to CPU, optionally pin, then hook
        for name, exp in self.experts.items():
            self._move(exp, "cpu", "static")
            if self.pin:
                for p in exp.parameters():
                    try:
                        p.data = p.data.pin_memory()
                    except Exception:
                        pass  # pinned-memory limits; degrade gracefully

            self.hooks.append(
                exp.register_forward_pre_hook(self._make_pre_hook(name))
            )
            self.hooks.append(
                exp.register_forward_hook(self._make_post_hook(name))
            )

        self._prepared = True
        log.info(
            "[tinct] MoEStreamer ready: %d experts offloaded, %d resident slots.",
            len(self.experts), self.cache.capacity,
        )
        return self

    def release(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self._prepared = False

    # ------------------------------------------------------------------ hooks

    def _make_pre_hook(self, name: str):
        def hook(module, inputs):
            if self.cache.is_resident(name):
                self.stats["cache_hits"] += 1
                return
            for evicted in self.cache.admit(name):
                self._move(self.experts[evicted], "cpu", "evict")  # synchronous
            self._move(module, self.device, "stream")
        return hook

    def _make_post_hook(self, name: str):
        def hook(module, inputs, output):
            self.cache.touch(name)
        return hook

    # ------------------------------------------------------------- placement

    def _move(self, module, device: str, reason: str) -> None:
        nbytes = sum(p.numel() * p.element_size() for p in module.parameters())
        module.to(device)
        self.placement_log.append((reason, device, nbytes))
        if reason == "stream":
            self.stats["h2d_streams"] += 1
            self.stats["bytes_h2d"] += nbytes
        elif reason == "evict":
            self.stats["d2h_evictions"] += 1
            self.stats["bytes_d2h"] += nbytes
