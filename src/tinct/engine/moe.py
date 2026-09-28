"""
MoE expert offloading engine.

Keeps router, attention, norms, embed, and lm_head on GPU.
Keeps expert MLPs on CPU and streams them to GPU on demand
with an LFU (LRU tie-break) residency cache.

Makes a ~93GB fp16 Mixtral 8x7B fit in a 24GB GPU for certification.

Design principles:

- **No VRAM spike at load.** The full model is never ``.to(device)`` —
  non-expert leaves move to the target device individually; experts stay on
  CPU from the start, so nothing transiently occupies GPU memory.
- **LFU residency, not per-forward round-trips.** Streaming an expert to the
  device and evicting it after every forward would be PCIe-bound. Up to
  ``max_resident_experts`` stay hot; eviction happens only under pressure, and
  frequency (not just recency) decides the victim, so an expert that a gate
  keeps routing to is not evicted merely for being loaded early.
- **Synchronous eviction.** D2H eviction runs after the expert's forward
  completed (sequential generation) and is synchronous — no ``non_blocking``
  — so a pending kernel can never read a half-transferred weight.
- **CPU-testable.** All bookkeeping lives in the pure-Python
  :class:`ExpertLFUCache`; hooks are verified via ``placement_log`` rather
  than real device inspection.

Known limitation: the streamer is **inference-grade**. Experts stream at
forward and are evicted synchronously; autograd may then require evicted
weights at backward and fail with a device mismatch. Training MoE models uses
the standard 4-bit QLoRA path; training-grade streaming (a re-streaming
autograd function) is future work.
"""

import logging
from typing import Iterator

log = logging.getLogger(__name__)


def iter_moe_routers(model) -> Iterator[tuple[str, object]]:
    """
    Yields (path, block) for every MoE routing block, architecture-agnostic.

    Structural definition: a module exposing both a router ('gate') and a
    list of expert submodules ('experts'). Matches Mixtral
    (block_sparse_moe), Qwen-MoE / DeepSeek / Phi-MoE (mlp), and future
    families without name regexes.
    """
    for name, module in model.named_modules():
        if (
            hasattr(module, "gate")
            and hasattr(module, "experts")
            and hasattr(module.experts, "__len__")
            and hasattr(module.experts, "__getitem__")
        ):
            yield name, module


def iter_moe_experts(model) -> Iterator[tuple[str, object]]:
    """Yields (path, expert) for every routed expert MLP in the model."""
    for router_name, router in iter_moe_routers(model):
        for i, expert in enumerate(router.experts):
            yield f"{router_name}.experts.{i}", expert


class ExpertLFUCache:
    """
    LFU cache with LRU tie-breaking.

    Keeps frequently used experts pinned to VRAM, preventing PCIe thrashing
    when a hot expert is repeatedly routed to during safety gates. If two
    experts share a hit count, the least-recently-used one is evicted.

    Two consequences of the policy, both intentional:
    - An evicted expert's count is dropped, so a re-admitted expert restarts
      at 1 (standard LFU; hot experts that get evicted lose their history).
    - A just-admitted expert also starts at 1, so under sustained pressure the
      newest entrant is the next eviction candidate.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._resident: set[str] = set()
        self.counts: dict[str, int] = {}
        self.history: list[str] = []  # LRU order (oldest at index 0)

    def is_resident(self, key: str) -> bool:
        return key in self._resident

    def resident(self) -> set[str]:
        return set(self._resident)

    def touch(self, key: str) -> None:
        """Record one use of a resident expert (count + recency)."""
        if key in self._resident:
            self.counts[key] += 1
            # Move to most-recently-used end of history
            self.history.remove(key)
            self.history.append(key)

    def admit(self, key: str) -> list[str]:
        """Make key resident. Returns keys evicted to make room."""
        if key in self._resident:
            self.touch(key)
            return []

        evicted: list[str] = []
        while len(self._resident) >= self.capacity:
            # Victim: lowest hit count, ties broken by oldest in history (LRU).
            victim = min(
                self._resident,
                key=lambda k: (self.counts.get(k, 0), self.history.index(k)),
            )
            self._resident.remove(victim)
            self.history.remove(victim)
            del self.counts[victim]
            evicted.append(victim)

        self._resident.add(key)
        self.counts[key] = 1
        self.history.append(key)
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
        self.cache = ExpertLFUCache(max_resident_experts)
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
        # Time-series telemetry: one snapshot per recorded generation step,
        # for live charts and post-hoc thrash analysis.
        self.telemetry: list[dict] = []
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

    # -------------------------------------------------------------- telemetry

    def record_step(self) -> dict:
        """Log one hardware snapshot (call after each prompt/generation).

        Counters are cumulative, matching :attr:`stats`; diff consecutive
        entries to chart per-step rates. Appends unbounded — gate runs record
        tens of steps, but a very long eval should be chunked or the list
        persisted separately rather than embedded in a signed bundle.
        """
        import torch

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024**2

        snapshot = {
            "step": len(self.telemetry),
            "vram_mb": round(vram_mb, 1),
            "resident_experts": len(self.cache.resident()),
            "h2d_streams": self.stats["h2d_streams"],
            "cache_hits": self.stats["cache_hits"],
        }
        self.telemetry.append(snapshot)
        return snapshot

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
