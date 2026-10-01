"""LFU vs LRU expert-cache benchmark (CPU only, no model needed).

Simulates MoE routing traffic with a Zipfian (long-tail) distribution and
counts cache misses — each miss equals one H2D PCIe expert stream.

The point of this benchmark is to isolate the POLICY from the CAPACITY:

  * Comparing LFU at capacity 8 against LFU at capacity 2 measures capacity,
    not policy — the stream reduction there comes from having more slots.
  * The real claim from the LFU upgrade is: at the SAME capacity, LFU streams
    fewer experts than LRU on realistic skewed routing traffic.

Run:
    python benchmarks/bench_lfu.py
"""
from __future__ import annotations

import random
import sys
from collections import OrderedDict

from tinct.engine.moe import ExpertLFUCache

NUM_EXPERTS = 256  # Mixtral 8x7B: 32 layers x 8 experts
DEFAULT_REQUESTS = 10_000
SEED = 20260912  # fixed: a benchmark that reports different numbers each run proves nothing


class LRUCacheRef:
    """Reference LRU (the pre-v1.1 eviction policy), same API as ExpertLFUCache."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._order: OrderedDict[int, None] = OrderedDict()

    def is_resident(self, key: int) -> bool:
        return key in self._order

    def resident(self) -> set[int]:
        return set(self._order)

    def touch(self, key: int) -> None:
        if key in self._order:
            self._order.move_to_end(key)

    def admit(self, key: int) -> list[int]:
        if key in self._order:
            self.touch(key)
            return []
        evicted: list[int] = []
        while len(self._order) >= self.capacity:
            victim, _ = self._order.popitem(last=False)  # least-recently-used
            evicted.append(victim)
        self._order[key] = None
        return evicted


def zipf_requests(num_requests: int, num_experts: int = NUM_EXPERTS,
                  exponent: float = 1.1, seed: int = SEED) -> list[int]:
    """Rank-ordered traffic: expert 0 is hottest, expert N-1 is coldest."""
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** exponent for i in range(num_experts)]
    return rng.choices(range(num_experts), weights=weights, k=num_requests)


def count_streams(cache, requests: list[int]) -> int:
    """Replay traffic through a cache; return H2D streams (= misses).

    Mirrors the streamer's hooks exactly: a miss admits (streams) the expert,
    and every use — including the admitting one — counts as a touch.
    """
    visits = 0
    for req in requests:
        if not cache.is_resident(req):
            visits += 1
            cache.admit(req)
        cache.touch(req)
    return visits


def benchmark(capacity: int, requests: list[int]) -> tuple[int, int]:
    """Return (lfu_streams, lru_streams) for one capacity, same traffic."""
    lfu = count_streams(ExpertLFUCache(capacity=capacity), requests)
    lru = count_streams(LRUCacheRef(capacity=capacity), requests)
    return lfu, lru


def main() -> None:
    requests = zipf_requests(DEFAULT_REQUESTS)
    hot = set(requests[: int(len(requests) * 0.2)])
    print(f"--- MoE cache benchmark: {DEFAULT_REQUESTS:,} routing decisions, "
          f"{NUM_EXPERTS} experts, Zipf exponent 1.1 (seed {SEED}) ---")
    print(f"    distinct experts touched: {len(set(requests))} | "
          f"top-20% of requests hit {len(hot)} distinct experts\n")

    print(f"{'capacity':>9} | {'LFU streams':>11} | {'LRU streams':>11} | {'LFU advantage':>13} | {'hit rate (LFU/LRU)':>18}")
    print("-" * 80)

    results: dict[int, tuple[int, int]] = {}
    for capacity in (2, 4, 8, 16, 32):
        lfu, lru = benchmark(capacity, requests)
        results[capacity] = (lfu, lru)
        advantage = 100 * (lru - lfu) / lru if lru else 0.0
        lfu_hit = 100 * (1 - lfu / DEFAULT_REQUESTS)
        lru_hit = 100 * (1 - lru / DEFAULT_REQUESTS)
        print(f"{capacity:>9} | {lfu:>11,} | {lru:>11,} | {advantage:>12.1f}% | "
              f"{lfu_hit:>7.1f}% /{lru_hit:>7.1f}%")

    print()
    lfu2, lru2 = results[2]
    lfu8, _ = results[8]
    print(f"Policy win at matched capacity (2 slots): "
          f"{100 * (lru2 - lfu2) / lru2:.1f}% fewer streams for LFU.")
    print(f"Capacity effect (LFU 8 vs LFU 2):          "
          f"{100 * (lfu2 - lfu8) / lfu2:.1f}% fewer streams — this is NOT a policy win.")
    print("\nNote: at capacity 2 the working set is far larger than the cache, so both")
    print("policies thrash and the difference is small. The policy advantage grows with")
    print("capacity, which is the argument for raising --offload-experts' residency.")


if __name__ == "__main__":
    sys.exit(main())
