"""Policy regression guard for the Phase-1 LFU upgrade.

On skewed (Zipfian) routing traffic, LFU must stream fewer experts than the
pre-v1.1 LRU policy at the SAME capacity — and LFU at half the capacity must
match LRU's hit rate, which is the "half the resident-expert VRAM" claim.

The reference LRU and the traffic generator live in
benchmarks/bench_lfu.py (loaded directly, so there is exactly one definition).
"""

import importlib.util
from pathlib import Path

import pytest

from tinct.engine.moe import ExpertLFUCache

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_lfu.py"
_spec = importlib.util.spec_from_file_location("bench_lfu", _BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)  # pure stdlib + tinct import, safe to exec

REQUESTS = bench.zipf_requests(10_000)


def lfu_streams(capacity: int) -> int:
    return bench.count_streams(ExpertLFUCache(capacity=capacity), REQUESTS)


def lru_streams(capacity: int) -> int:
    return bench.count_streams(bench.LRUCacheRef(capacity=capacity), REQUESTS)


@pytest.mark.parametrize("capacity", [2, 8])
def test_lfu_streams_no_more_than_lru_at_matched_capacity(capacity: int):
    lfu = lfu_streams(capacity)
    lru = lru_streams(capacity)
    assert lfu < lru, f"LFU ({lfu}) must beat LRU ({lru}) at capacity {capacity}"


def test_lfu_at_half_capacity_matches_lru_hit_rate():
    # The enterprise claim: LFU with half the residency is not worse than LRU.
    lfu8 = lfu_streams(8)
    lru16 = lru_streams(16)
    assert lfu8 <= lru16


def test_benchmark_traffic_is_skewed_and_seeded():
    # The premise of the whole comparison: the traffic is a long tail, and the
    # same seed reproduces it exactly.
    again = bench.zipf_requests(10_000)
    assert again == REQUESTS
    assert len(set(REQUESTS)) > 100  # a real long tail, not two hot experts
