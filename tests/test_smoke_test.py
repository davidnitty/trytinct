"""Tests for the generation smoke test gate predicates (no ML deps)."""

from tinct.evals.smoke_test import classify_response


def test_import_does_not_require_heavy_deps():
    import tinct.evals.smoke_test  # noqa: F401


def test_empty_response_flagged():
    assert classify_response("   ") == (True, False)
    assert classify_response("") == (True, False)


def test_normal_response_passes():
    assert classify_response("The capital of France is Paris.") == (False, False)


def test_repetitive_long_response_flagged():
    looping = "yes " * 30  # 30 identical words
    assert classify_response(looping) == (False, True)


def test_short_repetition_not_flagged():
    # Fewer than the 15-word minimum: not counted as a loop.
    assert classify_response("a a a a a a a") == (False, False)


def test_varied_long_response_not_repetitive():
    varied = "the quick brown fox jumps over the lazy dog near the river bank"
    assert classify_response(varied) == (False, False)
