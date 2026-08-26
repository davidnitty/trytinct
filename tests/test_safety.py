"""Tests for the behavioral certification gates (all CPU-testable).

The detection cores (canary leakage + refusal regression) are pure, so we mock
the model callable rather than loading any weights.
"""

import json
from pathlib import Path

import pytest

from tinct.safety.canaries import (
    canary_text,
    check_canary_leakage,
    generate_canaries,
    inject_canaries,
    save_canaries,
)
from tinct.safety.refusal import (
    SAFETY_PROMPTS,
    check_refusal_regression,
    count_refusals,
    is_refusal,
)


# -- canary injection --------------------------------------------------------

def test_generate_canaries_count_and_shape():
    canaries = generate_canaries(num_canaries=5, seed=42)
    assert len(canaries) == 5
    for c in canaries:
        assert "canary_id" in c
        assert "secret_phrase" in c
        assert "injected_at" in c
        assert c["secret_phrase"].strip()


def test_generate_canaries_deterministic_with_seed():
    a = generate_canaries(num_canaries=3, seed=7)
    b = generate_canaries(num_canaries=3, seed=7)
    assert [c["secret_phrase"] for c in a] == [c["secret_phrase"] for c in b]


def test_inject_canaries_appends_to_dataset(tmp_path: Path):
    data = tmp_path / "good.jsonl"
    rows = [{"text": "<|begin_of_text|>user: q<|eot_id|>"} for _ in range(3)]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    examples, canaries = inject_canaries(data, num_canaries=5, seed=1)
    assert len(canaries) == 5
    assert len(examples) == 8  # 3 original + 5 canaries
    # every canary example is tracked and contains its secret phrase.
    for c in canaries:
        assert any("canary-" + c["canary_id"] in ex.get("text", "") for ex in examples)


def test_save_canaries_writes_json(tmp_path: Path):
    canaries = generate_canaries(num_canaries=2, seed=0)
    path = save_canaries(tmp_path, canaries)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == canaries


# -- canary leakage ----------------------------------------------------------

def _leaky_model(canaries):
    def model(prompt: str) -> str:
        # Always echo every secret phrase regardless of the prompt.
        return " ".join(c["secret_phrase"] for c in canaries)
    return model


def _safe_model():
    return lambda prompt: "I'm just a normal answer that doesn't leak anything."


def test_canary_leakage_detection():
    canaries = generate_canaries(num_canaries=4, seed=0)
    result = check_canary_leakage(_leaky_model(canaries), canaries)
    assert result["status"] == "FAIL"
    assert result["canaries_tested"] == 4
    assert result["canaries_leaked"] == 4
    assert result["leakage_rate"] == 1.0


def test_canary_no_leakage():
    canaries = generate_canaries(num_canaries=4, seed=0)
    result = check_canary_leakage(_safe_model(), canaries)
    assert result["status"] == "PASS"
    assert result["canaries_leaked"] == 0


def test_canary_threshold_respected():
    # Two of four canaries leak -> rate 0.5 -> not > 0.5 -> PASS (boundary).
    canaries = generate_canaries(num_canaries=4, seed=0)
    leaked = {canaries[0]["secret_phrase"], canaries[1]["secret_phrase"]}

    def model(prompt: str) -> str:
        return " ".join(leaked)  # only two secrets reproduced

    result = check_canary_leakage(model, canaries, threshold=0.5)
    assert result["leakage_rate"] == 0.5
    assert result["status"] == "PASS"


def test_canary_text_embeds_secret():
    canary = {"canary_id": "abc123", "secret_phrase": "purple elephant dancing"}
    text = canary_text(canary)
    assert "canary-abc123" in text
    assert "purple elephant dancing" in text


# -- refusal regression ------------------------------------------------------

def test_is_refusal_matches_known_phrases():
    assert is_refusal("I can't help with that.")
    assert is_refusal("I'm not able to assist with that request.")
    assert is_refusal("As an AI, I won't help with this.")
    assert not is_refusal("Sure, here's how to do it.")
    assert not is_refusal("The capital of France is Paris.")


def test_count_refusals():
    responses = ["I can't help", "Sure!", "I cannot assist", "Here you go"]
    assert count_refusals(responses) == 2


def test_refusal_regression_detection():
    base_responses = ["I can't help"] * 20          # 100% refusal
    adapter_responses = ["I can't help"] * 12 + ["Sure!"] * 8  # 60% refusal
    result = check_refusal_regression(base_responses, adapter_responses, threshold=0.2)
    assert result["status"] == "FAIL"
    assert result["regression_rate"] == 0.4


def test_no_refusal_regression():
    base_responses = ["I can't help"] * 20
    adapter_responses = ["I can't help"] * 18 + ["Sorry"] * 2  # 90% refusal
    result = check_refusal_regression(base_responses, adapter_responses, threshold=0.2)
    assert result["status"] == "PASS"
    assert result["regression_rate"] == pytest.approx(0.1)


def test_refusal_regression_zero_base():
    # No base refusals -> regression is defined as 0 (no false failure).
    result = check_refusal_regression(["Sure"] * 5, ["Sure"] * 5, threshold=0.2)
    assert result["status"] == "PASS"
    assert result["regression_rate"] == 0.0


def test_safety_prompts_nonempty():
    assert len(SAFETY_PROMPTS) >= 20
    assert all(p.strip() for p in SAFETY_PROMPTS)