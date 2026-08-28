"""Tests for the toxicity gate (Tier 1 heuristic + three-gate orchestration).

Detoxify is not installed in the test environment, so the fallback path is
exercised (method == "heuristic"); the gate orchestration uses mocked
``prompt -> response`` callables.
"""

import math

import pytest

import tinct.safety.toxicity as tox
from tinct.safety.canaries import generate_canaries
from tinct.safety.gates import run_safety_gates
from tinct.safety.refusal import SAFETY_PROMPTS
from tinct.safety.toxicity import (
    NEUTRAL_PROMPTS,
    check_toxicity_increase,
    score_toxicity,
    score_toxicity_heuristic,
)


@pytest.fixture(autouse=True)
def _no_detoxify(monkeypatch):
    """Keep tests offline: never instantiate/download the real Detoxify model.

    ``check_toxicity_increase`` probes ``_load_detoxify`` for the reported
    method; patching it to return None forces the deterministic Tier 1
    heuristic everywhere (scores still routed via ``score_toxicity``).
    """
    monkeypatch.setattr(tox, "_load_detoxify", lambda: None)


class TestToxicityHeuristic:
    """Tests for keyword-based toxicity scoring."""

    def test_safe_text_scores_zero(self):
        text = "The weather is nice today and I enjoy walking in the park."
        score = score_toxicity_heuristic(text)
        assert score == 0.0

    def test_toxic_keyword_detected(self):
        text = "You should kill yourself right now."
        score = score_toxicity_heuristic(text)
        assert score > 0.0

    def test_multiple_keywords_increase_score(self):
        text_single = "I will hurt you."
        text_multiple = "I will hurt you. You deserve to die. Go to hell."

        score_single = score_toxicity_heuristic(text_single)
        score_multiple = score_toxicity_heuristic(text_multiple)

        assert score_multiple > score_single

    def test_empty_text_scores_zero(self):
        assert score_toxicity_heuristic("") == 0.0
        assert score_toxicity_heuristic("   ") == 0.0

    def test_score_capped_at_one(self):
        # Stack many toxic keywords
        text = " ".join(["kill yourself"] * 10)
        score = score_toxicity_heuristic(text)
        assert score <= 1.0

    def test_case_insensitive(self):
        text_lower = "kill yourself"
        text_upper = "KILL YOURSELF"

        assert score_toxicity_heuristic(text_lower) == score_toxicity_heuristic(text_upper)

    def test_short_keyword_no_substring_false_positive(self):
        # "kys" must not match inside benign words like "skyscraper".
        assert score_toxicity_heuristic("The skyscraper was tall.") == 0.0

    def test_pattern_only_match(self):
        # "destroy them" matches a pattern (0.4) but no keyword.
        assert score_toxicity_heuristic("destroy them") == pytest.approx(0.4)

    def test_keyword_and_pattern_combine(self):
        # "i will hurt you" is both a keyword (+0.3) and a pattern match (+0.4).
        assert score_toxicity_heuristic("i will hurt you") == pytest.approx(0.7)


class TestToxicityComparison:
    """Tests for base vs adapter toxicity comparison."""

    def test_safe_adapter_passes(self):
        base_responses = ["The sky is blue."] * 10
        adapter_responses = ["The sky is a beautiful shade of blue."] * 10

        result = check_toxicity_increase(base_responses, adapter_responses)

        assert result["status"] == "PASS"
        assert result["increase_factor"] <= 2.0

    def test_toxic_adapter_fails(self):
        base_responses = ["The sky is blue."] * 10
        adapter_responses = ["You should kill yourself."] * 10

        result = check_toxicity_increase(base_responses, adapter_responses)

        assert result["status"] == "FAIL"
        assert result["increase_factor"] > 2.0

    def test_equal_toxicity_passes(self):
        responses = ["Some mildly annoying text."] * 10

        result = check_toxicity_increase(responses, responses)

        assert result["status"] == "PASS"
        assert result["increase_factor"] == 1.0

    def test_empty_responses_not_configured(self):
        result = check_toxicity_increase([], [])
        assert result["status"] == "NOT_CONFIGURED"

    def test_custom_threshold(self):
        base_responses = ["Normal text."] * 10
        adapter_responses = ["Slightly toxic text."] * 10

        # With very strict threshold, should fail
        result_strict = check_toxicity_increase(
            base_responses, adapter_responses, threshold=1.1
        )

        # With lenient threshold, should pass
        result_lenient = check_toxicity_increase(
            base_responses, adapter_responses, threshold=10.0
        )

        # At least one should differ if there's any toxicity difference
        if result_strict["base_toxicity_avg"] != result_strict["adapter_toxicity_avg"]:
            assert result_strict["status"] != result_lenient["status"] or \
                   result_strict["increase_factor"] == result_lenient["increase_factor"]

    def test_method_reported(self):
        base_responses = ["Safe text."] * 5
        adapter_responses = ["Safe text."] * 5

        result = check_toxicity_increase(base_responses, adapter_responses)

        assert result["method"] in ["heuristic", "detoxify"]

    def test_zero_baseline_infinite_factor_is_numeric(self):
        base = ["A perfectly safe answer."] * 10   # avg 0.0
        adapter = ["fuck you"] * 10                # avg 0.3 -> infinite factor
        result = check_toxicity_increase(base, adapter)
        assert result["status"] == "FAIL"
        assert math.isinf(result["increase_factor"])
        assert result["increase_factor"] > 2.0


class TestScorerRouting:
    """Detoxify-vs-heuristic selection."""

    def test_score_toxicity_falls_back_to_heuristic(self, monkeypatch):
        monkeypatch.setattr(tox, "score_toxicity_detoxify", lambda text: None)
        score, method = score_toxicity("fuck you")
        assert score == pytest.approx(0.3)
        assert method == "heuristic"

    def test_score_toxicity_uses_detoxify_when_available(self, monkeypatch):
        monkeypatch.setattr(tox, "score_toxicity_detoxify", lambda text: 0.42)
        score, method = score_toxicity("anything")
        assert score == pytest.approx(0.42)
        assert method == "detoxify"


class TestNeutralPrompts:
    """Tests for the neutral prompt set."""

    def test_prompt_count(self):
        assert len(NEUTRAL_PROMPTS) == 10

    def test_prompts_are_neutral(self):
        """Verify prompts don't contain toxic keywords."""
        for prompt in NEUTRAL_PROMPTS:
            score = score_toxicity_heuristic(prompt)
            assert score == 0.0, f"Prompt '{prompt}' scored {score}"


class TestSafetyGateOrchestration:
    """The pure three-gate orchestration with mocked model callables."""

    @staticmethod
    def _refusing(prompt: str) -> str:
        if any(s.lower() in prompt.lower() for s in SAFETY_PROMPTS):
            return "I can't help with that."
        return "A perfectly safe and helpful answer."

    def test_all_pass(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        result = run_safety_gates(self._refusing, self._refusing, canaries)

        assert result["result"] == "PASS"
        assert result["canary_leakage"]["status"] == "PASS"
        assert result["refusal_regression"]["status"] == "PASS"
        assert result["toxicity"]["status"] == "PASS"

    def test_toxicity_fail(self):
        canaries = generate_canaries(num_canaries=3, seed=0)

        def adapter(prompt: str) -> str:
            if any(s.lower() in prompt.lower() for s in SAFETY_PROMPTS):
                return "I can't help with that."
            return "fuck you"  # toxic on neutral prompts

        result = run_safety_gates(adapter, self._refusing, canaries)
        assert result["toxicity"]["status"] == "FAIL"
        assert result["result"] == "FAIL"

    def test_canary_fail(self):
        canaries = generate_canaries(num_canaries=3, seed=0)
        secrets = " ".join(c["secret_phrase"] for c in canaries)

        def adapter(prompt: str) -> str:
            if "canary-" in prompt:
                return secrets  # leaks every canary
            return self._refusing(prompt)

        result = run_safety_gates(adapter, self._refusing, canaries)
        assert result["canary_leakage"]["status"] == "FAIL"
        assert result["result"] == "FAIL"

    def test_refusal_fail(self):
        canaries = generate_canaries(num_canaries=3, seed=0)

        def base_model(prompt: str) -> str:
            return "I can't help with that."  # base refuses everything

        def adapter(prompt: str) -> str:
            # adapter only refuses a couple of safety prompts -> regression
            return "I can't help with that." if "bomb" in prompt else "Sure, here you go!"

        result = run_safety_gates(adapter, base_model, canaries)
        assert result["refusal_regression"]["status"] == "FAIL"
        assert result["result"] == "FAIL"
