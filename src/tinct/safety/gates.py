"""Safety gate orchestration for ``tinct eval --safety``.

Loads the base model and the trained adapter (lazily, same pattern as
:mod:`tinct.evals.smoke_test`), then runs the two behavioral certification
gates: canary leakage and refusal regression. The detection functions in
:mod:`tinct.safety.canaries` / :mod:`tinct.safety.refusal` are the tested,
dependency-free core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from tinct.engine.deps import ensure_train_deps
from tinct.safety.canaries import check_canary_leakage
from tinct.safety.refusal import SAFETY_PROMPTS, check_refusal_regression

# 50 tokens is enough to reproduce a canary phrase or a refusal, and keeps
# CPU-only safety evals fast.
_MAX_NEW_TOKENS = 60


def _make_generator(model_name: str, adapter_dir: Optional[Path]) -> Callable[[str], str]:
    """Return a ``prompt -> response`` callable backed by the (adapted) model."""
    ensure_train_deps()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir)) if adapter_dir else base
    model.eval()

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # tokenizer has no chat template -> raw prompt
            text = prompt
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

    return generate


def run_safety_gates(model_name: str, adapter_dir: Path, canaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Run both behavioral gates and return the ``safety_gates`` evidence dict."""
    adapter_generate = _make_generator(model_name, adapter_dir)
    base_generate = _make_generator(model_name, None)

    canary_leakage = check_canary_leakage(adapter_generate, canaries)

    base_responses = [base_generate(prompt) for prompt in SAFETY_PROMPTS]
    adapter_responses = [adapter_generate(prompt) for prompt in SAFETY_PROMPTS]
    refusal_regression = check_refusal_regression(base_responses, adapter_responses)

    return {
        "canary_leakage": canary_leakage,
        "refusal_regression": refusal_regression,
    }
