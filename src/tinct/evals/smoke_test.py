"""Generation smoke test — proves a trained adapter actually works.

Loads the base model, attaches the tinct-trained LoRA adapter, and runs a few
basic prompts. It fails the gate if the model outputs **empty** text or clearly
**repetitive** (looping) text.

The gate predicates are pure functions (:func:`classify_response`) so they are
unit-testable without torch/transformers; the model loading and generation
happen lazily inside :func:`run_generation_smoke_test`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from tinct.engine.deps import ensure_train_deps
from tinct.utils.logging import get_logger

log = get_logger("tinct.evals.smoke_test")

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Write a short poem about the ocean.",
    "Explain quantum computing in one sentence.",
]

# A response is "repetitive" if it is long enough and its first 5 words have
# fewer than 3 unique tokens (a strong signature of a generation loop).
_MIN_WORDS = 15
_UNIQUE_FIRST_WORDS = 3


def classify_response(response: str) -> tuple[bool, bool]:
    """Pure gate predicate over one generated response.

    Returns ``(is_empty, is_repetitive)``.
    """
    is_empty = len(response.strip()) == 0
    words = response.split()
    is_repetitive = len(words) > _MIN_WORDS and len(set(words[:5])) < _UNIQUE_FIRST_WORDS
    return is_empty, is_repetitive


def run_generation_smoke_test(
    model_name: str,
    adapter_path: Path,
    eval_report_path: Path,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 50,
    offload_experts: bool = False,
) -> bool:
    """Load base model + LoRA adapter and gate generation quality.

    Writes an ``eval_report.json`` at ``eval_report_path`` and returns True if
    every prompt produces non-empty, non-repetitive output. With
    ``offload_experts`` (MoE models), the model loads on CPU and experts
    stream to GPU on demand; offload stats land in the eval report.
    """
    ensure_train_deps()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("[tinct eval] Loading base model and attaching adapter...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CPU-safe dtype: float32 when no CUDA (float16 fails on CPU).
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    streamer = None
    if offload_experts:
        # MoE path: CPU load (no full-model VRAM spike) + expert streaming.
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch_dtype, device_map=None,
        )
        from tinct.engine.moe import MoEStreamer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        streamer = MoEStreamer(base_model, device=device, max_resident_experts=2).prepare()
        base_model._tinct_streamer = streamer  # keep alive; .stats feeds evidence
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch_dtype, device_map="auto",
        )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    prompts = prompts or DEFAULT_PROMPTS
    results = []
    empty_count = 0
    repetition_count = 0

    log.info("[tinct eval] Running generation smoke test...")
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # tokenizer has no chat template -> use raw prompt
            text = prompt
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        is_empty, is_repetitive = classify_response(response)
        if is_empty:
            empty_count += 1
        if is_repetitive:
            repetition_count += 1
        results.append({
            "prompt": prompt,
            "response_preview": response[:100],
            "empty": is_empty,
            "repetitive": is_repetitive,
        })
        log.debug("[tinct eval] %s -> empty=%s repetitive=%s",
                  prompt[:40], is_empty, is_repetitive)

    report = {
        "gate": "generation_smoke_test",
        "status": "PASS" if empty_count == 0 and repetition_count == 0 else "FAIL",
        "empty_responses": empty_count,
        "repetitive_responses": repetition_count,
        "details": results,
    }
    if streamer is not None:
        # Offload bookkeeping for the evidence bundle.
        report["offload_stats"] = dict(streamer.stats)
    eval_report_path.parent.mkdir(parents=True, exist_ok=True)
    eval_report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log.info("[tinct eval] Smoke test: %s", report["status"])
    return report["status"] == "PASS"
