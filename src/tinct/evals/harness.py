"""Eval harness — computes the gate metric by running the trained adapter on a
hold-out validation set.

Heavy dependencies are imported lazily here (transformers/torch), triggering
the ``[train]`` extra guard. For a full production workflow the harness would
also integrate ``lm-evaluation-harness``; that is surfaced as a config option
but not required to pass the gate.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from tinct.core.config import EvalConfig, TrainConfig
from tinct.engine.deps import ensure_train_deps
from tinct.evals.base import EvalResult

# Prompt template reused from the Llama trainer, minus the answer so we can
# measure loss over the target output only.
from tinct.engine.hf_trainer import _to_text


def _eval_loss_from_history(history: List[Dict[str, Any]]) -> Optional[float]:
    """Pull the final eval_loss from a log_history list."""
    for entry in reversed(history or []):
        if isinstance(entry, dict) and "eval_loss" in entry and entry["eval_loss"] is not None:
            return float(entry["eval_loss"])
    return None


class LlamaEvalHarness:
    """Runs a trained Llama adapter over validation records and returns metrics."""

    family = "llama"

    def run(self, adapter_dir: Path, model_id: str, valid_records: List[Dict[str, Any]],
            train_cfg: TrainConfig, eval_cfg: EvalConfig) -> EvalResult:
        ensure_train_deps(quant=train_cfg.quant)

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for rec in valid_records:
                text = _to_text(rec, train_cfg, True)
                enc = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=train_cfg.max_seq_len)
                input_ids = enc.input_ids.to(model.device)
                labels = input_ids.clone()
                loss = model(input_ids=input_ids, labels=labels).loss
                # Number of tokens in the output region is approximated by the
                # full sequence for the simple hold-out loss metric.
                n = input_ids.numel()
                total_loss += float(loss.item()) * n
                total_tokens += n

        mean_loss = total_loss / total_tokens if total_tokens else float("inf")
        value: float = mean_loss
        if eval_cfg.metric == "perplexity":
            value = math.exp(mean_loss) if math.isfinite(mean_loss) else float("inf")

        return EvalResult(
            metric=eval_cfg.metric,
            value=value,
            threshold=eval_cfg.threshold,
            higher_is_better=eval_cfg.higher_is_better,
        )
