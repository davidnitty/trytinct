"""Llama LoRA/QLoRA trainer built on Hugging Face Transformers + TRL + PEFT.

All heavy imports happen inside :meth:`HfLlamaTrainer.train`, so importing this
module (or ``tinct``) never pulls in torch/transformers. Running ``tinct train``
triggers :func:`tinct.engine.deps.ensure_train_deps`, which raises a clear error
if the ``[train]`` extra is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tinct.engine.base import TrainingEngine, TrainingRun
from tinct.engine.deps import ensure_train_deps
from tinct.utils.logging import get_logger

log = get_logger("tinct.engine.hf_trainer")

# Default instruct-style prompt template (matches the Data Doctor's schema).
SYSTEM_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
USER_TEMPLATE = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}{input}<|eot_id|>"
ASSISTANT_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"


def _to_text(rec: Dict[str, Any], cfg: Any, add_output: bool) -> str:
    """Format one record into Llama-3 chat template text."""
    system = str(rec.get(cfg.system_column) or "") if cfg.system_column else ""
    instruction = str(rec.get(cfg.instruction_column, ""))
    input_ctx = str(rec.get(cfg.input_column) or "") if cfg.input_column else ""
    output = str(rec.get(cfg.output_column, "")) if add_output else ""

    parts = []
    if system:
        parts.append(SYSTEM_TEMPLATE.format(system=system))
    user = USER_TEMPLATE.format(instruction=instruction, input=input_ctx)
    parts.append(user)
    if add_output:
        parts.append(ASSISTANT_TEMPLATE.format(output=output))
    else:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


class HfLlamaTrainer(TrainingEngine):
    """LoRA / QLoRA trainer for the Llama model family."""

    family = "llama"

    def train(
        self,
        run_name: str,
        run_dir: Path,
        model: str,
        train_records: List[Dict[str, Any]],
        valid_records: List[Dict[str, Any]],
        config: Any,  # TrainConfig
    ) -> TrainingRun:
        ensure_train_deps(quant=config.quant)

        # Lazy heavy imports (all guarded by ensure_train_deps).
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        transformers.set_seed(config.seed)

        # --- tokenizer & base model ---------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if config.quant == "qlora":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        # --- LoRA ----                                                       
        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        trainable, total = model.get_nb_trainable_parameters()
        log.info("Trainable LoRA parameters: %s / %s (%.3f%%)",
                 trainable, total, 100.0 * trainable / total)

        # --- datasets ------------------------------------------------------
        train_ds = Dataset.from_list([{"text": _to_text(r, config, True)} for r in train_records])
        valid_ds = Dataset.from_list([{"text": _to_text(r, config, True)} for r in valid_records])

        # --- SFT config ----------------------------------------------------
        train_cfg = SFTConfig(
            output_dir=str(run_dir),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_epochs,
            max_steps=config.max_steps,
            max_seq_length=config.max_seq_len,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            seed=config.seed,
            bf16=config.use_bf16,
            fp16=False,
            gradient_checkpointing=True,
            logging_dir=str(run_dir / "logs"),
            report_to=["tensorboard"],
            eval_strategy="steps",
            eval_steps=max(1, config.save_steps),
            save_total_limit=1,
            push_to_hub=False,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            args=train_cfg,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(run_dir / "adapter"))
        tokenizer.save_pretrained(str(run_dir / "adapter"))

        metrics = trainer.state.log_history or [{}]
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return TrainingRun(
            name=run_name,
            run_dir=run_dir,
            adapter_dir=run_dir / "adapter",
            metrics_path=metrics_path,
            trainer_state_path=run_dir / "trainer_state.json",
            family=self.family,
            model=model.config.get("_name_or_path", "") if hasattr(model, "config") else "",
            metrics={"n_train": len(train_records), "n_valid": len(valid_records)},
        )
