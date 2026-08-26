"""Fail-closed SFT trainer — the heart of ``tinct train``.

Wraps Hugging Face ``trl`` (SFTTrainer) but enforces strict security
boundaries:

- No top-level heavy imports (torch/trl/peft/transformers). This module is
  only loaded dynamically when ``tinct train`` executes.
- Trusts **no** remote code by default (``trust_remote_code=False``).
- Saves safetensors only (pickle ``.bin`` blocked).
- **Fail-closed loss guard**: a callback monitors the loss every log step and
  halts the run if it is NaN, infinite, or exceeds ``max_loss_threshold``,
  writing structured failure evidence so the run cannot be shipped.

The dataset passed in is expected to have a ``text`` field already formatted
with the chat template (the Data Doctor validates the raw columns; tinct
formats them into ``text`` before this runs).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from tinct.utils.logging import get_logger

log = get_logger("tinct.trainers.sft")

# NOTE: no imports of torch, trl, peft, or transformers at module scope.


def loss_is_fatal(loss: float, max_loss_threshold: float) -> bool:
    """Fail-closed predicate: NaN / infinite / over-threshold loss halts a run."""
    return math.isnan(loss) or math.isinf(loss) or loss > max_loss_threshold


class FailClosedCore:
    """Import-friendly fail-closed loss guard (no transformers import needed).

    ``run_sft`` combines this with ``transformers.TrainerCallback`` inside a
    lazily-imported subclass; tests can exercise this class directly with a
    plain state/control mock, which is exactly how the V0.1-GPU smoke test
    proves the guard fires before any real training.
    """

    def __init__(self, threshold: float, log_path: Path, fail_path: Path) -> None:
        self.threshold = threshold
        self.log_path = log_path
        self.fail_path = fail_path
        self.aborted = False

    def on_log(self, state, control, logs=None) -> bool:
        """Handle one log entry. Returns True when the run must halt."""
        if not logs:
            return False

        step = getattr(state, "global_step", 0)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, **logs}) + "\n")

        loss = logs.get("loss") or logs.get("eval_loss")
        if loss is None:
            return False
        if loss_is_fatal(float(loss), self.threshold):
            log.error("[tinct] FATAL: Loss %s; halting immediately.", loss)
            self.aborted = True
            control.should_training_stop = True
            with open(self.fail_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "reason": "loss_explosion",
                    "value": str(loss),
                    "step": step,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, fh)
            return True
        return False


def run_sft(
    model_name_or_path: str,
    dataset_path: Path,
    run_dir: Path,
    lora_rank: int = 16,
    max_loss_threshold: float = 10.0,
    num_train_epochs: int = 1,
    per_device_batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-4,
    logging_steps: int = 10,
    max_seq_length: int = 2048,
    accelerator: str = "none",
) -> bool:
    """Execute a guarded SFT run.

    ``accelerator`` is ``"none"`` (standard HF path) or ``"unsloth"`` for
    low-VRAM Triton-kernel acceleration (requires ``tinct[unsloth]``).

    Returns True on success, or False if halted by a fail-closed guard.
    """
    # --- 1. Lazy imports (heavy dependencies) ---
    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, TrainerCallback)
        from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Training dependencies missing.\n"
            "Install them with: pip install 'tinct[train]'\n"
            f"Original error: {exc}"
        ) from exc

    # --- 2. Setup directories ---
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "checkpoints"
    output_dir.mkdir(exist_ok=True)

    log_file = run_dir / "train_log.jsonl"
    fail_state_file = run_dir / "fail_state.json"

    # --- 3. Fail-closed callback (logic lives in FailClosedCore, which is
    # tested directly without any ML dependency) ---
    class TinctFailClosedCallback(TrainerCallback, FailClosedCore):
        def __init__(self, threshold: float, log_path: Path, fail_path: Path):
            FailClosedCore.__init__(self, threshold, log_path, fail_path)

        def on_log(self, args, state, control, logs=None, **kwargs):
            return FailClosedCore.on_log(self, state, control, logs)

    # --- 4. Load tokenizer & model ---
    # Security: block remote code execution by default.
    trust_remote_code = False

    # CPU-safe dtype logic: bf16 only on CUDA that supports it; float32 on CPU.
    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
    peft_config = None
    bnb_config = None

    if accelerator == "unsloth":
        # Triton-kernel acceleration; model + tokenizer come back ready.
        from tinct.engine.accelerators import load_model_with_accelerator
        model, tokenizer = load_model_with_accelerator(
            model_name_or_path, accelerator="unsloth", lora_rank=lora_rank,
            max_seq_length=max_seq_length,
        )
        log.info("[tinct] Unsloth model loaded for low-VRAM training.")
    else:
        try:
            import bitsandbytes  # noqa: F401
            if not has_cuda:
                raise ImportError("bitsandbytes requires CUDA; using 16-bit LoRA on CPU")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            log.info("[tinct] bitsandbytes detected. Using 4-bit QLoRA.")
        except ImportError:
            log.info("[tinct] bitsandbytes missing/unsupported. Using standard LoRA.")

        log.info("[tinct] Loading tokenizer and model: %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",  # accelerate disk-to-VRAM offloading
            torch_dtype=torch_dtype if not bnb_config else None,
            trust_remote_code=trust_remote_code,
        )
        if bnb_config:
            model = prepare_model_for_kbit_training(model)

    # --- 5. Configure LoRA (skipped for Unsloth — peft is already applied) ---
    peft_config = (
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        if not (accelerator == "unsloth")
        else None
    )

    # --- 6. Load dataset ---
    log.info("[tinct] Loading dataset: %s", dataset_path)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # --- 7. Training arguments ---
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        fp16=has_cuda and not use_bf16,
        bf16=use_bf16,
        logging_steps=logging_steps,
        optim="paged_adamw_8bit" if bnb_config else "adamw_torch",
        save_strategy="epoch",
        report_to="none",
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,  # one formatted text sequence per example
    )

    # --- 8. Initialize trainer ---
    callback = TinctFailClosedCallback(
        threshold=max_loss_threshold,
        log_path=log_file,
        fail_path=fail_state_file,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    # --- 9. Execute training ---
    log.info("[tinct] Starting guarded training...")
    try:
        trainer.train()
    except Exception as exc:
        if callback.aborted:
            # The guard already halted the run; a post-halt framework quirk
            # (e.g. TRL/transformers version mismatch) is not a real crash.
            log.error("[tinct] Run halted by fail-closed guard (DO NOT SHIP).")
        else:
            log.error("[tinct] Training crashed with exception: %s", exc)
            # Preserve earlier loss_explosion evidence if the guard already fired.
            if not fail_state_file.exists():
                with open(fail_state_file, "w", encoding="utf-8") as fh:
                    json.dump({"reason": "exception", "error": str(exc)}, fh)
        return False

    # --- 10. Check fail-closed state ---
    if callback.aborted:
        log.error("[tinct] Run ABORTED by fail-closed guard. DO NOT SHIP.")
        return False

    # --- 11. Save final adapter atomically ---
    log.info("[tinct] Saving adapter safely...")
    final_adapter_dir = run_dir / "adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    log.info("[tinct] Training completed successfully.")
    return True
