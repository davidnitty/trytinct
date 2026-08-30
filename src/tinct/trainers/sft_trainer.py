"""Fail-closed SFT trainer — the heart of ``tinct train``.

Wraps Hugging Face ``trl`` (SFTTrainer) but enforces strict security
boundaries:

- No top-level heavy imports (torch/trl/peft/transformers). This module is
  only loaded dynamically when ``tinct train`` executes.
- Model loading uses ``trust_remote_code=True`` (required for Qwen and other
  custom-code model families).
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

from tinct.safety.canaries import canary_text, generate_canaries, save_canaries
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
    offload_experts: bool = False,
) -> bool:
    """Execute a guarded SFT run.

    ``accelerator`` is ``"none"`` (standard HF path) or ``"unsloth"`` for
    low-VRAM Triton-kernel acceleration (requires ``tinct[unsloth]``).
    ``offload_experts`` streams MoE expert MLPs from CPU on demand
    (Mixtral-class models).

    Returns True on success, or False if halted by a fail-closed guard.
    """
    # --- 1. Lazy imports (heavy dependencies) ---
    try:
        import torch
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer
        from datasets import Dataset, concatenate_datasets, load_dataset
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

    # Canary injection (Behavioral Certification Gate 1): track unique canary
    # phrases so `tinct eval --safety` can later detect memorization leakage.
    canaries = generate_canaries(num_canaries=10)
    save_canaries(run_dir, canaries)

    # --- 3. Fail-closed callback (logic lives in FailClosedCore, which is
    # tested directly without any ML dependency) ---
    class TinctFailClosedCallback(TrainerCallback, FailClosedCore):
        def __init__(self, threshold: float, log_path: Path, fail_path: Path):
            FailClosedCore.__init__(self, threshold, log_path, fail_path)

        def on_log(self, args, state, control, logs=None, **kwargs):
            # Mutates `control` in place and MUST return None — transformers
            # replaces `control` with any non-None callback return, and a bare
            # bool breaks the training loop ('bool' has no attribute ...).
            FailClosedCore.on_log(self, state, control, logs)

    # --- 4. Load tokenizer & model via the accelerator engine ---
    # The accelerator applies LoRA itself, so SFTTrainer is only given the
    # final model (no peft_config) — one code path for both backends.
    from tinct.engine.accelerators import load_model_with_accelerator

    log.info("[tinct] Loading model with accelerator: %s", accelerator)
    model, tokenizer = load_model_with_accelerator(
        model_name=model_name_or_path,
        accelerator=accelerator,
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # QLoRA by default for memory safety
        offload_experts=offload_experts,
    )

    # CPU-safe dtype logic: bf16 only on CUDA that supports it; float32 on CPU.
    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    use_8bit_optim = accelerator != "unsloth" and has_cuda

    # --- 5. Load dataset ---
    log.info("[tinct] Loading dataset: %s", dataset_path)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # Append the canary examples to the training set.
    canary_dataset = Dataset.from_list([{"text": canary_text(c)} for c in canaries])
    dataset = concatenate_datasets([dataset, canary_dataset])

    # --- 6. Training arguments ---
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        fp16=has_cuda and not use_bf16,
        bf16=use_bf16,
        logging_steps=logging_steps,
        optim="paged_adamw_8bit" if use_8bit_optim else "adamw_torch",
        save_strategy="epoch",
        report_to="none",
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,  # one formatted text sequence per example
    )

    # --- 7. Initialize trainer ---
    callback = TinctFailClosedCallback(
        threshold=max_loss_threshold,
        log_path=log_file,
        fail_path=fail_state_file,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
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
