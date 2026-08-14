"""DPO trainer — the heart of V0.2, with the Reward Inversion Guard.

DPO (Direct Preference Optimization) teaches a model what to *prefer* —
trained on pairs of ``(prompt, chosen, rejected)`` answers. DPO is unstable:
when it goes wrong the model does not merely fail, it **anti-aligns** — it
learns to *prefer the rejected (bad) answers*.

The **Reward Inversion Guard** monitors DPO rewards every log step and halts
the run once ``rejected`` consistently outranks ``chosen``, writing
``fail_state.json`` (``reason: reward_inversion``) so the run can never ship.

The guard logic lives in :class:`RewardInversionCore` — an import-friendly,
unit-testable class with no heavy ML dependency. The model loading and training
run lazily inside :func:`run_dpo`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tinct.engine.deps import ensure_train_deps
from tinct.utils.logging import get_logger

log = get_logger("tinct.trainers.dpo")

# TRL logs DPO rewards under these keys (tolerates 0.x and 1.x spellings).
_CHOSEN_KEYS = ("rewards/chosen", "rewards_chosen")
_REJECTED_KEYS = ("rewards/rejected", "rewards_rejected")


class RewardInversionCore:
    """Import-friendly DPO reward-inversion guard (no transformers import).

    :meth:`on_log` is called for each training log. It persists a structured
    log, tracks how many consecutive steps ``rejected > chosen``, and after a
    tolerance threshold returns True (and writes ``fail_state.json``) so the
    run is killed and marked DON'T SHIP.
    """

    def __init__(self, log_path: Path, fail_path: Path, threshold: int = 3) -> None:
        self.log_path = log_path
        self.fail_path = fail_path
        self.threshold = threshold
        self.inversion_count = 0
        self.aborted = False

    def on_log(self, state, control, logs=None) -> bool:
        """Handle one log entry. Returns True when the run must halt."""
        if not logs:
            return False

        step = getattr(state, "global_step", 0)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, **logs}) + "\n")

        chosen = _first_present(logs, _CHOSEN_KEYS)
        rejected = _first_present(logs, _REJECTED_KEYS)
        if chosen is None or rejected is None:
            return False
        chosen, rejected = float(chosen), float(rejected)

        if rejected > chosen:
            self.inversion_count += 1
            log.warning("[tinct] Reward Inversion at step %s (chosen %.4f < rejected %.4f)",
                        step, chosen, rejected)
            if self.inversion_count >= self.threshold:
                log.error("[tinct] FATAL: model is anti-aligning (preferring bad answers); halting.")
                self.aborted = True
                control.should_training_stop = True
                with open(self.fail_path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "reason": "reward_inversion",
                        "chosen_reward": chosen,
                        "rejected_reward": rejected,
                        "step": step,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }, fh)
                return True
        else:
            self.inversion_count = 0  # healthy — reset the persistence counter
        return False


def _first_present(logs: dict, keys) -> float | None:
    for k in keys:
        v = logs.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def run_dpo(
    model_name_or_path: str,
    dataset_path: Path,
    run_dir: Path,
    beta: float = 0.1,
    lora_rank: int = 16,
    per_device_batch_size: int = 1,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-5,
    logging_steps: int = 10,
    max_length: int = 1024,
    num_train_epochs: int = 1,
    inversion_threshold: int = 3,
) -> bool:
    """Execute guarded DPO training.

    Returns True on success, or False if halted by a fail-closed guard.
    """
    ensure_train_deps()
    # --- 1. Lazy imports ---
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from peft import LoraConfig
        from trl import DPOTrainer, DPOConfig
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "DPO dependencies missing.\nInstall with: pip install 'tinct[train]'\n"
            f"Original error: {exc}"
        ) from exc

    # --- 2. Setup ---
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train_log.jsonl"
    fail_state_file = run_dir / "fail_state.json"

    # --- 3. The Reward Inversion Guard (logic in RewardInversionCore) ---
    class RewardInversionGuard(TrainerCallback, RewardInversionCore):
        def __init__(self, log_path: Path, fail_path: Path, threshold: int):
            RewardInversionCore.__init__(self, log_path, fail_path, threshold)

        def on_log(self, args, state, control, logs=None, **kwargs):
            return RewardInversionCore.on_log(self, state, control, logs)

    # --- 4. Load model & tokenizer ---
    log.info("[tinct] Loading model and tokenizer: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CPU-safe dtype: bf16 on CUDA that supports it, else float32.
    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, device_map="auto",
    )

    # --- 5. Load dataset (expects prompt/chosen/rejected) ---
    log.info("[tinct] Loading DPO dataset: %s", dataset_path)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # --- 6. Configure DPO + LoRA ---
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    dpo_config = DPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        max_length=max_length,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        report_to="none",
        bf16=use_bf16,
        fp16=has_cuda and not use_bf16,
        optim="adamw_torch",
    )

    # --- 7. Initialize trainer with guard ---
    guard = RewardInversionGuard(log_file, fail_state_file, inversion_threshold)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[guard],
    )

    # --- 8. Execute ---
    log.info("[tinct] Starting DPO training with Reward Inversion Guard...")
    try:
        trainer.train()
    except Exception as exc:
        if guard.aborted:
            log.error("[tinct] Run halted by Reward Inversion Guard (DO NOT SHIP).")
        else:
            log.error("[tinct] Training crashed: %s", exc)
            if not fail_state_file.exists():
                with open(fail_state_file, "w", encoding="utf-8") as fh:
                    json.dump({"reason": "exception", "error": str(exc)}, fh)
        return False

    if guard.aborted:
        log.error("[tinct] Run ABORTED due to Reward Inversion (DO NOT SHIP).")
        return False

    # --- 9. Save adapter ---
    log.info("[tinct] Saving adapter safely...")
    trainer.save_model(str(run_dir / "adapter"))
    log.info("[tinct] DPO training completed successfully.")
    return True
