"""DPO trainer — the heart of V0.2, with the Reward Inversion Guard.

DPO (Direct Preference Optimization) teaches a model what to *prefer* —
trained on pairs of ``(prompt, chosen, rejected)`` answers. DPO is unstable:
when it goes wrong the model does not merely fail, it **anti-aligns** — it
learns to *prefer the rejected (bad) answers*.

The **Reward Inversion Guard** monitors DPO rewards every log step and halts
the run once ``rejected`` consistently outranks ``chosen``, writing
``fail_state.json`` (``reason: reward_inversion``) so the run can never ship.

:class:`RewardInversionCore` is a **pure, import-friendly** tracker (no ML
dependency, no file I/O): it accumulates the reward trajectory, tracks
consecutive inversions, and produces :meth:`final_metrics` — the evidence that
answers *"did alignment happen?"*. The heavy ML path (and the file I/O) runs
lazily inside :func:`run_dpo`, which persists ``dpo_metrics.json`` on success
**and** on a guard halt, so even an aborted run leaves an auditable trail.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tinct.engine.deps import ensure_train_deps
from tinct.safety.canaries import generate_canaries, save_canaries
from tinct.utils.logging import get_logger

log = get_logger("tinct.trainers.dpo")

# TRL logs DPO rewards under these keys (tolerates 0.x and 1.x spellings).
_CHOSEN_KEYS = ("rewards/chosen", "rewards_chosen")
_REJECTED_KEYS = ("rewards/rejected", "rewards_rejected")


class RewardInversionCore:
    """Pure DPO reward monitor — tracks inversions AND the reward trajectory.

    Import-friendly and GPU-free: feed one log step at a time via
    :meth:`observe` (or :meth:`on_log` when you have the raw log dict), then
    read :meth:`final_metrics` for the evidence bundle.
    """

    def __init__(self, inversion_threshold: int = 3) -> None:
        self.inversion_threshold = inversion_threshold
        self.consecutive_inversions = 0
        self.aborted = False
        self.reason: Optional[str] = None
        # Reward trajectory (per logged step).
        self.margins: list[dict] = []

    # -- feeding ------------------------------------------------------------

    def observe(self, chosen, rejected, step: int) -> None:
        """Feed one log step. Handles None (missing rewards) gracefully."""
        if chosen is None or rejected is None:
            return  # missing rewards: skip, don't count as an inversion

        chosen = float(chosen)
        rejected = float(rejected)
        margin = chosen - rejected
        self.margins.append({
            "step": step,
            "chosen": chosen,
            "rejected": rejected,
            "margin": margin,
        })

        if rejected > chosen:
            self.consecutive_inversions += 1
            if self.consecutive_inversions >= self.inversion_threshold:
                self.aborted = True
                self.reason = "reward_inversion"
        else:
            self.consecutive_inversions = 0  # healthy reset

    def on_log(self, state, control, logs=None) -> bool:
        """Convenience: extract rewards from a log dict, observe, and flag the
        trainer to stop. Returns True when the run must halt."""
        if not logs:
            return False
        chosen = _first_present(logs, _CHOSEN_KEYS)
        rejected = _first_present(logs, _REJECTED_KEYS)
        self.observe(chosen, rejected, getattr(state, "global_step", 0))
        if self.aborted:
            control.should_training_stop = True
            return True
        return False

    # -- reporting ----------------------------------------------------------

    def final_metrics(self) -> Optional[dict]:
        """Summary for the evidence bundle. None if no rewards were ever logged."""
        if not self.margins:
            return None
        final = self.margins[-1]
        margins = [m["margin"] for m in self.margins]
        return {
            "training_method": "dpo",
            "final_chosen_reward": final["chosen"],
            "final_rejected_reward": final["rejected"],
            "final_reward_margin": final["margin"],
            "max_reward_margin": max(margins),
            "min_reward_margin": min(margins),
            "num_logged_steps": len(self.margins),
            "reward_inversion_detected": self.aborted,
        }


def _first_present(logs: dict, keys) -> Optional[float]:
    for k in keys:
        v = logs.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _persist_metrics(core: RewardInversionCore, run_dir: Path) -> None:
    """Write ``dpo_metrics.json`` if any rewards were logged. Always safe to
    call — on success AND on a guard halt (auditable trail)."""
    metrics = core.final_metrics()
    if metrics:
        (run_dir / "dpo_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )


def run_dpo(
    model_name_or_path: str,
    dataset_path: Path,
    run_dir: Path,
    beta: float = 0.1,
    lora_rank: int = 16,
    accelerator: str = "none",
    max_seq_length: int = 2048,
    per_device_batch_size: int = 1,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-5,
    logging_steps: int = 10,
    max_length: int = 1024,
    num_train_epochs: int = 1,
    inversion_threshold: int = 3,
) -> bool:
    """Execute guarded DPO training.

    ``accelerator`` is ``"none"`` (standard HF path) or ``"unsloth"`` for
    low-VRAM Triton-kernel acceleration (requires ``tinct[unsloth]``).

    Returns True on success, or False if halted by a fail-closed guard.
    ``dpo_metrics.json`` is persisted in either case.
    """
    ensure_train_deps()
    # --- 1. Lazy imports ---
    try:
        import torch
        from transformers import TrainerCallback
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

    # Track canaries so `tinct eval --safety` runs the leakage gate uniformly.
    # (DPO trains preference pairs, so we don't inject canary text into its
    # dataset — canary memorization is an SFT-phase concern.)
    save_canaries(run_dir, generate_canaries(num_canaries=10))

    # --- 3. The Reward Inversion Guard ---
    class RewardInversionGuard(TrainerCallback, RewardInversionCore):
        def __init__(self, threshold: int, log_path: Path, fail_path: Path):
            RewardInversionCore.__init__(self, inversion_threshold=threshold)
            self.log_path = log_path
            self.fail_path = fail_path

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return False
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": state.global_step, **logs}) + "\n")

            halted = RewardInversionCore.on_log(self, state, control, logs)
            if halted and not self.fail_path.exists():
                last = self.margins[-1]
                with open(self.fail_path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "reason": self.reason,
                        "chosen_reward": last["chosen"],
                        "rejected_reward": last["rejected"],
                        "step": last["step"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }, fh)
            return halted

    # --- 4. Load model & tokenizer via the accelerator engine ---
    # The accelerator applies LoRA itself, so DPOTrainer gets the final model
    # with no peft_config — one code path for both backends.
    from tinct.engine.accelerators import load_model_with_accelerator

    log.info("[tinct] Loading model with accelerator: %s", accelerator)
    model, tokenizer = load_model_with_accelerator(
        model_name=model_name_or_path,
        accelerator=accelerator,
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # CPU-safe dtype logic: bf16 only on CUDA that supports it; float32 on CPU.
    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())

    # --- 5. Load dataset (expects prompt/chosen/rejected) ---
    log.info("[tinct] Loading DPO dataset: %s", dataset_path)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # --- 6. Configure DPO ---
    # Unsloth is more memory-efficient, so DPO can afford a slightly larger
    # batch when it is the active accelerator (ref_model=None already means the
    # base model is used as the reference, which Unsloth handles natively).
    dpo_config = DPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        beta=beta,
        per_device_train_batch_size=2 if accelerator == "unsloth" else per_device_batch_size,
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
    guard = RewardInversionGuard(inversion_threshold, log_file, fail_state_file)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
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
    finally:
        # Always persist the reward trajectory — even an aborted run leaves an
        # auditable trail (reward_inversion_detected=true records WHY it failed).
        _persist_metrics(guard, run_dir)

    if guard.aborted:
        log.error("[tinct] Run ABORTED due to Reward Inversion (DO NOT SHIP).")
        return False

    # --- 9. Save adapter ---
    log.info("[tinct] Saving adapter safely...")
    trainer.save_model(str(run_dir / "adapter"))
    log.info("[tinct] DPO training completed successfully.")
    return True
