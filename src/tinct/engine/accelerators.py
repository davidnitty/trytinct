"""Acceleration backends for model loading.

``tinct train --accelerator unsloth`` swaps the standard Hugging Face loading
path for **Unsloth**, which rewrites the underlying Triton kernels for Llama
(and Qwen) models: up to ~70% lower VRAM and ~2x faster training — the magic
bullet for low-end (8–16 GB) hardware.

Unsloth is CUDA-only and is NOT installed with the core or ``[train]`` extras;
requesting it without the package yields an actionable
``pip install 'tinct[unsloth]'`` hint. All heavy imports here are lazy.
"""

from __future__ import annotations

from tinct.utils.logging import get_logger

log = get_logger("tinct.engine.accelerators")

UNSLOTH_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_model_with_accelerator(
    model_name: str,
    accelerator: str = "none",
    lora_rank: int = 16,
    max_seq_length: int = 2048,
):
    """Load the base model and apply LoRA, optionally via Unsloth.

    Returns ``(model, tokenizer)``. ``accelerator`` is ``"none"`` (standard
    Hugging Face path) or ``"unsloth"`` (massive VRAM reduction).
    """
    if accelerator == "unsloth":
        return _load_unsloth(model_name, lora_rank, max_seq_length)
    return _load_standard(model_name, lora_rank)


def _load_unsloth(model_name: str, lora_rank: int, max_seq_length: int):
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is not installed. It is required for low-VRAM training.\n"
            "Install with: pip install 'tinct[unsloth]'"
        ) from exc

    log.info("[tinct] Loading %s via Unsloth for low-VRAM training...", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,  # QLoRA by default for maximum memory savings
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=UNSLOTH_TARGET_MODULES,
        lora_alpha=lora_rank,
        lora_dropout=0,  # Unsloth recommends 0 dropout
        bias="none",
        use_gradient_checkpointing="unsloth",  # ~30% more VRAM saved
        random_state=42,
    )
    return model, tokenizer


def _load_standard(model_name: str, lora_rank: int):
    """Standard Hugging Face fallback (CPU-safe dtype)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("[tinct] Loading %s via standard Hugging Face...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if (has_cuda and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch_dtype,
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=UNSLOTH_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer