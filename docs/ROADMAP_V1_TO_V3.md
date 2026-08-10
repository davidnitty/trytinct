# TryTinct Backend Development Roadmap (V1 to V3)

This guide outlines the step-by-step backend architecture for building the **TryTinct** CLI. The strategy is to build a rock-solid foundation with dense Llama models in V1, expand to Qwen's specific optimizations in V2, and finally tackle the complex MoE/Reasoning architecture of Deepseek in V3.

---

## Current milestone: V0 (Llama)

The current codebase ships V0: a validated, end-to-end Llama LoRA/QLoRA
pipeline with signed ship evidence. See `BACKEND_STRUCTURE_AND_SECURITY.md`.

## 🏗️ Tech Stack & Core Dependencies

- **Language:** Python 3.10+
- **CLI Framework:** `Typer` (Command prefix will be `tinct`)
- **Training Engine:** `Transformers`, `TRL` (Transformer Reinforcement Learning), `PEFT`
- **Acceleration:** `Unsloth` (V2/Qwen), `DeepSpeed` (ZeRO-3), `bitsandbytes`
- **Inference & Eval:** `vLLM` or `SGLang`, `lm-evaluation-harness`
- **Data Handling:** `datasets`, `sentence-transformers`

### Suggested Directory Structure
```text
trytinct/
├── src/
│   ├── cli/            # Typer commands: `tinct advise`, `tinct train`, `tinct ship`
│   ├── core/           # Data Doctor, Config Generator, Rule Engine
│   ├── engine/         # Layer Streaming, MoE Routing, VRAM Management
│   ├── trainers/       # SFT, DPO, GRPO custom training loops
│   ├── evals/          # vLLM wrappers, benchmark suites, reward modeling
│   └── utils/          # Logging, hardware detection, safetensors parsing
├── tests/
├── pyproject.toml
└── README.md
```

## Roadmap

- **V1 (dense Llama):** solid foundation — LoRA/QLoRA SFT, eval gating, evidence.
- **V2 (Qwen):** Unsloth acceleration, Qwen-specific optimizations, GRPO.
- **V3 (DeepSeek):** MoE routing, layer streaming, Reasoning/RL reward modeling.
