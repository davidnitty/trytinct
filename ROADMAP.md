# tinct Roadmap

Current version: **v1.0.0**
Last updated: 2026-08

---

## In Progress

### v1.1.0 — MoE Support (Mixture of Experts) 🚧
- **Kimi K3 Instruct** support
- **DeepSeek V2/V3** support
- Expert-level layer streaming (only load active experts into VRAM)
- MoE-aware checkpointing (save only active experts to save disk space)
- Custom expert routing logic for the `accelerators.py` engine

**Phase 1 (Mixtral vanguard) — landed:**
- ✅ Expert Collapse safety gate (`safety/moe_gates.py`): forward-hook router
  tracking, <1% utilization → FAIL, runs in `tinct eval --safety` / `tinct certify`
- ✅ Routing Regression gate: adapter vs base utilization comparison
  (>50% relative drop on experts the base used ≥2% → FAIL) — base-model
  laziness is never blamed on the adapter
- ✅ MoE expert-offloading engine (`engine/moe.py`): CPU-resident experts with
  LRU streaming (`--offload-experts` on train/eval/certify; offload stats in evidence)
- ✅ Structural router detection (gate + experts, no name regexes) — covers
  Mixtral, Qwen-MoE, DeepSeek, and future families
- ✅ Mixtral routes to Mistral `[INST]` template validation (Data Doctor)

---

## Completed

### v1.0.0 — Integration Layer ✅
- `tinct certify` command for external adapters (Unsloth, LLaMA-Factory, Axolotl)
- Mistral / Mixtral model family support
- `tinct doctor` preflight check
- Model Registry (Tier 1/2/3)
- Fixed critical `TrainerControl` callback bug
- Detoxify fallback + shared HF cache

### v0.5.0 — Toxicity Gate ✅
- Two-tier toxicity scoring: keyword heuristic (dependency-free) +
  optional Detoxify model (`tinct[toxicity]`)
- Base vs adapter comparison on 10 neutral prompts
- Ship gate: adapter toxicity >2x baseline → FAIL
- Aggregate gate verdict (`safety_gates.result`) across all three gates

### v0.4.0 — Safety Gates ✅
- Canary leakage detection (injection + eval gate, >50% → FAIL)
- Refusal regression testing (20 safety prompts, >20% drop → FAIL)
- `tinct eval --safety` command → `safety_gates.json`
- Safety gates fold into the signed evidence manifest
- Ship blocks on safety failures

### v0.3.0 — Acceleration + Qwen ✅
- Unsloth accelerator engine (`engine/accelerators.py`)
- `--accelerator unsloth` for 60-70% VRAM reduction (SFT + DPO)
- Fail-fast accelerator guard before model download (exit 3)
- Qwen chat template validation (`<|im_start|>` / `<|im_end|>`, balance + turns)
- Llama-3 template validation (BOS / EOT)
- Model-family-aware Data Doctor (`tinct validate --model <id>`)

### v0.2.0 — DPO Certification ✅
- DPO training with `trl.DPOTrainer` (`tinct train --method dpo`)
- Reward inversion guard (3 consecutive inversions → halt → DON'T SHIP)
- Reward trajectory tracking (`dpo_metrics.json`, persisted even on halt)
- Ship gates: positive final reward margin required
- Dual trl API support (0.x + 1.x reward-key spellings)

### v0.1.0 — Core Certification Engine ✅
- CLI scaffold: `init`, `validate`, `advise`, `train`, `eval`, `ship`
- Data Doctor: schema validation, template checks, dedup, fail-closed min rows
- Fail-closed SFT training with loss monitoring (NaN / Inf / threshold)
- Ed25519 evidence signing (`tinct security check --run`)
- Model family gate (Llama only)
- `.tinct/` project structure + secure key permissions (0700 on POSIX)

---

## Planned

### Later — Hardening + Dashboard
- **Frontend dashboard**: visualize evidence, safety gates, DPO metrics
- **Multi-GPU training**: DeepSpeed ZeRO-3 integration
- **Model export**: GGUF / ONNX export with signed provenance
- **CI integration**: `tinct` as a GitHub Action for automated certification
- **Documentation site**: full API reference + tutorials

### Later — Advanced Safety
- **Prompt injection resistance**: eval gate for injection attacks
- **PII leakage**: structured PII detection beyond canaries
- **Bias amplification**: compare bias scores pre/post training
- **Custom safety policies**: user-defined gate configurations

### v2.0.0 — Distributed MoE
- **MLA (Multi-head Latent Attention)**: compressed KV-cache handling
- **GRPO (Group Relative Policy Optimization)**: DeepSeek-R1 style training
- **Multi-node training**: distributed MoE fine-tuning

---

## Versioning Policy

- **v0.x**: Rapid iteration, breaking changes allowed
- **v1.0**: Stable API, backwards-compatible evidence format
- **v2.0**: Major architectural changes (MLA, GRPO, distributed)

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
