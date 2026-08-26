# tinct v0.4.0 — Safety Certification

**Release date:** 2026-08-26
**Commit:** `b34ed73`

## What's New

### Behavioral Certification Gates

tinct now certifies model *behavior*, not just training mechanics.

**Canary Leakage Detection**
- Injects unique canary strings during SFT training
- Checks if model memorized training data during eval
- Fails if >50% of canaries are reproduced
- Prevents PII/trade secret leakage in production

**Refusal Regression Testing**
- Compares safety refusals between base model and adapter
- Tests 20 prompts across violence, fraud, self-harm, illegal acts
- Fails if refusal rate drops >20%
- Prevents DPO from accidentally teaching unsafe preferences

### Unsloth Acceleration (V0.3)

- `--accelerator unsloth` flag for low-VRAM training
- Reduces memory usage by 60-70% via Triton kernels
- Enables 7B-8B models on 8-16GB machines
- Available for both SFT and DPO

### Qwen Model Support (V0.3)

- Strict Qwen chat template validation
- Detects missing `<|im_start|>`/`<|im_end|>` tokens
- Validates token balance and turn structure
- Family-aware Data Doctor via `--model` flag

### DPO Reward Inversion Guards (V0.2)

- Monitors reward scores during DPO training
- Halts if model prefers rejected answers (3 consecutive inversions)
- Persists reward trajectory to `dpo_metrics.json`
- Ship gates on positive final reward margin

---

## Breaking Changes

None. All changes are additive.

---

## Upgrade Guide

```bash
pip install --upgrade tinct
```

### Run the new safety gates

```bash
# 1. Train (SFT injects canaries automatically)
tinct train --dataset examples/good_data.jsonl --accelerator unsloth

# 2. Evaluate behavior, not just loss
tinct eval --safety

# 3. Ship (refuses on any FAILED safety gate)
tinct ship
```

Safety gate results are folded into the signed evidence manifest
(`.tinct/runs/<run>/safety_gates.json` → `safety_gates` in the evidence report),
so a SHIP verdict now carries cryptographic proof that the model neither
leaks its training data nor regressed on safety refusals.

---

## What's next

- **V0.5:** Toxicity regression gate (base vs adapter toxicity scoring).
- **Frontend V0:** dashboard to visualize evidence, metrics, and safety gates.
