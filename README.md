# tinct

**CLI-first post-training stack for LLMs.**

tinct validates instruction data, fine-tunes a **Llama** adapter with
**LoRA/QLoRA**, evaluates the result, and produces a **SHIP / DON'T-SHIP**
decision backed by a **signed cryptographic evidence report**.

```
validate → advise → train → eval → ship
```

## Quick start

```bash
pip install -e .                 # lightweight core only
tinct init my-project
cd my-project
tinct validate data.jsonl
tinct train   --model meta-llama/Llama-3.1-8B --data data.jsonl   # needs [train]
tinct eval    --run latest
tinct ship    --run latest
tinct security check
```

> State (config, runs, cache, keys, evidence) lives under `.tinct/`.

## Design principles

- **Local-first** — everything runs on your machine.
- **Secure by default / fail-closed** — validation blocks training; eval blocks shipping.
- **Reproducible & audit-friendly** — seeded runs, artifact hashes, signed manifests.
- **Lightweight core** — importing `tinct` never pulls torch/transformers.

## Heavy ML dependencies (optional extras)

Heavy packages are loaded lazily only when a command needs them:

```bash
pip install -e ".[train]"   # torch, transformers, TRL, PEFT, bitsandbytes, ...
pip install -e ".[eval]"    # scikit-learn, ...
pip install -e ".[full]"    # everything
pip install -e ".[dev]"     # pytest
```

If a heavy engine is missing, the command fails with a clear
`pip install tinct[train]` style hint rather than a confusing import error.


## Roadmap status

- **V0 (current):** Llama LoRA/QLoRA end-to-end with signed ship evidence.
- **V2/V3:** Qwen + DeepSeek optimizations (see roadmap doc).

> **Note:** actually fine-tuning Llama-3.1-8B requires a suitable GPU and the
> model weights. The core CLI (init/validate/eval/ship/security check) runs on
> CPU with no ML dependencies installed.

## The V0.1 happy path (acceptance test, GPU box)

The full certification loop — train → eval (generation smoke test) → ship
(signed evidence):

```bash
pip install -e ".[train]"
tinct init demo meta-llama/Llama-3.2-1B     # needs HF token (gated model)
cd demo
cp ../examples/good_data.jsonl .
tinct train --dataset good_data.jsonl --max-loss-threshold 10.0
tinct eval                                  # runs generation smoke test
tinct ship                                  # hashes adapter + signs evidence
```

What happens:

1. `tinct train` — validates data, chunks + hashes the base model, fine-tunes
   with the fail-closed loss guard (threshold from `--max-loss-threshold`).
2. `tinct eval` — the **generation smoke test** loads base model + adapter,
   runs the 3 prompts, and fails the gate on empty or looping text; writes
   `.tinct/runs/<name>/eval_report.json`.
3. `tinct ship` — the **certification engine**:
   - refuses if `fail_state.json` exists (guard tripped) → `DON'T SHIP` (exit 2)
   - requires `eval_report.json` with `status: PASS` → else `DON'T SHIP`
   - hashes the adapter (`adapter_sha256`, deterministic), signs the evidence
     manifest with Ed25519, and saves it under `.tinct/evidence/`
   - prints `VERDICT: SHIP`

`examples/good_data.jsonl` is a 24-row Llama-3 chat-template dataset for the
happy path; `examples/broken_data.jsonl` is the garbage dataset that trips the
loss guard.

## V0.1-GPU milestone: prove the fail-closed guard (no 4-hour run)

Trigger the loss-explosion guard intentionally with the bundled broken dataset
and an artificially low threshold:

```bash
pip install -e ".[train]"
tinct init smoke meta-llama/Llama-3.2-1B      # requires HF token (gated model)
cd smoke
tinct train --dataset ../examples/broken_data.jsonl --max-loss-threshold 0.1
```

Expected: the run starts, the loss is garbage, the **Fail-Closed Callback** trips
almost immediately, and you get:

```
[tinct] FATAL: Loss exploded ...
[tinct] VERDICT: DON'T SHIP (Training failed / fail-closed guard).
```

with `.tinct/runs/<name>/fail_state.json` recording `reason: loss_explosion` and
the structured logs populating `.tinct/runs/<name>/train_log.jsonl`.

**CPU-only / no HF token?** Use a tiny non-gated Llama-family model instead —
this is the exact pipeline that was verified end-to-end on a CPU box:

```bash
tinct init smoke hf-internal-testing/tiny-random-LlamaForCausalLM
cd smoke
tinct train --dataset ../examples/broken_data.jsonl --max-loss-threshold 2.0
# -> FATAL: Loss 10.38 ... halting immediately. / VERDICT: DON'T SHIP
```

The same guard logic is also covered by unit tests (`test_sft_trainer.py`) that
run with no ML dependencies installed — `FailClosedCore` is exercised directly.

To let it run past the guard instead, omit `--max-loss-threshold` and watch
`train_log.jsonl` populate (it will still halt at the first NaN/Inf/over-
threshold step).
