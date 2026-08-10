# TryTinct

**CLI-first post-training stack for LLMs.**

TryTinct validates instruction data, fine-tunes a **Llama** adapter with
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
tinct eval    --checkpoint runs/latest
tinct ship    --checkpoint runs/latest
tinct security check
```

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
