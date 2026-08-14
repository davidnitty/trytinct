# tinct

![CI](https://github.com/davidnitty/trytinct/actions/workflows/ci.yml/badge.svg)

**CLI-first post-training stack for LLMs.** Validate → train → eval → ship,
with a fail-closed security model and **cryptographically signed evidence** for
every checkpoint you certify.

```
tinct init → tinct validate → tinct train → tinct eval → tinct ship
```

tinct validates instruction data, fine-tunes a **Llama** adapter with
**LoRA/QLoRA**, runs a generation smoke test, and produces a **SHIP /
DON'T-SHIP** decision backed by an **Ed25519-signed evidence manifest**.

---

## Why tinct?

- **Fail-closed by default** — the Data Doctor blocks bad data before training;
  a loss-explosion guard kills runs instantly; the eval gate blocks bad
  checkpoints; the ship engine refuses to certify anything without evidence.
- **Proof, not vibes** — every run records hashes of the data, base-model
  chunks, adapter weights, and logs, then signs them with an Ed25519 key.
- **Local-first** — everything runs on your machine; state lives in `.tinct/`.
- **Lightweight core** — importing `tinct` never pulls torch/transformers; ML
  deps are optional extras loaded lazily only when a command needs them.

## Quick start

```bash
pip install -e ".[train]"                      # core + training stack
tinct init my-project meta-llama/Llama-3.1-8B  # scaffold + signing key
cd my-project
tinct validate data.jsonl                      # Data Doctor (fail-closed)
tinct train  --dataset data.jsonl --max-loss-threshold 10.0
tinct eval                                      # generation smoke test
tinct ship                                      # signed SHIP/DON'T-SHIP
tinct security check --run latest               # verify the signature
```

> All state (config, runs, cache, keys, evidence) lives under `.tinct/` —
> git-ignored by default.

## CLI reference

| Command | Purpose |
| --- | --- |
| `tinct init <name> <model>` | Scaffold a project + Ed25519 signing key. |
| `tinct validate <dataset>` | Data Doctor: schema, empties, duplicates, lengths. |
| `tinct advise <dataset>` | Recommend a post-training method. |
| `tinct train` | SFT (loss guard) or `--method dpo` (reward-inversion guard). |
| `tinct eval` | Generation smoke test → `eval_report.json`. |
| `tinct ship` | Certification: gate checks + adapter hash + signed evidence. |
| `tinct security check` | Audit secrets, key perms, evidence signatures. |

### The fail-closed loss guard

Training halts instantly on **NaN / Inf / over-threshold** loss, writes
`fail_state.json`, and the run is forever marked **DON'T SHIP**:

```
[tinct] FATAL: Loss 10.3804 … halting immediately.
[tinct] VERDICT: DON'T SHIP (Training failed / fail-closed guard).
```

### The certification engine (`tinct ship`)

1. Refuses if `fail_state.json` exists (guard tripped).
2. Requires `eval_report.json` with `status: PASS`.
3. Hashes the adapter (`adapter_sha256`) — proves exactly which weights ship.
4. Signs the evidence manifest with Ed25519 and saves it under `.tinct/evidence/`.

## Optional extras

```bash
pip install -e .              # lightweight core (typer, pydantic, cryptography)
pip install -e ".[train]"     # + torch, transformers, TRL, PEFT, accelerate, datasets
pip install -e ".[eval]"      # + scikit-learn
pip install -e ".[full]"      # everything
pip install -e ".[dev]"       # + pytest
```

If a heavy engine is missing, the command fails with a clear
`pip install tinct[train]` hint instead of a confusing import error.

## Project layout

```
src/tinct/
├── cli/          # typer commands (init, validate, advise, train, eval, ship, security)
├── core/         # config, Data Doctor, model gate, project/init logic
├── engine/       # model chunking, streaming, lazy dep guards
├── trainers/     # fail-closed SFT trainer (TRL)
├── evals/        # generation smoke test, loss gate, harness
├── security/     # Ed25519 signing, evidence manifests, audits
└── storage/      # TinctPaths — single source of truth for the state tree
```

## Examples

- `examples/good_data.jsonl` — 24 rows of Llama-3 chat templates (happy path).
- `examples/broken_data.jsonl` — garbage data that trips the loss guard.

## The real-GPU verification (closing V0.1)

The true acceptance test runs the same loop on the **actual target model**
(`Llama-3.1-8B`, or `Llama-3.2-1B` to cut GPU time ~8x). One command on a GPU
box (checks GPU + HF token first, then init → validate → train → eval → ship →
verify signature, exiting 0 only on a verified SHIP):

```bash
bash scripts/gpu_verify.sh meta-llama/Llama-3.1-8B 10.0
# or: bash scripts/gpu_verify.sh meta-llama/Llama-3.2-1B 10.0   (fastest)
```

Prereqs on that box: NVIDIA GPU, `pip install -e ".[train]"`, and a Hugging
Face token (`huggingface-cli login` or `HF_TOKEN`) — Llama models are gated.

**One-shot on a fresh GPU pod (RunPod / Vast.ai / any NVIDIA box):**

```bash
HF_TOKEN=hf_... bash scripts/pod_bootstrap.sh meta-llama/Llama-3.2-1B 10.0
```

`scripts/pod_bootstrap.sh` clones the repo, installs `.[train]`, checks the
GPU + token, and runs the full verification — a real SHIP in ~15 minutes,
typically well under a dollar of pod time on 3.2-1B.

Verify a run's evidence structure and signature after the fact:

```bash
python scripts/verify_evidence.py <run_id>   # exit 0 = signed SHIP, all artifacts hashed
```

There is also a `.github/workflows/gpu-verify.yml` (workflow_dispatch) that
runs the same script on a self-hosted GPU runner (label `gpu`, `HF_TOKEN`
secret) — see the workflow comments to enable it.

## Docs

- [`BACKEND_STRUCTURE_AND_SECURITY.md`](docs/BACKEND_STRUCTURE_AND_SECURITY.md)
- [`ROADMAP_V1_TO_V3.md`](docs/ROADMAP_V1_TO_V3.md)

## Roadmap

- **V0 (current):** Llama LoRA/QLoRA end-to-end with signed ship evidence.
- **V0.2:** DPO with the **Reward Inversion Guard** (`tinct train --method dpo`
  on `prompt/chosen/rejected` data — `examples/dpo_data.jsonl`). The guard
  halts on persistent reward inversion and `tinct ship` refuses to certify any
  run whose final reward margin is non-positive — so a DPO SHIP carries
  cryptographic proof that **chosen > rejected** (`dpo_metrics.json` is folded
  into the signed evidence).
- **V2:** Qwen optimizations (Unsloth, GRPO).
- **V3:** DeepSeek MoE/reasoning.

## License

MIT
