# TryTinct Backend Structure & Security

This document defines the full backend structure for **TryTinct**, a CLI-first post-training stack for LLMs.

TryTinct helps users:

1. Validate and diagnose training data.
2. Auto-select a post-training method.
3. Train models with low-VRAM streaming.
4. Gate every model checkpoint with evals.
5. Produce cryptographic evidence for every shipped model.

The backend must be:

- Local-first
- Secure by default
- Fail-closed
- Reproducible
- Audit-friendly
- Safe against malicious datasets, models, configs, and evals

---

# 1. Product Layers

## 1.1 CLI Layer

User-facing commands.

Examples:

```bash
tinct init
tinct advise
tinct doctor
tinct train
tinct eval
tinct ship
tinct security check
tinct server start
```

---

# 2. Command Reference

This repository implements the TryTinct backend as a **local-first CLI** with a
lightweight core and lazily-loaded ML engines.

| Command | Purpose |
| --- | --- |
| `tinct init` | Scaffold a new project (`tinct.yaml`, directories, signing key). |
| `tinct validate` | Run the **Data Doctor** over an instruction dataset. |
| `tinct advise` | Recommend a post-training method from data + budget (a summary of `validate`). |
| `tinct train` | Fine-tune a Llama adapter with LoRA/QLoRA (low-VRAM streaming). |
| `tinct eval` | Gate the checkpoint against hold-out benchmarks / thresholds. |
| `tinct ship` | Produce the signed SHIP / DON'T-SHIP decision + evidence report. |
| `tinct security check` | Audit project integrity: secrets, hashes, key, permissions. |

## 2.1 Current scope (V0 / Llama)

- **Model family:** Llama (LoRA/QLoRA). Qwen and DeepSeek support are future work.
- **Data:** small instruction datasets in JSON / JSONL.
- **Heavy ML deps** live behind optional extras and are imported only when a
  command needs them (see `tinct.engine.deps`).

# 3. Security Model

- **Local-first:** everything runs on the user's machine; no telemetry.
- **Fail-closed:** validation failures block training; eval failures block shipping.
- **Reproducible:** seeded training, pinned config, full artifact hashing.
- **Audit-friendly:** every step appends to an artifact manifest.
- **Cryptographic evidence:** the ship manifest is signed with an Ed25519 key.

See `src/tinct/security/` for implementation.
