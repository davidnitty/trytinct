#!/usr/bin/env bash
# tinct V0.1 real-GPU verification — one command on a GPU box.
#
#   ./scripts/gpu_verify.sh [MODEL] [LOSS_THRESHOLD]
#
# Defaults: MODEL=meta-llama/Llama-3.1-8B, THRESHOLD=10.0
# (use meta-llama/Llama-3.2-1B to cut GPU time roughly 8x).
#
# Runs the full certification loop on the REAL target model:
#   init -> validate -> train -> eval (smoke) -> ship -> security check --run
# Exits 0 only if you see SHIP with a verified Ed25519 signature.
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B}"
THRESHOLD="${2:-10.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "== tinct V0.1 GPU verification =="
echo "   model:      $MODEL"
echo "   threshold:  $THRESHOLD"

# --- Preconditions -----------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no GPU detected (nvidia-smi missing)."; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "ERROR: no NVIDIA GPU available."; exit 1; }

if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo "ERROR: no Hugging Face token found."
  echo "  Log in with:  huggingface-cli login   (or set HF_TOKEN)"
  echo "  Required because $MODEL is a gated model."
  exit 1
fi

cd "$(mktemp -d)"   # isolated scratch dir for the verification
echo "   working dir: $(pwd)"

# --- 1. Init ---------------------------------------------------------------
tinct init verify "$MODEL"

# --- 2. Validate the good dataset ------------------------------------------
cd verify
cp "$ROOT/examples/good_data.jsonl" .
tinct validate good_data.jsonl

# --- 3. Train (fail-closed guard, real model) ------------------------------
tinct train --dataset good_data.jsonl --max-loss-threshold "$THRESHOLD"

RUN_ID="$(ls .tinct/runs | head -1)"
echo "== run: $RUN_ID =="

# --- 4. Eval: generation smoke test -----------------------------------------
tinct eval --run "$RUN_ID"

# --- 5. Ship: certification + signed evidence -------------------------------
tinct ship --run "$RUN_ID"

# --- 6. Verify the signature -------------------------------------------------
tinct security check --run "$RUN_ID"

echo
echo "== V0.1 REAL-GPU VERIFICATION: PASS =="
echo "   $MODEL certified and shipped with a valid Ed25519 signature."
