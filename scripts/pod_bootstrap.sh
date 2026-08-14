#!/usr/bin/env bash
# One-shot bootstrap for a fresh GPU pod (RunPod / Vast.ai / any NVIDIA box).
#
#   HF_TOKEN=hf_... bash scripts/pod_bootstrap.sh [MODEL] [LOSS_THRESHOLD]
#
# Defaults: MODEL=meta-llama/Llama-3.2-1B, THRESHOLD=10.0
# (Llama-3.1-8B also supported; 3.2-1B is ~8x faster and costs well under $1.)
#
# Requires: NVIDIA GPU + Python 3.11+ preinstalled on the pod image.
# Steps: clone tinct -> install [train] -> verify GPU/token -> run full
#        init/validate/train/eval/ship/sign loop -> PASS on verified SHIP.
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.2-1B}"
THRESHOLD="${2:-10.0}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "  Get one: huggingface.co -> Settings -> Access Tokens -> New token (Read)"
  echo "  Then:    HF_TOKEN=hf_... bash $0 $MODEL $THRESHOLD"
  exit 1
fi
export HF_TOKEN

echo "== tinct pod bootstrap =="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | head -1 || { echo "ERROR: no NVIDIA GPU."; exit 1; }
python3 --version

git clone --depth 1 https://github.com/davidnitty/trytinct.git /tmp/tinct
cd /tmp/tinct
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[train]"

echo "== running verification =="
bash scripts/gpu_verify.sh "$MODEL" "$THRESHOLD"
