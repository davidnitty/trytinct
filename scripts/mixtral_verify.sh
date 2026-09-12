#!/usr/bin/env bash
#
# V1.1 FINAL BOSS: Mixtral 8x7B on a single 48GB GPU.
#
#   Phase A (train):  standard 4-bit QLoRA — Mixtral NF4 is ~23GB, fits 48GB.
#   Phase B (verify): certify WITH --offload-experts — the MoEStreamer's
#                     designed use: inference-grade expert streaming.
#
# Requirements: >=48GB VRAM (A6000/A40/L6000/A100), >=128GB system RAM,
# >=200GB disk, HF account with the Mistral license accepted.
set -euo pipefail

MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$HOME/mixtral_verify}"
RUN_NAME="${RUN_NAME:-mixtral_smoke}"
ADAPTER="$PROJECT_DIR/.tinct/runs/$RUN_NAME/adapter"

banner() { echo; echo "=== $* ==="; }

banner "0. Preflight"
command -v nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA GPU visible"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "system RAM: $(free -g | awk '/Mem:/ {print $2}') GB (need >=110 usable)"
echo "disk free  : $(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9') GB (need >=120 after OS)"
[ -n "${HF_TOKEN:-}" ] || huggingface-cli whoami >/dev/null 2>&1 || {
  echo "ERROR: not authenticated with HuggingFace (Mixtral is gated)."
  echo "       export HF_TOKEN=... or run: huggingface-cli login"
  exit 1
}

banner "1. Install tinct (editable) + bitsandbytes (QLoRA training path)"
pip install -e "$REPO_DIR[train]"
pip install bitsandbytes

banner "2. Project"
if [ ! -f "$PROJECT_DIR/.tinct/project.yaml" ]; then
  tinct init "$(basename "$PROJECT_DIR")" "$MODEL" --root "$(dirname "$PROJECT_DIR")"
fi

banner "3. Phase A — train a small adapter (4-bit QLoRA, NO streaming)"
if [ ! -d "$ADAPTER" ]; then
  tinct train --root "$PROJECT_DIR" \
    --dataset "$REPO_DIR/examples/mistral_data.jsonl" \
    --run "$RUN_NAME" \
    --max-loss-threshold 15
else
  echo "adapter already exists: $ADAPTER (skipping training)"
fi

banner "4. Phase B — certify with expert streaming (THE FINAL BOSS)"
set +e
tinct certify \
  --root "$PROJECT_DIR" \
  --adapter "$ADAPTER" \
  --base-model "$MODEL" \
  --offload-experts
CERT_EXIT=$?
set -e

banner "5. Result"
echo "certify exit code: $CERT_EXIT  (0 = SHIP, 2 = DON'T SHIP)"
echo "evidence bundle:"
ls -la "$PROJECT_DIR/.tinct/evidence/" || true
echo
echo "Gate statuses in the signed bundle:"
python - <<PY
import json, pathlib, sys
bundles = sorted(pathlib.Path("$PROJECT_DIR/.tinct/evidence").glob("*_evidence.json"))
if bundles:
    d = json.loads(bundles[-1].read_text(encoding="utf-8"))
    print("decision:", d.get("decision"))
    for k, v in (d.get("safety_gates") or {}).items():
        if isinstance(v, dict) and "status" in v:
            print(f"  {k}: {v['status']}")
    print("adapter sha256:", (d.get("artifacts") or {}).get("adapter_sha256", {}).get("sha256", "?")[:16], "...")
sys.exit(0 if CERT_EXIT == 0 else 1)
PY
