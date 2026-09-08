#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Start 2 policy servers on GPU 0 and GPU 1
# Usage: bash run_servers.sh
# ============================================================

STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

CKPT="${CKPT:-results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PORT0=9884
PORT1=9883

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

echo "Starting servers..."
echo "  GPU ${GPU0} → port ${PORT0}"
echo "  GPU ${GPU1} → port ${PORT1}"

CUDA_VISIBLE_DEVICES="${GPU0}" python deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port "${PORT0}" --use_bf16 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" python deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port "${PORT1}" --use_bf16 &
PID1=$!

echo "Waiting for servers..."
for port in ${PORT0} ${PORT1}; do
  for i in $(seq 1 60); do
    if curl -s --max-time 1 "http://127.0.0.1:${port}" >/dev/null 2>&1; then
      echo "  Port ${port} ready"
      break
    fi
    sleep 3
  done
done

echo "Both servers ready (PIDs: ${PID0} ${PID1})"
wait
