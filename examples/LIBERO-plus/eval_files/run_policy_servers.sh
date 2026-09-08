#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Start multiple policy servers on different GPUs for parallel eval
# Usage:
#   your_ckpt="path/to/checkpoint.pt" bash run_policy_servers.sh
# ============================================================

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
ABot_python="${ABot_python:-/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python}"
your_ckpt="${your_ckpt:-examples/LIBERO-plus/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
base_port="${base_port:-9883}"
USE_BF16="${USE_BF16:-1}"
ENABLE_WM="${ENABLE_WM:-1}"
GPUS="${GPUS:-0 1 7}"  # GPUs to use

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

echo "=== Starting policy servers ==="
echo "Checkpoint: ${your_ckpt}"
echo "GPUs: ${GPUS}"
echo "Base port: ${base_port}"
echo "World Model: ${ENABLE_WM}"
echo ""

SERVER_PIDS=()

for gpu in ${GPUS}; do
    port=$((base_port + gpu))
    echo "[GPU ${gpu}] Starting server on port ${port}..."

    CMD=(
      "${ABot_python}" deployment/model_server/server_policy.py
      --ckpt_path "${your_ckpt}"
      --port "${port}"
    )

    if [[ "${USE_BF16}" == "1" ]]; then
      CMD+=(--use_bf16)
    fi

    if [[ "${ENABLE_WM}" == "1" ]]; then
      CMD+=(--enable_world_model)
    fi

    # Start server in background, log to file
    LOG_DIR="${STARVLA_DIR}/results/server_logs"
    mkdir -p "${LOG_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}" > "${LOG_DIR}/server_gpu${gpu}_port${port}.log" 2>&1 &
    pid=$!
    SERVER_PIDS+=($pid)
    echo "[GPU ${gpu}] Server PID: ${pid}"
done

echo ""
echo "=== All servers started ==="
echo "PIDs: ${SERVER_PIDS[@]}"
echo ""
echo "To stop all servers: kill ${SERVER_PIDS[@]}"

# Write PID file for cleanup
echo "${SERVER_PIDS[@]}" > /tmp/libero_plus_servers.pid

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "=== Stopping all servers ==="
    for pid in "${SERVER_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null && echo "  Killed PID ${pid}" || true
    done
    rm -f /tmp/libero_plus_servers.pid
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# Wait indefinitely
echo "Servers running. Press Ctrl+C to stop..."
while true; do sleep 60; done
