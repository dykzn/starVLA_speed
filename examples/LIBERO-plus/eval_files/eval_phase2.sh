#!/usr/bin/env bash
# Phase2 checkpoint evaluation (OSMesa, GPU 0,4)
# 用法: bash examples/LIBERO-plus/eval_files/eval_phase2.sh [CKPT_PATH]
set -euo pipefail

STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

P=/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python
CKPT="${1:-results/Checkpoints/phase2_action_head_finetune/checkpoints/steps_5000_pytorch_model.pt}"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export LIBERO_HOME=/data3/dengyongkang/my_project/LIBERO-plus
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"

# === 1. 清理旧 server ===
pkill -f 'server_policy.*988[78]' 2>/dev/null || true

# === 2. Server ===
CUDA_VISIBLE_DEVICES=0 $P deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 9887 --use_bf16 &
CUDA_VISIBLE_DEVICES=4 $P deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 9888 --use_bf16 &
echo "[1/3] Servers starting (loading model...)"

# === 3. 等 server ready ===
for port in 9887 9888; do
  while ! ss -tlnp 2>/dev/null | grep -q ":${port} "; do sleep 2; done
  echo "       port ${port} ready"
done
echo "[2/3] Servers ready"

# === 4. Client ===
LOG=results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")_phase2_$(basename $CKPT | sed 's/_pytorch_model.pt//')
mkdir -p $LOG

CUDA_VISIBLE_DEVICES="" $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9887 --args.task-suite-name libero_10 \
  --args.num-trials-per-task 1 --args.log-path $LOG > $LOG/libero_10.log 2>&1 &
echo "  libero_10 PID=$!"

CUDA_VISIBLE_DEVICES="" $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9887 --args.task-suite-name libero_goal \
  --args.num-trials-per-task 1 --args.log-path $LOG > $LOG/libero_goal.log 2>&1 &
echo "  libero_goal PID=$!"

CUDA_VISIBLE_DEVICES="" $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9888 --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 --args.log-path $LOG > $LOG/libero_spatial.log 2>&1 &
echo "  libero_spatial PID=$!"

CUDA_VISIBLE_DEVICES="" $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9888 --args.task-suite-name libero_object \
  --args.num-trials-per-task 1 --args.log-path $LOG > $LOG/libero_object.log 2>&1 &
echo "  libero_object PID=$!"

echo "[3/3] Running"
echo "  CKPT: $CKPT"
echo "  GPU 0 port 9887 → libero_10 + libero_goal"
echo "  GPU 4 port 9888 → libero_spatial + libero_object"
echo "  Logs: $LOG"
echo "  Kill: pkill -f 'server_policy.*988[78]'; pkill -f 'eval_libero.*988[78]'"
