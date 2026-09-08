#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

PYTHON=/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python
LIBERO_HOME=/data3/dengyongkang/my_project/LIBERO-plus
CKPT="${CKPT:-results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt}"
PREV_LOG_DIR="${PREV_LOG_DIR:-results/libero_plus_eval/logs/20260625_103928}"
VIDEO_BASE="${VIDEO_BASE:-results/libero_plus_eval/phase1_wm_warmup_v2_step6000_glfw}"

export DISPLAY=:99
export MUJOCO_GL=glfw
export LIBERO_HOME=/data3/dengyongkang/my_project/LIBERO-plus
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

# === cleanup ===
pkill -f 'server_policy.*--port 988[46]' 2>/dev/null || true
pkill Xvfb 2>/dev/null || true
sleep 1

# === Xvfb ===
Xvfb :99 -screen 0 1024x768x24 &
sleep 1
echo "[ok] Xvfb started on :99"

# === 2 servers ===
CUDA_VISIBLE_DEVICES=1 ${PYTHON} deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port 9884 --use_bf16 > /tmp/srv_9884.log 2>&1 &
echo "[server] GPU 1 port 9884 PID $!"

CUDA_VISIBLE_DEVICES=3 ${PYTHON} deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port 9886 --use_bf16 > /tmp/srv_9886.log 2>&1 &
echo "[server] GPU 3 port 9886 PID $!"

# === wait for servers ===
echo -n "[wait] "
for port in 9884 9886; do
  for i in $(seq 1 90); do
    if ${PYTHON} -c "
import sys; sys.path.insert(0,'${STARVLA_DIR}')
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
try:
    c=WebsocketClientPolicy('127.0.0.1',${port}); del c; sys.exit(0)
except: sys.exit(1)" 2>/dev/null; then
      echo -n "port${port} "
      break
    fi
    sleep 2
  done
done
echo "ready!"

# === read previous progress ===
for suite in libero_10 libero_goal libero_spatial libero_object; do
  prev_log="${PREV_LOG_DIR}/${suite}.log"
  cnt=0
  [[ -f "$prev_log" ]] && cnt=$(grep -c "Success:" "$prev_log" 2>/dev/null) || true
  echo "${cnt}" > "/tmp/resume_glfw_${suite}.cnt"
  echo "[progress] ${suite}: ${cnt}"
done

# === 4 eval clients ===
LOG_DIR="results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")_glfw"
mkdir -p "${LOG_DIR}"
echo ""
echo "=== Starting eval clients ==="

# GPU 1 port 9884
CUDA_VISIBLE_DEVICES=1 ${PYTHON} ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9884 \
  --args.task-suite-name libero_10 --args.num-trials-per-task 1 \
  --args.start-idx $(cat /tmp/resume_glfw_libero_10.cnt) \
  --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_10.log" 2>&1 &
echo "  libero_10:    PID $!  GPU=1 port=9884 start=$(cat /tmp/resume_glfw_libero_10.cnt)"

CUDA_VISIBLE_DEVICES=1 ${PYTHON} ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9884 \
  --args.task-suite-name libero_goal --args.num-trials-per-task 1 \
  --args.start-idx $(cat /tmp/resume_glfw_libero_goal.cnt) \
  --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_goal.log" 2>&1 &
echo "  libero_goal:  PID $!  GPU=1 port=9884 start=$(cat /tmp/resume_glfw_libero_goal.cnt)"

# GPU 3 port 9886
CUDA_VISIBLE_DEVICES=3 ${PYTHON} ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9886 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 1 \
  --args.start-idx $(cat /tmp/resume_glfw_libero_spatial.cnt) \
  --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_spatial.log" 2>&1 &
echo "  libero_spatial: PID $!  GPU=3 port=9886 start=$(cat /tmp/resume_glfw_libero_spatial.cnt)"

CUDA_VISIBLE_DEVICES=3 ${PYTHON} ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9886 \
  --args.task-suite-name libero_object --args.num-trials-per-task 1 \
  --args.start-idx $(cat /tmp/resume_glfw_libero_object.cnt) \
  --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_object.log" 2>&1 &
echo "  libero_object: PID $!  GPU=3 port=9886 start=$(cat /tmp/resume_glfw_libero_object.cnt)"

echo ""
echo "Log dir: ${LOG_DIR}"
echo "Monitor: tail -f ${LOG_DIR}/*.log"
echo "Kill all: pkill -f 'server_policy.*988[46]'; pkill -f 'eval_libero.*988[46]'; pkill Xvfb"

wait
echo "[done] All eval clients finished"
