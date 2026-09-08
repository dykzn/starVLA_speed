#!/usr/bin/env bash
set -euo pipefail

# Resume LIBERO-plus eval with GLFW rendering on GPUs 1,3
# Continues from previous OSMesa run progress
STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

PYTHON=/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python
LIBERO_HOME="${LIBERO_HOME:-/data3/dengyongkang/my_project/LIBERO-plus}"
CKPT="${CKPT:-results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt}"
PREV_LOG_DIR="results/libero_plus_eval/logs/20260625_103928"
VIDEO_BASE="${VIDEO_BASE:-results/libero_plus_eval/phase1_wm_warmup_v2_step6000_glfw}"

# GLFW setup
XVFB_DISPLAY=:99
export DISPLAY=${XVFB_DISPLAY}
export MUJOCO_GL=glfw
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

# GPU mapping: 2 GPUs, 2 suites each
# GPU 1 port 9884: libero_10 + libero_goal
# GPU 3 port 9886: libero_spatial + libero_object
declare -A SUITE_PORT=( ["libero_10"]=9884 ["libero_goal"]=9884 ["libero_spatial"]=9886 ["libero_object"]=9886 )

LOG_DIR="results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")_glfw_resume"
mkdir -p "${LOG_DIR}"

# --- read progress ---
echo "=== Reading previous progress ==="
for suite in libero_10 libero_goal libero_spatial libero_object; do
  prev_log="${PREV_LOG_DIR}/${suite}.log"
  if [[ -f "$prev_log" ]]; then
    cnt=$(grep -c "Success:" "$prev_log" 2>/dev/null) || cnt=0
    echo "  ${suite}: ${cnt} done"
  else
    cnt=0
    echo "  ${suite}: no previous log"
  fi
  # Store in file (avoid bash variable naming issues)
  echo "${cnt}" > "/tmp/resume_${suite}.cnt"
done

# --- start Xvfb ---
echo ""
echo "=== Starting Xvfb on ${XVFB_DISPLAY} ==="
pkill Xvfb 2>/dev/null || true; sleep 1
Xvfb ${XVFB_DISPLAY} -screen 0 1024x768x24 &
XVFB_PID=$!
sleep 2

# --- start servers ---
echo ""
echo "=== Starting 2 servers ==="
for gpu in 1 3; do
  port=$((9883 + gpu))
  echo "  GPU ${gpu} → port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" --port "${port}" --use_bf16 > "${LOG_DIR}/server_gpu${gpu}.log" 2>&1 &
done

echo "Waiting for servers..."
for gpu in 1 3; do
  port=$((9883 + gpu))
  for i in $(seq 1 60); do
    if ${PYTHON} -c "
import sys; sys.path.insert(0,'${STARVLA_DIR}')
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
try:
    c = WebsocketClientPolicy('127.0.0.1', ${port}); del c
    sys.exit(0)
except: sys.exit(1)" 2>/dev/null; then
      echo "  GPU ${gpu} port ${port} ready"
      break
    fi
    sleep 2
  done
done

# --- launch eval clients ---
echo ""
echo "=== Launching 4 eval clients (GLFW) ==="

for suite in libero_10 libero_goal libero_spatial libero_object; do
  port=${SUITE_PORT[$suite]}
  start_idx=$(cat "/tmp/resume_${suite}.cnt")
  # GPU for GLFW rendering (same GPU as server handles this suite)
  if [[ "$suite" == "libero_10" || "$suite" == "libero_goal" ]]; then
    render_gpu=1
  else
    render_gpu=3
  fi

  CUDA_VISIBLE_DEVICES="${render_gpu}" ${PYTHON} ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.host 127.0.0.1 --args.port "${port}" \
    --args.task-suite-name "${suite}" --args.num-trials-per-task 1 \
    --args.start-idx "${start_idx}" \
    --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
    > "${LOG_DIR}/${suite}.log" 2>&1 &
  echo "  ${suite}: PID=$! start=${start_idx} GPU=${render_gpu} port=${port}"
done

echo ""
echo "=== Running... ==="
echo "Log dir: ${LOG_DIR}"
echo "Monitor: tail -f ${LOG_DIR}/*.log"
echo ""

# Monitor + wait
tail -f "${LOG_DIR}"/*.log 2>/dev/null | grep --line-buffered -E \
    "Starting episode|Success:|Current total|Total success|ERROR|Traceback|Task range" &
MONITOR_PID=$!

wait
kill ${MONITOR_PID} 2>/dev/null || true

# Stop servers
pkill -f 'server_policy.*--port 988[45]' 2>/dev/null || true

echo ""
echo "=== Done ==="
