#!/usr/bin/env bash
set -euo pipefail

# Profile with Xvfb + GLFW (GPU-accelerated via virtual display)
STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

PYTHON=/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python
LIBERO_HOME="${LIBERO_HOME:-/data3/dengyongkang/my_project/LIBERO-plus}"
CKPT="results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt"

# Start Xvfb
XVFB_DISPLAY=${XVFB_DISPLAY:-:99}
echo "=== Starting Xvfb on ${XVFB_DISPLAY} ==="
Xvfb ${XVFB_DISPLAY} -screen 0 1024x768x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=${XVFB_DISPLAY}

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"
export MUJOCO_GL=glfw
export PYOPENGL_PLATFORM=glfw
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

# Quick test: single suite, 3 tasks
LOG_DIR="results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")_profile_glfw"
mkdir -p "${LOG_DIR}"

echo "=== Starting server on GPU 5 ==="
CUDA_VISIBLE_DEVICES=5 ${PYTHON} deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port 9895 --use_bf16 > /tmp/glfw_server.log 2>&1 &
SERVER_PID=$!

echo "Waiting for server..."
for i in $(seq 1 60); do
  if curl -s --max-time 1 http://127.0.0.1:9895 >/dev/null 2>&1; then
    echo "Server ready"
    break
  fi
  sleep 2
done

echo ""
echo "=== Running GLFW profile (3 tasks, libero_object) ==="
CUDA_VISIBLE_DEVICES=5 ${PYTHON} examples/LIBERO-plus/eval_files/eval_libero_profile.py \
  --args.host 127.0.0.1 --args.port 9895 \
  --args.task-suite-name libero_object --args.num-trials-per-task 1 \
  --args.start-idx 10 --args.end-idx 13 \
  --args.video-out-path /tmp/profile_videos_glfw --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_object_glfw.log" 2>&1

echo ""
echo "=== Results ==="
grep -E 'PROFILE|Avg episode|Inference|Env simulation|Preproc|Video|Env create' "${LOG_DIR}/libero_object_glfw.log"
grep '⏱️' "${LOG_DIR}/libero_object_glfw.log"

# Cleanup
kill ${SERVER_PID} 2>/dev/null || true
kill ${XVFB_PID} 2>/dev/null || true
