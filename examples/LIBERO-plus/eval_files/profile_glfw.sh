#!/usr/bin/env bash
set -euo pipefail
STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"
PYTHON=/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python
LIBERO_HOME="${LIBERO_HOME:-/data3/dengyongkang/my_project/LIBERO-plus}"
CKPT="results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt"

# Ensure Xvfb is running
XVFB_DISPLAY=${XVFB_DISPLAY:-:99}
if ! pgrep -x Xvfb >/dev/null; then
  Xvfb ${XVFB_DISPLAY} -screen 0 1024x768x24 &
  sleep 1
fi

export DISPLAY=${XVFB_DISPLAY}
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"
export MUJOCO_GL=glfw
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

declare -A SUITE_GPU=( ["libero_spatial"]=3 ["libero_object"]=4 )
declare -A SUITE_PORT=( ["libero_spatial"]=9895 ["libero_object"]=9896 )

LOG_DIR="results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")_profile_glfw"
mkdir -p "${LOG_DIR}"

echo "=== Starting GLFW servers ==="
SERVER_PIDS=()
for suite in libero_spatial libero_object; do
  gpu=${SUITE_GPU[$suite]}
  port=${SUITE_PORT[$suite]}
  echo "  GPU ${gpu} port ${port} → ${suite}"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" --port "${port}" --use_bf16 > "/tmp/glfw_server_${suite}.log" 2>&1 &
  SERVER_PIDS+=($!)
done

echo "Waiting for servers..."
for suite in libero_spatial libero_object; do
  port=${SUITE_PORT[$suite]}
  for i in $(seq 1 45); do
    if curl -s --max-time 1 "http://127.0.0.1:${port}" >/dev/null 2>&1; then
      echo "  ${suite} ready"
      break
    fi
    sleep 2
  done
done

echo ""
echo "=== Running GLFW profile (5 tasks each) ==="
for suite in libero_spatial libero_object; do
  port=${SUITE_PORT[$suite]}
  gpu=${SUITE_GPU[$suite]}
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} examples/LIBERO-plus/eval_files/eval_libero_profile.py \
    --args.host 127.0.0.1 --args.port "${port}" \
    --args.task-suite-name "${suite}" --args.num-trials-per-task 1 \
    --args.start-idx 0 --args.end-idx 5 \
    --args.video-out-path /tmp/profile_videos_glfw --args.log-path "${LOG_DIR}" \
    > "${LOG_DIR}/${suite}_glfw.log" 2>&1 &
done

echo "Waiting..."
wait

echo "=== Stopping servers ==="
for pid in "${SERVER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done

echo ""
echo "=== GLFW RESULTS ==="
for suite in libero_spatial libero_object; do
  echo ""
  echo "--- ${suite} (GLFW) ---"
  grep -E 'PROFILE|Avg episode|Inference|Env simulation|Preproc|Video|Env create' "${LOG_DIR}/${suite}_glfw.log"
done
echo ""
echo "=== OSMesa (prior) for comparison ==="
echo "  libero_spatial: Env simulation 31.6s (300ms/step)"
echo "  libero_object:  Env simulation 26.9s (164ms/step)"
