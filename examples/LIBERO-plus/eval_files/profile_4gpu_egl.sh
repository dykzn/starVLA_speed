#!/usr/bin/env bash
set -euo pipefail

# Profile with EGL rendering (GPU-accelerated) - compare with OSMesa
STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

PYTHON=/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python
LIBERO_HOME="${LIBERO_HOME:-/data3/dengyongkang/my_project/LIBERO-plus}"
CKPT="results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt"

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

declare -A SUITE_GPU=( ["libero_10"]=0 ["libero_goal"]=1 ["libero_spatial"]=3 ["libero_object"]=4 )
declare -A SUITE_PORT=( ["libero_10"]=9890 ["libero_goal"]=9891 ["libero_spatial"]=9892 ["libero_object"]=9893 )

LOG_DIR="results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")_profile_egl"
mkdir -p "${LOG_DIR}"

echo "=== [EGL] Starting 4 servers ==="
SERVER_PIDS=()
for suite in libero_10 libero_goal libero_spatial libero_object; do
  gpu=${SUITE_GPU[$suite]}
  port=${SUITE_PORT[$suite]}
  echo "  GPU ${gpu} port ${port} → ${suite}"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" --port "${port}" --use_bf16 &
  SERVER_PIDS+=($!)
done

echo "[EGL] Waiting for servers..."
for suite in libero_10 libero_goal libero_spatial libero_object; do
  port=${SUITE_PORT[$suite]}
  for i in $(seq 1 60); do
    if curl -s --max-time 1 "http://127.0.0.1:${port}" >/dev/null 2>&1; then
      echo "  ${suite} (port ${port}) ready"
      break
    fi
    sleep 2
  done
done

echo ""
echo "=== [EGL] Running 4 profiles (5 tasks each) ==="
for suite in libero_10 libero_goal libero_spatial libero_object; do
  port=${SUITE_PORT[$suite]}
  gpu=${SUITE_GPU[$suite]}
  # EGL needs GPU access for rendering; use same GPU as server
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} examples/LIBERO-plus/eval_files/eval_libero_profile.py \
    --args.host 127.0.0.1 --args.port "${port}" \
    --args.task-suite-name "${suite}" --args.num-trials-per-task 1 \
    --args.start-idx 0 --args.end-idx 5 \
    --args.video-out-path /tmp/profile_videos_egl --args.log-path "${LOG_DIR}" \
    > "${LOG_DIR}/${suite}_profile.log" 2>&1 &
done

echo "Waiting for all profiles..."
wait

echo ""
echo "=== Stopping servers ==="
for pid in "${SERVER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done

echo ""
echo "=== [EGL] Results ==="
for suite in libero_10 libero_goal libero_spatial libero_object; do
  echo ""
  echo "--- ${suite} ---"
  grep 'Avg episode time\|Inference\|Env simulation\|Preprocessing\|Video\|Env create' "${LOG_DIR}/${suite}_profile.log" 2>/dev/null || true
  grep '⏱️' "${LOG_DIR}/${suite}_profile.log" 2>/dev/null | tail -5
done

echo ""
echo "Full logs: ${LOG_DIR}"
