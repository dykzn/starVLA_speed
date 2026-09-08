#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluate phase1_wm_warmup_v2 step 6000 on ALL 4 LIBERO suites
# 2 GPUs each run 1 server → 4 eval clients in parallel
# ============================================================

STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

# --- config ---
CKPT="${CKPT:-results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt}"
LIBERO_HOME="${LIBERO_HOME:-/data3/dengyongkang/my_project/LIBERO-plus}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PORT0=9884
PORT1=9883
NUM_TRIALS="${NUM_TRIALS:-1}"

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

LOG_DIR="results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")"
VIDEO_BASE="results/libero_plus_eval/phase1_wm_warmup_v2_step6000"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "  StarVLA LIBERO Full Evaluation"
echo "============================================"
echo "Checkpoint: ${CKPT}"
echo "Log dir:    ${LOG_DIR}"
echo "GPU ${GPU0} → port ${PORT0}: libero_10 + libero_goal"
echo "GPU ${GPU1} → port ${PORT1}: libero_spatial + libero_object"
echo "Trials:     ${NUM_TRIALS}"
echo ""

# --- start servers ---
echo "[server] Starting on GPU ${GPU0} port ${PORT0}..."
CUDA_VISIBLE_DEVICES="${GPU0}" python deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port "${PORT0}" --use_bf16 &
SERVER0_PID=$!

echo "[server] Starting on GPU ${GPU1} port ${PORT1}..."
CUDA_VISIBLE_DEVICES="${GPU1}" python deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port "${PORT1}" --use_bf16 &
SERVER1_PID=$!

# --- wait for servers to be ready ---
echo "[server] Waiting for servers to be ready..."
for port in ${PORT0} ${PORT1}; do
  for i in $(seq 1 60); do
    if curl -s --max-time 1 "http://127.0.0.1:${port}" >/dev/null 2>&1; then
      echo "[server] Port ${port} ready"
      break
    fi
    sleep 2
  done
done

# --- launch evals ---
echo ""
echo "[eval] Launching 4 eval clients..."

CUDA_VISIBLE_DEVICES="" python ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port "${PORT0}" \
  --args.task-suite-name libero_10 --args.num-trials-per-task "${NUM_TRIALS}" \
  --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_10.log" 2>&1 &
PID10=$!

CUDA_VISIBLE_DEVICES="" python ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port "${PORT0}" \
  --args.task-suite-name libero_goal --args.num-trials-per-task "${NUM_TRIALS}" \
  --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_goal.log" 2>&1 &
PID_GOAL=$!

CUDA_VISIBLE_DEVICES="" python ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port "${PORT1}" \
  --args.task-suite-name libero_spatial --args.num-trials-per-task "${NUM_TRIALS}" \
  --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_spatial.log" 2>&1 &
PID_SPATIAL=$!

CUDA_VISIBLE_DEVICES="" python ./examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port "${PORT1}" \
  --args.task-suite-name libero_object --args.num-trials-per-task "${NUM_TRIALS}" \
  --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
  > "${LOG_DIR}/libero_object.log" 2>&1 &
PID_OBJECT=$!

echo "[eval] PIDs: libero_10=${PID10}  libero_goal=${PID_GOAL}  libero_spatial=${PID_SPATIAL}  libero_object=${PID_OBJECT}"

# --- monitor progress ---
echo ""
echo "[monitor] Tailing all logs (Ctrl+C to stop monitoring, evals continue)..."
tail -f "${LOG_DIR}"/*.log 2>/dev/null | grep --line-buffered -E \
    "Starting episode|Success:|Task range|Current total|Total success|ERROR|WARNING|Traceback" &
MONITOR_PID=$!

# --- wait for all evals ---
wait ${PID10} ${PID_GOAL} ${PID_SPATIAL} ${PID_OBJECT}
kill ${MONITOR_PID} 2>/dev/null || true

# --- stop servers ---
echo ""
echo "[server] Stopping servers..."
kill ${SERVER0_PID} ${SERVER1_PID} 2>/dev/null || true

# --- aggregate results ---
echo ""
echo "============================================"
echo "  Results"
echo "============================================"

python -c "
import json, os
log_dir = '${LOG_DIR}'
suites = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
overall = {}
for suite in suites:
    path = os.path.join(log_dir, f'{suite}.json')
    if not os.path.exists(path):
        print(f'WARNING: {suite}.json not found')
        continue
    with open(path) as f:
        data = json.load(f)
    for cat, v in data.items():
        if cat not in overall:
            overall[cat] = {'total_count': 0, 'success_count': 0}
        overall[cat]['total_count'] += v['total_count']
        overall[cat]['success_count'] += v['success_count']

print('')
for cat in sorted(overall.keys()):
    tc = overall[cat]['total_count']
    sc = overall[cat]['success_count']
    rate = sc/tc*100 if tc > 0 else 0
    print(f'  {cat}: {sc}/{tc} = {rate:.1f}%')

total_sc = sum(v['success_count'] for v in overall.values())
total_tc = sum(v['total_count'] for v in overall.values())
print(f'')
print(f'  OVERALL: {total_sc}/{total_tc} = {total_sc/total_tc*100 if total_tc > 0 else 0:.1f}%')
print(f'')
print(f'Logs: {log_dir}')
"
echo ""
echo "=== Done ==="
