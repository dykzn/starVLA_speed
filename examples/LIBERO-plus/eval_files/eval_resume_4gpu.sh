#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Resume LIBERO-plus eval with 4 GPUs (one suite per GPU)
# Reads progress from previous log and continues from there
# ============================================================

STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

# --- config ---
CKPT="${CKPT:-results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt}"
LIBERO_HOME="${LIBERO_HOME:-/data3/dengyongkang/my_project/LIBERO-plus}"
NUM_TRIALS="${NUM_TRIALS:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
PREV_LOG_DIR="${PREV_LOG_DIR:-results/libero_plus_eval/logs/20260625_103928}"
VIDEO_BASE="${VIDEO_BASE:-results/libero_plus_eval/phase1_wm_warmup_v2_step6000}"

# 4 GPUs, each gets its own suite
declare -A SUITE_GPU=(
  ["libero_10"]=0
  ["libero_goal"]=1
  ["libero_spatial"]=3
  ["libero_object"]=4
)
declare -A SUITE_PORT=(
  ["libero_10"]=9884
  ["libero_goal"]=9885
  ["libero_spatial"]=9886
  ["libero_object"]=9887
)
declare -A SUITE_TOTAL=(
  ["libero_10"]=2519
  ["libero_goal"]=2591
  ["libero_spatial"]=2402
  ["libero_object"]=2518
)

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

LOG_DIR="results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "  Resume LIBERO-plus Eval (4 GPU)"
echo "============================================"
echo "Checkpoint:     ${CKPT}"
echo "Previous logs:  ${PREV_LOG_DIR}"
echo "New log dir:    ${LOG_DIR}"
echo ""

# --- read progress from previous logs ---
declare -A START_IDX

for suite in libero_10 libero_goal libero_spatial libero_object; do
  prev_log="${PREV_LOG_DIR}/${suite}.log"
  if [[ -f "$prev_log" ]]; then
    cnt=$(grep -c "Success:" "$prev_log" 2>/dev/null) || cnt=0
    START_IDX[$suite]=$cnt
    remaining=$((SUITE_TOTAL[$suite] - cnt))
    echo "  ${suite}: ${cnt}/${SUITE_TOTAL[$suite]} done → ${remaining} remaining (GPU ${SUITE_GPU[$suite]})"
  else
    START_IDX[$suite]=0
    echo "  ${suite}: no previous log, starting from 0"
  fi
done

# --- start 4 servers (one per GPU) ---
echo ""
echo "[server] Starting 4 policy servers..."

SERVER_PIDS=()
for suite in libero_10 libero_goal libero_spatial libero_object; do
  gpu=${SUITE_GPU[$suite]}
  port=${SUITE_PORT[$suite]}
  echo "[server] GPU ${gpu} port ${port} for ${suite}..."
  CUDA_VISIBLE_DEVICES="${gpu}" python deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" --port "${port}" --use_bf16 &
  SERVER_PIDS+=($!)
done

# --- wait for all servers to be ready ---
echo "[server] Waiting for servers to be ready..."
for suite in libero_10 libero_goal libero_spatial libero_object; do
  port=${SUITE_PORT[$suite]}
  for i in $(seq 1 60); do
    if curl -s --max-time 1 "http://127.0.0.1:${port}" >/dev/null 2>&1; then
      echo "[server] Port ${port} (${suite}) ready"
      break
    fi
    sleep 2
  done
done

# --- launch 4 eval clients ---
echo ""
echo "[eval] Launching 4 eval clients..."

EVAL_PIDS=()

for suite in libero_10 libero_goal libero_spatial libero_object; do
  start=${START_IDX[$suite]}
  total=${SUITE_TOTAL[$suite]}
  if [[ $start -ge $total ]]; then
    echo "[${suite}] Already complete (${start}/${total}), skipping"
    continue
  fi
  port=${SUITE_PORT[$suite]}

  CUDA_VISIBLE_DEVICES="" python ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.host 127.0.0.1 --args.port "${port}" \
    --args.task-suite-name "${suite}" --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.start-idx "${start}" \
    --args.video-out-path "${VIDEO_BASE}" --args.log-path "${LOG_DIR}" \
    > "${LOG_DIR}/${suite}.log" 2>&1 &
  EVAL_PIDS+=($!)
  echo "[${suite}] PID=$!  start=${start}  port=${port}"
done

echo ""
echo "[monitor] Tailing all logs (Ctrl+C to stop monitoring, evals continue in background)..."
echo "           Kill all:  kill ${EVAL_PIDS[*]} ${SERVER_PIDS[*]}"
echo ""

tail -f "${LOG_DIR}"/*.log 2>/dev/null | grep --line-buffered -E \
    "Starting episode|Success:|Current total|Total success|ERROR|WARNING|Traceback|Task range|%\|" &
MONITOR_PID=$!

# --- wait for all evals ---
wait "${EVAL_PIDS[@]}"
kill "${MONITOR_PID}" 2>/dev/null || true

# --- stop servers ---
echo ""
echo "[server] Stopping servers..."
for pid in "${SERVER_PIDS[@]}"; do
  kill "$pid" 2>/dev/null || true
done

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
