#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Eval clients only — assumes servers are already running
# Usage: bash run_eval_clients.sh
# ============================================================

STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

LIBERO_HOME="${LIBERO_HOME:-/data3/dengyongkang/my_project/LIBERO-plus}"
PORT0="${PORT0:-9884}"
PORT1="${PORT1:-9883}"
NUM_TRIALS="${NUM_TRIALS:-1}"
VIDEO_BASE="${VIDEO_BASE:-results/libero_plus_eval/phase1_wm_warmup_v2_step6000}"

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

LOG_DIR="results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "  Eval Clients (servers must be running)"
echo "============================================"
echo "Port ${PORT0}: libero_10 + libero_goal"
echo "Port ${PORT1}: libero_spatial + libero_object"
echo "Trials: ${NUM_TRIALS}"
echo "Log:    ${LOG_DIR}"
echo ""

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

echo "Clients launched:"
echo "  libero_10      PID=${PID10}"
echo "  libero_goal    PID=${PID_GOAL}"
echo "  libero_spatial PID=${PID_SPATIAL}"
echo "  libero_object  PID=${PID_OBJECT}"
echo ""

tail -f "${LOG_DIR}"/*.log 2>/dev/null | grep --line-buffered -E \
    "Starting episode|Success:|Task range|Current total|Total success|ERROR|WARNING|Traceback" &
MONITOR_PID=$!

wait ${PID10} ${PID_GOAL} ${PID_SPATIAL} ${PID_OBJECT}
kill ${MONITOR_PID} 2>/dev/null || true

# --- aggregate ---
echo ""
echo "============================================"
echo "  Results"
echo "============================================"
python -c "
import json, os
log_dir = '${LOG_DIR}'
overall = {}
for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
    path = os.path.join(log_dir, f'{suite}.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for cat, v in data.items():
            if cat not in overall:
                overall[cat] = {'total_count': 0, 'success_count': 0}
            overall[cat]['total_count'] += v['total_count']
            overall[cat]['success_count'] += v['success_count']
    else:
        print(f'WARNING: {suite}.json missing')
for cat in sorted(overall.keys()):
    tc = overall[cat]['total_count']
    sc = overall[cat]['success_count']
    rate = sc/tc*100 if tc > 0 else 0
    print(f'  {cat}: {sc}/{tc} = {rate:.1f}%')
total_sc = sum(v['success_count'] for v in overall.values())
total_tc = sum(v['total_count'] for v in overall.values())
print(f'  OVERALL: {total_sc}/{total_tc} = {total_sc/total_tc*100 if total_tc > 0 else 0:.1f}%')
"
echo "=== Done ==="
