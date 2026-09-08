#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Parallel LIBERO-plus evaluation across multiple GPUs
# Each suite runs against a different server port
#
# Usage:
#   LIBERO_HOME=/path/to/LIBERO-plus \
#   your_ckpt="path/to/checkpoint.pt" \
#   bash eval_libero_parallel.sh
# ============================================================

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

host="${host:-127.0.0.1}"
your_ckpt="${your_ckpt:-examples/LIBERO-plus/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_eval}"
base_port="${base_port:-9883}"
GPUS="${GPUS:-0 1 7}"
num_trials_per_task="${num_trials_per_task:-1}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "ERROR: LIBERO_HOME is required."
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
LOG_DIR="${output_dir}/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

# Custom suite→GPU mapping (can override via env var)
# Format: "suite1:gpu1 suite2:gpu2 ..."
SUITE_GPU_MAP="${SUITE_GPU_MAP:-}"
GPU_ARRAY=(${GPUS})

declare -A SUITE_PORT
if [[ -n "${SUITE_GPU_MAP}" ]]; then
    # Explicit mapping
    for pair in ${SUITE_GPU_MAP}; do
        suite="${pair%%:*}"
        gpu="${pair##*:}"
        SUITE_PORT[$suite]=$((base_port + gpu))
    done
else
    # Default: auto-distribute round-robin
    SUITES=(libero_goal libero_spatial libero_object libero_10)
    for i in "${!SUITES[@]}"; do
        gpu_idx=$((i % ${#GPU_ARRAY[@]}))
        gpu=${GPU_ARRAY[$gpu_idx]}
        SUITE_PORT[${SUITES[$i]}]=$((base_port + gpu))
    done
fi
SUITES=(libero_goal libero_spatial libero_object libero_10)

echo "=== LIBERO-plus Parallel Evaluation ==="
echo "Output dir: ${output_dir}"
echo "Log dir: ${LOG_DIR}"
echo "Trials per task: ${num_trials_per_task}"
echo "Host: ${host}"
echo ""

for idx in "${!SUITES[@]}"; do
    suite=${SUITES[$idx]}
    gpu=${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}
    echo "  ${suite} → port ${SUITE_PORT[$suite]} (GPU ${gpu})"
done
echo ""

EVAL_PIDS=()

for suite in "${SUITES[@]}"; do
    port=${SUITE_PORT[$suite]}
    video_out_path="${output_dir}/${suite}/${folder_name}"
    log_file="${LOG_DIR}/${suite}.log"

    echo "[${suite}] Starting eval on port ${port}..."

    MUJOCO_GL="${MUJOCO_GL}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}" CUDA_VISIBLE_DEVICES="" "${LIBERO_PYTHON}" ./examples/LIBERO-plus/eval_files/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-suite-name "${suite}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.video-out-path "${video_out_path}" \
        --args.log-path "${LOG_DIR}" \
        > "${log_file}" 2>&1 &
    pid=$!
    EVAL_PIDS+=($pid)
    echo "[${suite}] PID: ${pid}"
done

echo ""
echo "=== All evals running ==="
echo "PIDs: ${EVAL_PIDS[@]}"
echo ""
echo "--- Live progress (Ctrl+C to stop monitoring, evals continue in background) ---"
echo ""

# Monitor progress from log files in real-time
tail -f "${LOG_DIR}"/*.log 2>/dev/null | grep --line-buffered -E \
    "Starting episode|Success:|Current task success|Total success|Task:|%\|.*\|" &
MONITOR_PID=$!

# Wait for all evals to finish
wait "${EVAL_PIDS[@]}"

# Kill the monitor
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

# =============== Aggregate results ===============
echo ""
echo "=== Aggregating results ==="

cd "${STARVLA_DIR}"
SUITE_FILES=""
for suite in libero_10 libero_goal libero_object libero_spatial; do
    SUITE_FILES="${SUITE_FILES} ${LOG_DIR}/${suite}.json"
done

"${LIBERO_PYTHON}" -c "
import json, os

log_dir = '${LOG_DIR}'
task_suites = ['libero_10.json', 'libero_goal.json', 'libero_object.json', 'libero_spatial.json']
overall = {'overall': {'total_count': 0, 'success_count': 0}}

for suite in task_suites:
    path = os.path.join(log_dir, suite)
    if not os.path.exists(path):
        print(f'WARNING: {path} not found, skipping')
        continue
    with open(path) as f:
        results = json.load(f)
    for category in results:
        overall['overall']['total_count'] += results[category]['total_count']
        overall['overall']['success_count'] += results[category]['success_count']
        if category not in overall:
            overall[category] = results[category]
        else:
            overall[category]['total_count'] += results[category]['total_count']
            overall[category]['success_count'] += results[category]['success_count']

for category in overall:
    tc = overall[category]['total_count']
    sc = overall[category]['success_count']
    overall[category]['success_rate'] = round(sc / tc * 100, 1) if tc > 0 else 0.0
    print(f\"  {category}: {sc}/{tc} = {overall[category]['success_rate']}%\")

with open(os.path.join(log_dir, 'overall_results.json'), 'w') as f:
    json.dump(overall, f, indent=2)
print(f'\nResults saved to {log_dir}/overall_results.json')
"

echo ""
echo "=== Done ==="
echo "Results: ${LOG_DIR}/overall_results.json"
