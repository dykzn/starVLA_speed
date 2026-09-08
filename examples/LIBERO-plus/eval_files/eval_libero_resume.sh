#!/usr/bin/env bash
set -euo pipefail

# Resume LIBERO-plus eval from checkpoint, distributing remaining tasks across GPUs
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:?required}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
host="${host:-127.0.0.1}"
your_ckpt="${your_ckpt:-examples/LIBERO-plus/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_eval}"
base_port="${base_port:-9883}"
GPUS="${GPUS:-0 1 7}"
num_trials_per_task="${num_trials_per_task:-1}"

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
LOG_DIR="${output_dir}/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

GPU_ARRAY=(${GPUS})
N_GPUS=${#GPU_ARRAY[@]}

# Task totals for each suite (from LIBERO-plus benchmark)
declare -A SUITE_TOTAL=(
  ["libero_goal"]=2591
  ["libero_spatial"]=2402
  ["libero_object"]=2518
  ["libero_10"]=2519
)
declare -A START_IDX

# Find latest log dir to get current progress
LATEST_LOG=$(ls -dt "${output_dir}"/logs/20*/ 2>/dev/null | head -1)
if [[ -z "${LATEST_LOG}" ]]; then
  echo "No previous logs found, starting from scratch"
  for suite in libero_goal libero_spatial libero_object libero_10; do
    START_IDX[$suite]=0
  done
else
  echo "Reading progress from: ${LATEST_LOG}"
  for suite in libero_goal libero_spatial libero_object libero_10; do
    log="${LATEST_LOG}/${suite}.log"
    if [[ -f "$log" ]]; then
      cnt=$(grep -c "Success:" "$log" 2>/dev/null) || cnt=0
      START_IDX[$suite]=$cnt
      echo "  ${suite}: ${cnt}/${SUITE_TOTAL[$suite]} done, $((SUITE_TOTAL[$suite] - cnt)) remaining"
    else
      START_IDX[$suite]=0
      echo "  ${suite}: not started"
    fi
  done
fi

echo ""
echo "=== Launching resume eval across ${N_GPUS} GPUs ==="
for gpu in "${GPU_ARRAY[@]}"; do
  echo "  GPU ${gpu} → port $((base_port + gpu))"
done
echo ""

EVAL_PIDS=()

for suite in libero_goal libero_spatial libero_object libero_10; do
  start=${START_IDX[$suite]}
  total=${SUITE_TOTAL[$suite]}
  remaining=$((total - start))
  if [[ $remaining -le 0 ]]; then
    echo "[${suite}] Already complete, skipping"
    continue
  fi

  # Split remaining tasks across GPUs
  chunk=$((remaining / N_GPUS))
  rem=$((remaining % N_GPUS))

  for i in "${!GPU_ARRAY[@]}"; do
    gpu=${GPU_ARRAY[$i]}
    port=$((base_port + gpu))

    chunk_start=$((start + i * chunk + (i < rem ? i : rem)))
    chunk_size=$((chunk + (i < rem ? 1 : 0)))
    chunk_end=$((chunk_start + chunk_size))
    if [[ $chunk_end -gt $total ]]; then chunk_end=$total; fi
    if [[ $chunk_start -ge $chunk_end ]]; then continue; fi

    video_out_path="${output_dir}/${suite}/${folder_name}"
    log_file="${LOG_DIR}/${suite}_gpu${gpu}.log"

    echo "[${suite}] GPU ${gpu}: tasks [${chunk_start}, ${chunk_end}) → port ${port}"

    MUJOCO_GL="${MUJOCO_GL}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}" CUDA_VISIBLE_DEVICES="" \
    "${LIBERO_PYTHON}" ./examples/LIBERO-plus/eval_files/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-suite-name "${suite}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.start-idx "${chunk_start}" \
        --args.end-idx "${chunk_end}" \
        --args.video-out-path "${video_out_path}" \
        --args.log-path "${LOG_DIR}" \
        > "${log_file}" 2>&1 &
    pid=$!
    EVAL_PIDS+=($pid)
  done
done

echo ""
echo "=== ${#EVAL_PIDS[@]} eval processes running ==="
echo "Log dir: ${LOG_DIR}"

# Monitor
tail -f "${LOG_DIR}"/*.log 2>/dev/null | grep --line-buffered -E \
    "Starting episode|Success:|Current task success|Task range|%\|" &
MONITOR_PID=$!

wait "${EVAL_PIDS[@]}"
kill "${MONITOR_PID}" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Logs: ${LOG_DIR}"
