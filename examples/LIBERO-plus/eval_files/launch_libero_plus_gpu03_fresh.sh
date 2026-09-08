#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="/data3/dengyongkang/my_project/starVLA"
SOURCE_JOBS="${SOURCE_JOBS:-${STARVLA_DIR}/results/libero_plus_prompt_eval_after_prompt_20260907_212304/jobs.tsv}"
RUN_DIR="${RUN_DIR:-${STARVLA_DIR}/results/libero_plus_eval_gpu03_$(date '+%Y%m%d_%H%M%S')}"
RUN_TAG="$(date '+%m%d%H%M%S')"
SERVER_PYTHON="/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python"
CKPT="${STARVLA_DIR}/results/Checkpoints/starvla_pi_baseline_100k/checkpoints/steps_100000_pytorch_model.pt"
SERVER_SCRIPT="${STARVLA_DIR}/deployment/model_server/server_policy.py"
SHARD_SCRIPT="${STARVLA_DIR}/examples/LIBERO-plus/eval_files/run_egl_plus_shard.sh"
SERVER_PORT_G0=6720
SERVER_PORT_G3=6723
SERVER_SCREEN_G0="starvla_plus_neg_gpu0_${RUN_TAG}"
SERVER_SCREEN_G3="starvla_plus_pos_gpu3_${RUN_TAG}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/outputs" "${RUN_DIR}/mplconfig"
exec > >(tee -a "${RUN_DIR}/launcher.log") 2>&1

die() {
  echo "[$(date '+%F %T')] ERROR: $*" >&2
  exit 1
}

port_is_free() {
  local port="$1"
  if exec 3<>"/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
    exec 3>&-
    return 1
  fi
  return 0
}

port_is_ready() {
  local port="$1"
  timeout 2 bash -c ": </dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

echo "[$(date '+%F %T')] fresh LIBERO-plus launch: GPU0=negative, GPU3=positive"
echo "[$(date '+%F %T')] run_dir=${RUN_DIR}; run_tag=${RUN_TAG}"
[[ -f "${SOURCE_JOBS}" ]] || die "source jobs not found: ${SOURCE_JOBS}"
[[ -x "${SERVER_PYTHON}" ]] || die "server Python not executable: ${SERVER_PYTHON}"
[[ -f "${CKPT}" ]] || die "checkpoint not found: ${CKPT}"
[[ -f "${SERVER_SCRIPT}" ]] || die "server script not found: ${SERVER_SCRIPT}"
[[ -f "${SHARD_SCRIPT}" ]] || die "plus shard script not found: ${SHARD_SCRIPT}"
[[ ! -e "${RUN_DIR}/launch.started" ]] || die "run directory already launched: ${RUN_DIR}"

active_clients=$(docker ps --format '{{.Names}}' | awk '/^(starvla_prompt_neg_g0_|starvla_prompt_pos_g3_|starvla_migrate_base_g[03]_|starvla_plus_)/{print}' || true)
[[ -z "${active_clients}" ]] || die "old GPU0/3 clients are still running:\n${active_clients}"
port_is_free "${SERVER_PORT_G0}" || die "port ${SERVER_PORT_G0} is already in use"
port_is_free "${SERVER_PORT_G3}" || die "port ${SERVER_PORT_G3} is already in use"

JOBS="${RUN_DIR}/jobs.tsv"
printf '%s\n' 'variant|gpu|suite|shard|task_start|task_end|port|output_dir|log_file|mpl_dir|container_name|screen_name|target_episodes' > "${JOBS}"
while IFS='|' read -r variant source_gpu suite shard task_start task_end source_port source_output source_log source_mpl source_container source_screen source_target; do
  [[ "${variant}" == "variant" ]] && continue
  [[ "${variant}" == "negative" || "${variant}" == "positive" ]] || continue
  [[ "${source_gpu}" == "0" || "${source_gpu}" == "3" ]] || die "unexpected source GPU ${source_gpu}"
  if [[ "${source_gpu}" == "0" ]]; then
    gpu=0
    port="${SERVER_PORT_G0}"
  else
    gpu=3
    port="${SERVER_PORT_G3}"
  fi
  output_dir="${RUN_DIR}/outputs/${variant}/gpu${gpu}/${suite}/shard${shard}"
  log_file="${RUN_DIR}/logs/${variant}_gpu${gpu}_${suite}_s${shard}.log"
  mpl_dir="${RUN_DIR}/mplconfig/${variant}_gpu${gpu}_${suite}_s${shard}"
  container_name="starvla_plus_${variant}_g${gpu}_${suite}_s${shard}_${RUN_TAG}"
  screen_name="${container_name}"
  target_episodes=$((task_end - task_start))
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${variant}" "${gpu}" "${suite}" "${shard}" "${task_start}" "${task_end}" "${port}" \
    "${output_dir}" "${log_file}" "${mpl_dir}" "${container_name}" "${screen_name}" "${target_episodes}" >> "${JOBS}"
done < "${SOURCE_JOBS}"

job_count=$(awk -F'|' 'NR > 1 {n += 1} END {print n + 0}' "${JOBS}")
target_count=$(awk -F'|' 'NR > 1 {n += $13} END {print n + 0}' "${JOBS}")
target_count_per_variant=$(awk -F'|' 'NR > 1 && $1 == "negative" {n += $13} END {print n + 0}' "${JOBS}")
gpu0_count=$(awk -F'|' 'NR > 1 && $2 == 0 {n += 1} END {print n + 0}' "${JOBS}")
gpu3_count=$(awk -F'|' 'NR > 1 && $2 == 3 {n += 1} END {print n + 0}' "${JOBS}")
[[ "${job_count}" -eq 16 ]] || die "expected 16 shards, found ${job_count}"
[[ "${target_count}" -eq 20060 ]] || die "expected 20060 tasks across both variants, found ${target_count}"
[[ "${target_count_per_variant}" -eq 10030 ]] || die "expected 10030 tasks per variant, found ${target_count_per_variant}"
[[ "${gpu0_count}" -eq 8 && "${gpu3_count}" -eq 8 ]] || die "GPU split is ${gpu0_count}/${gpu3_count}, expected 8/8"

{
  echo "LIBERO-plus fresh evaluation"
  echo "started_at=$(date '+%F %T %Z')"
  echo "negative=GPU0 port=${SERVER_PORT_G0} screen=${SERVER_SCREEN_G0}"
  echo "positive=GPU3 port=${SERVER_PORT_G3} screen=${SERVER_SCREEN_G3}"
  echo "num_trials_per_task=1"
  echo "total_tasks_across_variants=${target_count}"
  echo "total_tasks_per_variant=${target_count_per_variant}"
  echo "jobs=${JOBS}"
  echo "source_jobs=${SOURCE_JOBS}"
  echo "eval_script=${STARVLA_DIR}/examples/LIBERO/eval_files/eval_libero.py"
} > "${RUN_DIR}/RUN_INFO.md"

start_server() {
  local gpu="$1"
  local port="$2"
  local screen_name="$3"
  local log_file="$4"
  local -a server_cmd
  local quoted_cmd
  local screen_cmd
  server_cmd=(env CUDA_VISIBLE_DEVICES="${gpu}" "${SERVER_PYTHON}" "${SERVER_SCRIPT}" --ckpt_path "${CKPT}" --port "${port}" --use_bf16 --max_batch_size 8 --batch_wait_ms 5 --fast_inference)
  printf -v quoted_cmd '%q ' "${server_cmd[@]}"
  screen_cmd="cd $(printf '%q' "${STARVLA_DIR}"); export PYTHONPATH=$(printf '%q' "${STARVLA_DIR}"); export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_MODULE_LOADING=LAZY; ${quoted_cmd} 2>&1 | tee -a $(printf '%q' "${log_file}")"
  echo "[$(date '+%F %T')] starting server GPU=${gpu} port=${port} screen=${screen_name}"
  screen -dmS "${screen_name}" bash -lc "${screen_cmd}"
}

wait_for_server() {
  local port="$1"
  local log_file="$2"
  local attempt
  for attempt in $(seq 1 180); do
    if port_is_ready "${port}"; then
      echo "[$(date '+%F %T')] server port ${port} ready after $((attempt * 2))s"
      return 0
    fi
    sleep 2
  done
  tail -80 "${log_file}" >&2 || true
  die "server port ${port} did not become ready"
}

start_server 0 "${SERVER_PORT_G0}" "${SERVER_SCREEN_G0}" "${RUN_DIR}/logs/server_gpu0_negative.log"
start_server 3 "${SERVER_PORT_G3}" "${SERVER_SCREEN_G3}" "${RUN_DIR}/logs/server_gpu3_positive.log"
wait_for_server "${SERVER_PORT_G0}" "${RUN_DIR}/logs/server_gpu0_negative.log"
wait_for_server "${SERVER_PORT_G3}" "${RUN_DIR}/logs/server_gpu3_positive.log"

launched=0
declare -A launched_gpu=([0]=0 [3]=0)
while IFS='|' read -r variant gpu suite shard task_start task_end port output_dir log_file mpl_dir container_name screen_name target_episodes; do
  [[ "${variant}" == "variant" ]] && continue
  mkdir -p "${output_dir}" "${mpl_dir}" "$(dirname "${log_file}")"
  client_cmd=(bash "${SHARD_SCRIPT}" "${gpu}" "${suite}" "${shard}" "${task_start}" "${task_end}" "${port}" "${output_dir}" "${log_file}" "${mpl_dir}" "${container_name}" "${variant}")
  printf -v quoted_client_cmd '%q ' "${client_cmd[@]}"
  echo "[$(date '+%F %T')] starting client GPU=${gpu} variant=${variant} suite=${suite} shard=${shard} tasks=[${task_start},${task_end}) screen=${screen_name}"
  screen -dmS "${screen_name}" bash -lc "cd $(printf '%q' "${STARVLA_DIR}"); export PLUS_NUM_TRIALS_PER_TASK=1; exec ${quoted_client_cmd}"
  launched=$((launched + 1))
  launched_gpu["${gpu}"]=$((launched_gpu["${gpu}"] + 1))
done < "${JOBS}"

[[ "${launched}" -eq 16 ]] || die "launched ${launched} clients instead of 16"
[[ "${launched_gpu[0]}" -eq 8 && "${launched_gpu[3]}" -eq 8 ]] || die "launched GPU split ${launched_gpu[0]}/${launched_gpu[3]} instead of 8/8"
touch "${RUN_DIR}/launch.started"
sleep 8
echo "[$(date '+%F %T')] launched 16 fresh LIBERO-plus clients"
screen -ls || true
docker ps --format '{{.Names}}\t{{.Status}}' | awk '/^starvla_plus_(negative|positive)_g[03]_/{print}' | sort || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo "[$(date '+%F %T')] monitor command:"
echo "  ${STARVLA_DIR}/examples/LIBERO-plus/eval_files/monitor_libero_plus_gpu03.sh ${RUN_DIR}"
