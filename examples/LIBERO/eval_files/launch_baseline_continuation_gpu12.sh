#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR="$1"
STARVLA_DIR="/data3/dengyongkang/my_project/starVLA"
SERVER_PYTHON="/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python"
CKPT="${STARVLA_DIR}/results/Checkpoints/starvla_pi_baseline_100k/checkpoints/steps_100000_pytorch_model.pt"
SERVER_SCRIPT="${STARVLA_DIR}/deployment/model_server/server_policy.py"
SHARD_SCRIPT="${STARVLA_DIR}/examples/LIBERO/eval_files/run_egl_shard.sh"
JOBS="${RUN_DIR}/jobs.tsv"
RUN_TAG="$(date '+%m%d%H%M%S')"

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

[[ -f "${JOBS}" ]] || die "jobs file not found: ${JOBS}"
[[ -x "${SERVER_PYTHON}" ]] || die "server Python not executable: ${SERVER_PYTHON}"
[[ -f "${CKPT}" ]] || die "checkpoint not found: ${CKPT}"
[[ -f "${SERVER_SCRIPT}" ]] || die "server script not found: ${SERVER_SCRIPT}"
[[ -x "${SHARD_SCRIPT}" ]] || die "EGL shard script not executable: ${SHARD_SCRIPT}"
[[ ! -e "${RUN_DIR}/launch.started" ]] || die "run was already launched: ${RUN_DIR}"

active_clients="$(docker ps --format '{{.Names}}' | awk '/^starvla_baseline_cont_g[12]_/{print}' || true)"
[[ -z "${active_clients}" ]] || die "continuation clients are already running:\n${active_clients}"
port_is_free 6711 || die "port 6711 is already in use"
port_is_free 6712 || die "port 6712 is already in use"

echo "[$(date '+%F %T')] starting baseline continuation on GPU1/GPU2"
echo "[$(date '+%F %T')] run_dir=${RUN_DIR}; run_tag=${RUN_TAG}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

start_server() {
  local gpu="$1"
  local port="$2"
  local screen_name="$3"
  local log_file="$4"
  local -a server_cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${SERVER_PYTHON}" "${SERVER_SCRIPT}"
    --ckpt_path "${CKPT}" --port "${port}" --use_bf16
    --max_batch_size 8 --batch_wait_ms 5 --fast_inference
  )
  local quoted_cmd
  local screen_cmd
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

start_server 1 6711 "starvla_baseline_cont_gpu1_server_${RUN_TAG}" "${RUN_DIR}/logs/server_gpu1.log"
start_server 2 6712 "starvla_baseline_cont_gpu2_server_${RUN_TAG}" "${RUN_DIR}/logs/server_gpu2.log"
wait_for_server 6711 "${RUN_DIR}/logs/server_gpu1.log"
wait_for_server 6712 "${RUN_DIR}/logs/server_gpu2.log"

launched=0
gpu1_count=0
gpu2_count=0
expected_jobs="$(awk -F'|' 'NR > 1 {n += 1} END {print n + 0}' "${JOBS}")"
expected_gpu1="$(awk -F'|' 'NR > 1 && $2 == 1 {n += 1} END {print n + 0}' "${JOBS}")"
expected_gpu2="$(awk -F'|' 'NR > 1 && $2 == 2 {n += 1} END {print n + 0}' "${JOBS}")"
[[ "${expected_jobs}" -gt 0 ]] || die "jobs file is empty"
[[ "${expected_gpu1}" -ge 1 && "${expected_gpu1}" -le 8 ]] || die "GPU1 job count is ${expected_gpu1}; expected 1-8"
[[ "${expected_gpu2}" -ge 1 && "${expected_gpu2}" -le 8 ]] || die "GPU2 job count is ${expected_gpu2}; expected 1-8"
while IFS='|' read -r variant gpu source_gpu suite shard task_id task_start task_end port output_dir log_file mpl_dir container_name screen_name target_episodes resume_manifest resume_completed resume_successes remaining_episodes; do
  [[ "${variant}" == "variant" ]] && continue
  mkdir -p "${output_dir}" "${mpl_dir}" "$(dirname "${log_file}")"
  command=(bash "${SHARD_SCRIPT}" "${gpu}" "${suite}" "${task_id}" "${task_start}" "${task_end}" "${port}" "${output_dir}" "${log_file}" "${mpl_dir}" "${container_name}" "${variant}" "${resume_manifest}")
  printf -v quoted_command '%q ' "${command[@]}"
  echo "[$(date '+%F %T')] starting client GPU=${gpu} suite=${suite} task=${task_id} resume=${resume_completed}/${target_episodes} screen=${screen_name}"
  screen -dmS "${screen_name}" bash -lc "cd $(printf '%q' "${STARVLA_DIR}"); exec ${quoted_command}"
  launched=$((launched + 1))
  if [[ "${gpu}" == "1" ]]; then
    gpu1_count=$((gpu1_count + 1))
  else
    gpu2_count=$((gpu2_count + 1))
  fi
done < "${JOBS}"

[[ "${launched}" -eq "${expected_jobs}" ]] || die "launched ${launched} clients instead of ${expected_jobs}"
[[ "${gpu1_count}" -eq "${expected_gpu1}" && "${gpu2_count}" -eq "${expected_gpu2}" ]] || die "GPU split is ${gpu1_count}/${gpu2_count}, expected ${expected_gpu1}/${expected_gpu2}"
touch "${RUN_DIR}/launch.started"
sleep 5
echo "[$(date '+%F %T')] launched ${launched} resumed baseline clients"
screen -ls | rg 'starvla_baseline_cont_gpu[12]_server|starvla_baseline_cont_g[12]_' || true
docker ps --format '{{.Names}}\t{{.Status}}' | awk '/^starvla_baseline_cont_g[12]_/{print}' | sort || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo "[$(date '+%F %T')] monitor command:"
echo "  bash ${STARVLA_DIR}/examples/LIBERO/eval_files/monitor_baseline_continuation_gpu12.sh ${RUN_DIR}"
