#!/usr/bin/env bash
set -euo pipefail

# Restart the unfinished sr0 suffixes with action-chunk-aware visual noise.
# This launcher keeps the old sr0_split results untouched and writes a new
# run directory so the timing/result comparison is explicit.

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "${STARVLA_DIR}")}"
LIBERO_PLUS_DIR="${LIBERO_PLUS_DIR:-${LIBERO_HOME:-${WORKSPACE_DIR}/LIBERO-plus}}"
LIBERO_HOME="${LIBERO_HOME:-${LIBERO_PLUS_DIR}}"
EGL_WRAPPER="${EGL_WRAPPER:-${WORKSPACE_DIR}/egl_docker/run_with_docker_egl.sh}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
SERVER_PYTHON="${SERVER_PYTHON:-${STARVLA_PYTHON:-python}}"
CKPT="${CKPT:-${STARVLA_DIR}/results/Checkpoints/starvla_pi_baseline_100k/checkpoints/steps_100000_pytorch_model.pt}"
SERVER_SCRIPT="${SERVER_SCRIPT:-${STARVLA_DIR}/deployment/model_server/server_policy.py}"
SHARD_SCRIPT="${SHARD_SCRIPT:-${STARVLA_DIR}/examples/LIBERO-plus/eval_files/run_egl_plus_shard.sh}"
PLUS_NUM_TRIALS_PER_TASK="${PLUS_NUM_TRIALS_PER_TASK:-1}"
NOISE_APPLY_INTERVAL="${NOISE_APPLY_INTERVAL:-8}"
GPU_NEGATIVE="${GPU_NEGATIVE:-0}"
GPU_POSITIVE="${GPU_POSITIVE:-2}"
SERVER_PORT_NEGATIVE="${SERVER_PORT_NEGATIVE:-6720}"
SERVER_PORT_POSITIVE="${SERVER_PORT_POSITIVE:-6722}"
RUN_DIR="${RUN_DIR:-${STARVLA_DIR}/results/libero_plus_sr0_fast_$(date '+%Y%m%d_%H%M%S')}"
RUN_TAG="$(date '+%m%d%H%M%S')"
SERVER_SCREEN_NEGATIVE="${SERVER_SCREEN_NEGATIVE:-starvla_plus_sr0fast_neg_g${GPU_NEGATIVE}_${RUN_TAG}}"
SERVER_SCREEN_POSITIVE="${SERVER_SCREEN_POSITIVE:-starvla_plus_sr0fast_pos_g${GPU_POSITIVE}_${RUN_TAG}}"

die() {
  echo "[$(date '+%F %T')] ERROR: $*" >&2
  exit 1
}

is_executable_command() {
  command -v "$1" >/dev/null 2>&1 || [[ -x "$1" ]]
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
  timeout 2 bash -c ": </dev/tcp/127.0.0.1/${1}" >/dev/null 2>&1
}

screen_exists() {
  screen -ls 2>/dev/null | grep -Eq "[.]${1}[[:space:]]|[.]${1}[[:space:]]+"
}

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/outputs" "${RUN_DIR}/mplconfig"
exec > >(tee -a "${RUN_DIR}/launcher.log") 2>&1

echo "[$(date '+%F %T')] LIBERO-plus sr0 fast launch"
echo "[$(date '+%F %T')] negative=GPU${GPU_NEGATIVE}, positive=GPU${GPU_POSITIVE}"
echo "[$(date '+%F %T')] noise_apply_interval=${NOISE_APPLY_INTERVAL}"
echo "[$(date '+%F %T')] run_dir=${RUN_DIR}"

[[ "${GPU_NEGATIVE}" =~ ^[0-9]+$ && "${GPU_POSITIVE}" =~ ^[0-9]+$ ]] || die "GPU ids must be integers"
[[ "${GPU_NEGATIVE}" != "${GPU_POSITIVE}" ]] || die "negative and positive must use different GPUs"
[[ "${PLUS_NUM_TRIALS_PER_TASK}" == "1" ]] || die "this unfinished run has one trial per task"
[[ "${NOISE_APPLY_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "NOISE_APPLY_INTERVAL must be positive"
[[ -d "${LIBERO_PLUS_DIR}" ]] || die "LIBERO_PLUS_DIR does not exist: ${LIBERO_PLUS_DIR}"
[[ -f "${CKPT}" ]] || die "checkpoint not found: ${CKPT}"
[[ -f "${SERVER_SCRIPT}" ]] || die "server script not found: ${SERVER_SCRIPT}"
[[ -f "${SHARD_SCRIPT}" ]] || die "shard script not found: ${SHARD_SCRIPT}"
[[ -x "${EGL_WRAPPER}" ]] || die "EGL_WRAPPER is not executable: ${EGL_WRAPPER}"
is_executable_command "${SERVER_PYTHON}" || die "server Python is not executable: ${SERVER_PYTHON}"
is_executable_command "${LIBERO_PYTHON}" || die "client Python is not executable: ${LIBERO_PYTHON}"
command -v screen >/dev/null 2>&1 || die "screen is required"
command -v docker >/dev/null 2>&1 || die "docker is required"
[[ ! -e "${RUN_DIR}/launch.started" ]] || die "run directory was already launched: ${RUN_DIR}"

active_old_clients="$(docker ps --format '{{.Names}}' | awk '/^starvla_plus_sr0split_/{print}' || true)"
[[ -z "${active_old_clients}" ]] || die "old sr0_split clients are still running:\n${active_old_clients}"
port_is_free "${SERVER_PORT_NEGATIVE}" || die "port ${SERVER_PORT_NEGATIVE} is already in use"
port_is_free "${SERVER_PORT_POSITIVE}" || die "port ${SERVER_PORT_POSITIVE} is already in use"
screen_exists "${SERVER_SCREEN_NEGATIVE}" && die "screen already exists: ${SERVER_SCREEN_NEGATIVE}"
screen_exists "${SERVER_SCREEN_POSITIVE}" && die "screen already exists: ${SERVER_SCREEN_POSITIVE}"

if [[ -n "${EGL_EXTRA_MOUNTS:-}" ]]; then
  EGL_EXTRA_MOUNTS="${EGL_EXTRA_MOUNTS};${LIBERO_PLUS_DIR}"
else
  EGL_EXTRA_MOUNTS="${LIBERO_PLUS_DIR}"
fi
export STARVLA_DIR LIBERO_PLUS_DIR LIBERO_HOME EGL_WRAPPER LIBERO_PYTHON CKPT
export PLUS_NUM_TRIALS_PER_TASK EGL_EXTRA_MOUNTS
export LIBERO_PLUS_NOISE_INTERVAL="${NOISE_APPLY_INTERVAL}"

JOBS="${RUN_DIR}/jobs.tsv"
printf '%s\n' 'variant|gpu|suite|shard|task_start|task_end|port|output_dir|log_file|mpl_dir|container_name|screen_name|target_episodes' > "${JOBS}"

append_job() {
  local variant="$1"
  local gpu="$2"
  local suite="$3"
  local shard="$4"
  local task_start="$5"
  local task_end="$6"
  local port="$7"
  local output_dir="${RUN_DIR}/outputs/${variant}/gpu${gpu}/${suite}/${shard}"
  local log_file="${RUN_DIR}/logs/${variant}_gpu${gpu}_${suite}_${shard}.log"
  local mpl_dir="${RUN_DIR}/mplconfig/${variant}_gpu${gpu}_${suite}_${shard}"
  local container_name="starvla_plus_sr0fast_${variant}_g${gpu}_${suite}_${shard}_${RUN_TAG}"
  local screen_name="${container_name}"
  local target_episodes=$((task_end - task_start))
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${variant}" "${gpu}" "${suite}" "${shard}" "${task_start}" "${task_end}" "${port}" \
    "${output_dir}" "${log_file}" "${mpl_dir}" "${container_name}" "${screen_name}" "${target_episodes}" >> "${JOBS}"
}

# The exact unfinished sr0 ranges from sr0_split_20260909.
append_job negative "${GPU_NEGATIVE}" libero_spatial a 1406 1647 "${SERVER_PORT_NEGATIVE}"
append_job negative "${GPU_NEGATIVE}" libero_spatial b 1647 1888 "${SERVER_PORT_NEGATIVE}"
append_job negative "${GPU_NEGATIVE}" libero_object  a 1407 1682 "${SERVER_PORT_NEGATIVE}"
append_job negative "${GPU_NEGATIVE}" libero_object  b 1682 1957 "${SERVER_PORT_NEGATIVE}"
append_job negative "${GPU_NEGATIVE}" libero_goal    a 1522 1786 "${SERVER_PORT_NEGATIVE}"
append_job negative "${GPU_NEGATIVE}" libero_goal    b 1786 2050 "${SERVER_PORT_NEGATIVE}"
append_job negative "${GPU_NEGATIVE}" libero_10      a 1492 1747 "${SERVER_PORT_NEGATIVE}"
append_job negative "${GPU_NEGATIVE}" libero_10      b 1747 2002 "${SERVER_PORT_NEGATIVE}"

append_job positive "${GPU_POSITIVE}" libero_spatial a 1403 1645 "${SERVER_PORT_POSITIVE}"
append_job positive "${GPU_POSITIVE}" libero_spatial b 1645 1888 "${SERVER_PORT_POSITIVE}"
append_job positive "${GPU_POSITIVE}" libero_object  a 1406 1681 "${SERVER_PORT_POSITIVE}"
append_job positive "${GPU_POSITIVE}" libero_object  b 1681 1957 "${SERVER_PORT_POSITIVE}"
append_job positive "${GPU_POSITIVE}" libero_goal    a 1521 1785 "${SERVER_PORT_POSITIVE}"
append_job positive "${GPU_POSITIVE}" libero_goal    b 1785 2050 "${SERVER_PORT_POSITIVE}"
append_job positive "${GPU_POSITIVE}" libero_10      a 1492 1747 "${SERVER_PORT_POSITIVE}"
append_job positive "${GPU_POSITIVE}" libero_10      b 1747 2002 "${SERVER_PORT_POSITIVE}"

job_count=$(awk -F'|' 'NR > 1 {n += 1} END {print n + 0}' "${JOBS}")
target_count=$(awk -F'|' 'NR > 1 {n += $13} END {print n + 0}' "${JOBS}")
negative_target=$(awk -F'|' '$1 == "negative" {n += $13} END {print n + 0}' "${JOBS}")
positive_target=$(awk -F'|' '$1 == "positive" {n += $13} END {print n + 0}' "${JOBS}")
[[ "${job_count}" -eq 16 ]] || die "expected 16 clients, found ${job_count}"
[[ "${negative_target}" -eq 2070 && "${positive_target}" -eq 2075 ]] || die "unexpected target counts"
[[ "${target_count}" -eq 4145 ]] || die "expected 4145 episodes, found ${target_count}"

{
  echo "LIBERO-plus sr0 fast evaluation"
  echo "started_at=$(date '+%F %T %Z')"
  echo "source_run=libero_plus_remaining_gpu02_20260908_230959/sr0_split_20260909"
  echo "negative=GPU${GPU_NEGATIVE} port=${SERVER_PORT_NEGATIVE} screen=${SERVER_SCREEN_NEGATIVE}"
  echo "positive=GPU${GPU_POSITIVE} port=${SERVER_PORT_POSITIVE} screen=${SERVER_SCREEN_POSITIVE}"
  echo "noise_apply_interval=${NOISE_APPLY_INTERVAL}"
  echo "expected_action_chunk_size=8"
  echo "libero_plus_dir=${LIBERO_PLUS_DIR}"
  echo "egl_wrapper=${EGL_WRAPPER}"
  echo "egl_extra_mounts=${EGL_EXTRA_MOUNTS}"
  echo "checkpoint=${CKPT}"
  echo "num_trials_per_task=${PLUS_NUM_TRIALS_PER_TASK}"
  echo "remaining_negative=${negative_target}"
  echo "remaining_positive=${positive_target}"
  echo "remaining_total=${target_count}"
  echo "jobs=${JOBS}"
} > "${RUN_DIR}/RUN_INFO.md"

start_server() {
  local gpu="$1"
  local port="$2"
  local screen_name="$3"
  local log_file="$4"
  local -a server_cmd
  local quoted_cmd
  server_cmd=(env CUDA_VISIBLE_DEVICES="${gpu}" "${SERVER_PYTHON}" "${SERVER_SCRIPT}" \
    --ckpt_path "${CKPT}" --port "${port}" --use_bf16 \
    --max_batch_size 8 --batch_wait_ms 5 --fast_inference)
  printf -v quoted_cmd '%q ' "${server_cmd[@]}"
  echo "[$(date '+%F %T')] starting server GPU=${gpu} port=${port} screen=${screen_name}"
  screen -dmS "${screen_name}" bash -lc \
    "cd $(printf '%q' "${STARVLA_DIR}"); export PYTHONPATH=$(printf '%q' "${STARVLA_DIR}"); export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_MODULE_LOADING=LAZY; ${quoted_cmd} 2>&1 | tee -a $(printf '%q' "${log_file}")"
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

start_server "${GPU_NEGATIVE}" "${SERVER_PORT_NEGATIVE}" "${SERVER_SCREEN_NEGATIVE}" "${RUN_DIR}/logs/server_gpu${GPU_NEGATIVE}_negative.log"
start_server "${GPU_POSITIVE}" "${SERVER_PORT_POSITIVE}" "${SERVER_SCREEN_POSITIVE}" "${RUN_DIR}/logs/server_gpu${GPU_POSITIVE}_positive.log"
wait_for_server "${SERVER_PORT_NEGATIVE}" "${RUN_DIR}/logs/server_gpu${GPU_NEGATIVE}_negative.log"
wait_for_server "${SERVER_PORT_POSITIVE}" "${RUN_DIR}/logs/server_gpu${GPU_POSITIVE}_positive.log"

launched=0
declare -A launched_gpu=( ["${GPU_NEGATIVE}"]=0 ["${GPU_POSITIVE}"]=0 )
while IFS='|' read -r variant gpu suite shard task_start task_end port output_dir log_file mpl_dir container_name screen_name target_episodes; do
  [[ "${variant}" == "variant" ]] && continue
  mkdir -p "${output_dir}" "${mpl_dir}" "$(dirname "${log_file}")"
  client_cmd=(bash "${SHARD_SCRIPT}" "${gpu}" "${suite}" "${shard}" "${task_start}" "${task_end}" "${port}" \
    "${output_dir}" "${log_file}" "${mpl_dir}" "${container_name}" "${variant}")
  printf -v quoted_client_cmd '%q ' "${client_cmd[@]}"
  echo "[$(date '+%F %T')] starting client GPU=${gpu} variant=${variant} suite=${suite} shard=${shard} tasks=[${task_start},${task_end}) screen=${screen_name} noise_interval=${NOISE_APPLY_INTERVAL}"
  screen -dmS "${screen_name}" bash -lc \
    "cd $(printf '%q' "${STARVLA_DIR}"); exec ${quoted_client_cmd}"
  launched=$((launched + 1))
  launched_gpu["${gpu}"]=$((launched_gpu["${gpu}"] + 1))
done < "${JOBS}"

[[ "${launched}" -eq 16 ]] || die "launched ${launched} clients instead of 16"
[[ "${launched_gpu[${GPU_NEGATIVE}]}" -eq 8 && "${launched_gpu[${GPU_POSITIVE}]}" -eq 8 ]] || die "client split is not 8/8"
touch "${RUN_DIR}/launch.started"

echo "[$(date '+%F %T')] launched 16 fast sr0 clients: 8 per GPU"
echo "[$(date '+%F %T')] monitor command:"
echo "  python ${STARVLA_DIR}/examples/LIBERO-plus/eval_files/monitor_libero_plus_gpu03.py ${RUN_DIR}"
screen -ls || true
docker ps --format '{{.Names}}\t{{.Status}}' | awk '/^starvla_plus_sr0fast_(negative|positive)_g[0-9]+_/{print}' | sort || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
