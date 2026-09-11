#!/usr/bin/env bash
set -euo pipefail

if (( $# != 11 )); then
  echo "Usage: $0 GPU SUITE SHARD TASK_START TASK_END PORT OUT LOG MPL_DIR CONTAINER_NAME PROMPT_VARIANT" >&2
  exit 2
fi

GPU_ID="$1"
SUITE="$2"
SHARD="$3"
TASK_START="$4"
TASK_END="$5"
PORT="$6"
OUT="$7"
LOG="$8"
MPL_DIR="$9"
CONTAINER_NAME="${10}"
PROMPT_VARIANT="${11}"

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "${STARVLA_DIR}")}"
LIBERO_PLUS_DIR="${LIBERO_PLUS_DIR:-${LIBERO_HOME:-${WORKSPACE_DIR}/LIBERO-plus}}"
LIBERO_CONFIG_PATH="$LIBERO_PLUS_DIR/libero"
LIBERO_HOME="${LIBERO_HOME:-${LIBERO_PLUS_DIR}}"
CKPT="${CKPT:-${STARVLA_DIR}/results/Checkpoints/starvla_pi_baseline_100k/checkpoints/steps_100000_pytorch_model.pt}"
EGL_WRAPPER="${EGL_WRAPPER:-${WORKSPACE_DIR}/egl_docker/run_with_docker_egl.sh}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${STARVLA_DIR}/examples/LIBERO/eval_files/eval_libero.py}"
NUM_TRIALS_PER_TASK="${PLUS_NUM_TRIALS_PER_TASK:-1}"
NOISE_APPLY_INTERVAL="${LIBERO_PLUS_NOISE_INTERVAL:-1}"

[[ -d "${LIBERO_PLUS_DIR}" ]] || { echo "LIBERO_PLUS_DIR does not exist: ${LIBERO_PLUS_DIR}" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "checkpoint does not exist: ${CKPT}" >&2; exit 1; }
[[ -x "${EGL_WRAPPER}" ]] || { echo "EGL_WRAPPER is not executable: ${EGL_WRAPPER}" >&2; exit 1; }
[[ "${NOISE_APPLY_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
  echo "LIBERO_PLUS_NOISE_INTERVAL must be a positive integer: ${NOISE_APPLY_INTERVAL}" >&2
  exit 1
}
command -v "${LIBERO_PYTHON}" >/dev/null 2>&1 || [[ -x "${LIBERO_PYTHON}" ]] || {
  echo "LIBERO_PYTHON is not executable or on PATH: ${LIBERO_PYTHON}" >&2
  exit 1
}

mkdir -p "$OUT" "$MPL_DIR" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] starting LIBERO-plus GPU=$GPU_ID suite=$SUITE shard=$SHARD tasks=[$TASK_START,$TASK_END) trials_per_task=$NUM_TRIALS_PER_TASK prompt_variant=$PROMPT_VARIANT noise_apply_interval=$NOISE_APPLY_INTERVAL"

EGL_GPU="$GPU_ID" \
EGL_DOCKER_NETWORK=host \
EGL_SHM_SIZE=2g \
EGL_CONTAINER_NAME="$CONTAINER_NAME" \
  "$EGL_WRAPPER" -- \
  env \
  LIBERO_HOME="$LIBERO_PLUS_DIR" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  PYTHONPATH="$LIBERO_PLUS_DIR:$STARVLA_DIR" \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  PYTHONSTARTUP= \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  CUDA_MODULE_LOADING=LAZY \
  MPLCONFIGDIR="$MPL_DIR" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  "$LIBERO_PYTHON" "$EVAL_SCRIPT" \
  --args.host 127.0.0.1 \
  --args.port "$PORT" \
  --args.pretrained-path "$CKPT" \
  --args.task-suite-name "$SUITE" \
  --args.task-start "$TASK_START" \
  --args.task-end "$TASK_END" \
  --args.prompt-variant "$PROMPT_VARIANT" \
  --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
  --args.noise-apply-interval "$NOISE_APPLY_INTERVAL" \
  --args.no-save-videos \
  --args.video-out-path "$OUT"

echo "[$(date '+%F %T')] finished LIBERO-plus GPU=$GPU_ID suite=$SUITE shard=$SHARD"
