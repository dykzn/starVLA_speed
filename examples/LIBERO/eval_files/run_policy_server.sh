#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/libero_example/checkpoints/steps_50000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-5}"
FAST_INFERENCE="${FAST_INFERENCE:-1}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
  --max_batch_size "${MAX_BATCH_SIZE}"
  --batch_wait_ms "${BATCH_WAIT_MS}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

if [[ "${FAST_INFERENCE}" == "1" ]]; then
  CMD+=(--fast_inference)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
