#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="/data3/dengyongkang/my_project/starVLA"
PYTHON="/data3/dengyongkang/my_project/rlinf-openpi/bin/python"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 2
fi
exec "$PYTHON" "${STARVLA_DIR}/examples/LIBERO-plus/eval_files/monitor_libero_plus_gpu03.py" "$1"
